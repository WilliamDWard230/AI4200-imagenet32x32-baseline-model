import torch
import torch.nn as nn
import torch.nn.functional as F

# Each task differs only in its input size and its number of classes.
# The same model class is used for all three.
TASKS = {
    "mnist":      {"image_size": 28, "channels": 1, "num_classes": 10, "patch_size": 4},    # 28x28 grayscale
    "cifar10":    {"image_size": 32, "channels": 3, "num_classes": 10, "patch_size": 8},    # 32x32 color
    "imagenet32": {"image_size": 32, "channels": 3, "num_classes": 1000, "patch_size": 8},  # 32x32 color
}

beta = 0.5


class Block(nn.Module):
    """One residual block:  x + Linear(tanh(Linear(LayerNorm(x))))

    The block expands to `hidden` features, applies tanh, and projects back to
    `width` features so its output can be added to its input (the skip connection).
    """

    def __init__(self, width, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(width)       # pre-norm: normalize before the layers
        self.fc1 = nn.Linear(width, hidden)   # expand:  width  -> hidden
        self.fc2 = nn.Linear(hidden, width)   # project: hidden -> width
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        h = F.silu(beta * self.fc1(self.norm(x)))
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h                          # skip connection: add the input back

class PatchMixingBlock(nn.Module):

    def __init__(self, num_patches, width, hidden = 128):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(num_patches, hidden)
        self.fc2 = nn.Linear(hidden, num_patches)

    def forward(self, x):
        h = self.norm(x).transpose(1,2)
        h = self.fc2(F.silu(beta * self.fc1(h)))
        return x + h.transpose(1,2)
    
    
class CompositeBlock(nn.Module):

    def __init__(self, num_patches, width, hidden, patch_hidden):
            super().__init__()
            self.patch_mixing = PatchMixingBlock(num_patches, width, patch_hidden)
            self.feature_mixing = Block(width, hidden)

    def forward(self, x):
        return self.feature_mixing(self.patch_mixing(x))



class ResidualMLP(nn.Module):
    """Baseline residual MLP.

    input -> Linear(in_features -> width)            project into the backbone
          -> depth x Block                           residual backbone, stays `width` wide
          -> LayerNorm -> Linear -> tanh -> Linear   head: a block without the skip
          -> logits (num_classes)

    The head has no skip connection because its output size (num_classes)
    differs from its input size (width), so the two cannot be added.
    """

    def __init__(self, image_size, channels, num_classes, width=128, hidden=512, depth=15, patch_size = 4, patch_hidden = 128):
        super().__init__()

        self.patch_size = patch_size
        self.channels = channels
        self.image_size = image_size

        self.num_patches = (self.image_size // patch_size) ** 2
        self.patch_features = self.channels * patch_size ** 2
        self.input_layer = nn.Linear(self.patch_features, width)
        self.blocks = nn.ModuleList([
            CompositeBlock(self.num_patches, width, hidden, patch_hidden)
            for _ in range(depth)
        ])

        # The head: same layers as a block, but the last layer outputs class scores.
        self.head_norm = nn.LayerNorm(width)
        self.head_fc1 = nn.Linear(width, hidden)
        self.head_fc2 = nn.Linear(hidden, num_classes)

        # Xavier (Glorot) initialization for every linear layer, zero biases.
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """Training forward pass. Returns raw logits (no softmax).

        nn.CrossEntropyLoss applies the softmax itself, so training uses logits.
        """
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(x.shape[0], self.num_patches, self.patch_features)
        x = self.input_layer(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.head_norm(x).mean(dim=1)  # global mean across patches
        return self.head_fc2(F.silu(beta * self.head_fc1(x)))

    def predict(self, x):
        """Inference pass, never used during training. Returns class probabilities."""
        self.eval()                           # switch layers to inference behavior
        with torch.no_grad():                 # no gradients needed at inference
            return torch.softmax(self.forward(x), dim=1)
