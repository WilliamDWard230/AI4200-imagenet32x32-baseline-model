import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from .model import TASKS, ResidualMLP


# Baseline settings. Deliberately plain so they are easy to beat:
# plain SGD, constant learning rate, no dropout, no schedule, no augmentation.
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
EPOCHS = {"mnist": 10, "cifar10": 30, "imagenet32": 10}


def prepare(X):
    """Convert uint8 images (0-255) to floats in roughly [-1, 1]."""
    return (torch.as_tensor(X).float() / 255.0 - 0.5) / 0.5


# ---------------------------------------------------------------------------
# The two functions below are the REQUIRED INTERFACE for every submission.
# The grading notebook calls them like this:
#
#     model = train("cifar10", X_train, y_train, seed=0, device="cuda")
#     probs = predict(model, X_test)
#
# Change anything inside them, but keep their names and arguments.
# ---------------------------------------------------------------------------

def sobel_filter(X):
    """
    Apply Sobel edge detection independently to RGB channels.
    X shape: (N, C, H, W), normalized to [-1, 1].
    """

    channels = X.shape[1]

    sobel_x = torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ], dtype=X.dtype, device=X.device)

    sobel_y = torch.tensor([
        [-1., -2., -1.],
        [ 0.,  0.,  0.],
        [ 1.,  2.,  1.]
    ], dtype=X.dtype, device=X.device)

    kernel_x = sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    kernel_y = sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)

    gx = F.conv2d(X, kernel_x, padding=1, groups=channels)
    gy = F.conv2d(X, kernel_y, padding=1, groups=channels)

    magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)

    # Normalize each image's edges into [0, 1]
    maximum = magnitude.amax(dim=(1, 2, 3), keepdim=True)
    magnitude = magnitude / maximum.clamp_min(1e-8)

    # Convert to the model's expected [-1, 1] range
    return magnitude * 2 - 1


def train(task, X_train, y_train, seed=0, device="cuda"):
    """Build a new model and train it from scratch. Returns the trained model."""
    torch.manual_seed(seed)                   # makes the run reproducible

    # 1. Build the model for this task. Only input size and class count differ.
    model = ResidualMLP(**TASKS[task]).to(device)

    # 2. Wrap the data so we can loop over shuffled mini-batches. (Added gaussian noise data aug)
    # Prepare original training images
    X = prepare(X_train)
    y = torch.as_tensor(y_train).long()

    # Sobel augmentation settings
    SOBEL_FRACTION = 0.10

    generator = torch.Generator().manual_seed(seed)

    # Randomly select 10% of training images
    n_sobel = int(len(X) * SOBEL_FRACTION)

    indices = torch.randperm(
        len(X),
        generator=generator
    )[:n_sobel]

    # Create Sobel-filtered copies
    X_sobel = sobel_filter(X[indices].clone())
    y_sobel = y[indices].clone()

    # Keep every original image and append Sobel copies
    X = torch.cat([X, X_sobel], dim=0)
    y = torch.cat([y, y_sobel], dim=0)

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print(f"Original images: {len(X_train)}")
    print(f"Sobel copies added: {n_sobel}")
    print(f"Total training images: {len(dataset)}")

    # 3. Choose the loss and the optimizer.
    loss_fn = nn.CrossEntropyLoss()           # expects raw logits
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 4. The training loop.
    model.train()
    for epoch in range(EPOCHS[task]):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)           # forward pass
            loss = loss_fn(logits, y_batch)   # how wrong are we?

            optimizer.zero_grad()             # clear old gradients
            loss.backward()                   # backward pass: compute gradients
            optimizer.step()                  # update the weights

            total_loss += loss.item() * len(X_batch)

        print(f"epoch {epoch + 1}: train loss {total_loss / len(dataset):.4f}")

    return model


def predict(model, X):
    """Return class probabilities for X, one row per example, on the CPU."""
    device = next(model.parameters()).device
    X = prepare(X)
    # Predict in chunks so a large test set does not run out of GPU memory.
    chunks = [model.predict(X[i:i + 1024].to(device)).cpu() for i in range(0, len(X), 1024)]
    return torch.cat(chunks)
