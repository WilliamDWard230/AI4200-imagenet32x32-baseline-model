import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import math

from .model import TASKS, ResidualMLP


# Baseline settings. Deliberately plain so they are easy to beat:
# plain SGD, constant learning rate, no dropout, no schedule, no augmentation.
LEARNING_RATE = 7e-4
BATCH_SIZE = 256
EPOCHS = {"mnist": 10, "cifar10": 100, "imagenet32": 10}


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





class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, augment=False, seed=0):
        self.X = X
        self.y = y
        self.augment = augment
        self.seed = seed

        self.n_original = len(X)
        self.n_horizontal = int(self.n_original * 0.15) if augment else 0
        self.n_erased = int(self.n_original * 0.15) if augment else 0
        self.n_rotated = int(self.n_original * 0.15) if augment else 0
        self.n_brightness = int(self.n_original * 0.15) if augment else 0
        self.n_shifted = int(self.n_original * 0.15) if augment else 0
        self.n_contrast = int(self.n_original * 0.15) if augment else 0

    def set_epoch(self, epoch):
        if self.augment:
            generator = torch.Generator().manual_seed(self.seed + epoch)

            self.horizontal_indices = torch.randperm(
                self.n_original, generator=generator
            )[:self.n_horizontal]

            self.erased_indices = torch.randperm(
                self.n_original, generator=generator
            )[:self.n_erased]

            self.rotated_indices = torch.randperm(
                self.n_original, generator=generator
            )[:self.n_rotated]

            self.brightness_indices = torch.randperm(
                self.n_original, generator=generator
            )[:self.n_brightness]

            self.shifted_indices = torch.randperm(
                self.n_original, generator=generator
            )[:self.n_shifted]

            self.contrast_indices = torch.randperm(
                self.n_original, generator=generator
            )[:self.n_contrast]

    def __len__(self):
        return (
            self.n_original
            + self.n_horizontal
            + self.n_erased
            + self.n_rotated
            + self.n_brightness
            + self.n_shifted
            + self.n_contrast
        )

    def __getitem__(self, idx):
        if idx < self.n_original:
            x = self.X[idx]
            y = self.y[idx]

        elif idx < self.n_original + self.n_horizontal:
            original_idx = self.horizontal_indices[
                idx - self.n_original
            ]

            x = torch.flip(self.X[original_idx], dims=[2])
            y = self.y[original_idx]

        elif idx < (
            self.n_original + self.n_horizontal + self.n_erased
        ):
            original_idx = self.erased_indices[
                idx - self.n_original - self.n_horizontal
            ]

            x = self.X[original_idx].clone()
            y = self.y[original_idx]

            C, H, W = x.shape

            # Erase approximately 2-8% of image area
            area_fraction = 0.02 + 0.06 * torch.rand(1).item()
            side = max(1, round((H * W * area_fraction) ** 0.5))
            side = min(side, H, W)

            top = torch.randint(
                0, H - side + 1, (1,)
            ).item()

            left = torch.randint(
                0, W - side + 1, (1,)
            ).item()

            # Fill erased region with gray
            x[:, top:top + side, left:left + side] = 127

        elif idx < (
            self.n_original
            + self.n_horizontal
            + self.n_erased
            + self.n_rotated
        ):
            original_idx = self.rotated_indices[
                idx
                - self.n_original
                - self.n_horizontal
                - self.n_erased
            ]

            x = self.X[original_idx].float()
            y = self.y[original_idx]

            # Random rotation between -10 and +10 degrees
            angle = 20.0 * torch.rand(1).item() - 10.0
            radians = math.radians(angle)

            cos_a = math.cos(radians)
            sin_a = math.sin(radians)

            theta = torch.tensor(
                [[
                    [cos_a, -sin_a, 0.0],
                    [sin_a,  cos_a, 0.0]
                ]],
                dtype=torch.float32
            )

            x = x.unsqueeze(0)

            grid = F.affine_grid(
                theta,
                x.size(),
                align_corners=False
            )

            # Shift pixel values so padding becomes gray
            x = x - 127.0

            x = F.grid_sample(
                x,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False
            )

            x = (x + 127.0).squeeze(0)

        elif idx < (
            self.n_original
            + self.n_horizontal
            + self.n_erased
            + self.n_rotated
            + self.n_brightness
        ):
            original_idx = self.brightness_indices[
                idx
                - self.n_original
                - self.n_horizontal
                - self.n_erased
                - self.n_rotated
            ]

            x = self.X[original_idx].float()
            y = self.y[original_idx]

            # Random brightness adjustment between 80% and 120%
            brightness_factor = 0.8 + 0.4 * torch.rand(1).item()

            x = (x * brightness_factor).clamp(0, 255)

        elif idx < (
            self.n_original
            + self.n_horizontal
            + self.n_erased
            + self.n_rotated
            + self.n_brightness
            + self.n_shifted
        ):
            original_idx = self.shifted_indices[
                idx
                - self.n_original
                - self.n_horizontal
                - self.n_erased
                - self.n_rotated
                - self.n_brightness
            ]

            x = self.X[original_idx].float()
            y = self.y[original_idx]

            C, H, W = x.shape

            # Random horizontal and vertical shift of +/- 3 pixels
            max_shift = 3

            x = F.pad(
                x,
                (max_shift, max_shift, max_shift, max_shift),
                mode="constant",
                value=127.0
            )

            top = torch.randint(
                0, 2 * max_shift + 1, (1,)
            ).item()

            left = torch.randint(
                0, 2 * max_shift + 1, (1,)
            ).item()

            x = x[:, top:top + H, left:left + W]

        else:
            original_idx = self.contrast_indices[
                idx
                - self.n_original
                - self.n_horizontal
                - self.n_erased
                - self.n_rotated
                - self.n_brightness
                - self.n_shifted
            ]

            x = self.X[original_idx].float()
            y = self.y[original_idx]

            # Random contrast adjustment between 80% and 120%
            contrast_factor = 0.8 + 0.4 * torch.rand(1).item()

            # Calculate image mean across channels and spatial dimensions
            mean = x.mean(dim=(0, 1, 2), keepdim=True)

            # Adjust pixel differences relative to image mean
            x = ((x - mean) * contrast_factor + mean).clamp(0, 255)

        # Normalize to approximately [-1, 1]
        x = x.float() / 127.5 - 1.0

        return x, y


def train(task, X_train, y_train, seed=0, device="cuda", epoch_callback=None):
    """Build and train a new model from scratch."""

    torch.manual_seed(seed)

    # 1. Initialize model
    model = ResidualMLP(**TASKS[task]).to(device)

    # 2. Prepare training data
    X = torch.as_tensor(X_train)
    y = torch.as_tensor(y_train).long()

    dataset = AugmentedDataset(
        X,
        y,
        augment=(task in ("cifar10", "imagenet32")),
        seed=seed
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # 3. Loss, optimizer, and scheduler
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS[task],
        eta_min=1e-5
    )

    # 4. Training loop
    for epoch in range(EPOCHS[task]):

        model.train()
        dataset.set_epoch(epoch)

        total_loss = 0.0

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(X_batch)

        scheduler.step()

        avg_train_loss = total_loss / len(dataset)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"epoch {epoch + 1}: "
            f"train loss {avg_train_loss:.4f} "
            f"lr {current_lr:.6f}"
        )

        # 5. Evaluate validation performance after each epoch
        if epoch_callback is not None:

            model.eval()

            with torch.no_grad():
                epoch_callback(
                    epoch=epoch + 1,
                    model=model,
                    train_loss=avg_train_loss,
                    lr=current_lr
                )

    # Return model ready for prediction
    model.eval()

    return model


def predict(model, X):
    """Return class probabilities for X, one row per example, on the CPU."""
    device = next(model.parameters()).device
    X = prepare(X)
    # Predict in chunks so a large test set does not run out of GPU memory.
    chunks = [model.predict(X[i:i + 1024].to(device)).cpu() for i in range(0, len(X), 1024)]
    return torch.cat(chunks)
