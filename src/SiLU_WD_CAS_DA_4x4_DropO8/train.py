import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import math
import numpy as np

from .model import TASKS, ResidualMLP


# Baseline settings. Deliberately plain so they are easy to beat:
# plain SGD, constant learning rate, no dropout, no schedule, no augmentation.
LEARNING_RATE = 1.3e-3
BATCH_SIZE = 128
EPOCHS = {"mnist": 10, "cifar10": 90, "imagenet32": 10}


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

def cutmix(X, y, g, np_rng, alpha=1.0):
    B, C, H, W = X.shape

    # Randomly pair images within the batch.
    indices = torch.randperm(B, generator=g).to(X.device)

    # Sample mixing proportion reproducibly.
    lam = np_rng.beta(alpha, alpha)

    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    # Random patch center.
    cx = torch.randint(W, (1,), generator=g).item()
    cy = torch.randint(H, (1,), generator=g).item()

    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)

    # Replace region with another image's region.
    X_mixed = X.clone()
    X_mixed[:, :, y1:y2, x1:x2] = X[
        indices, :, y1:y2, x1:x2
    ]

    # Correct lambda for the actual area replaced.
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))

    return X_mixed, y, y[indices], lam

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, augment=False, seed=0):
        self.X = X
        self.y = y
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone().float()
        y = self.y[idx]

        if self.augment:
            # Deterministic random generator for this epoch and image.
            g = torch.Generator().manual_seed(
                self.seed + self.epoch * len(self.X) + idx
            )

            C, H, W = x.shape

            # 1. Horizontal flip (50%)
            if torch.rand((), generator=g).item() < 0.50:
                x = torch.flip(x, dims=[2])

            # 2. Rotation +/- 10 degrees (30%)
            if torch.rand((), generator=g).item() < 0.30:
                angle = (torch.rand((), generator=g).item()
                         * 20.0 - 10.0)
                radians = math.radians(angle)

                cos_a = math.cos(radians)
                sin_a = math.sin(radians)

                theta = torch.tensor([[
                    [cos_a, -sin_a, 0.0],
                    [sin_a,  cos_a, 0.0]
                ]], dtype=torch.float32)

                grid = F.affine_grid(
                    theta, (1, C, H, W),
                    align_corners=False
                )

                x = F.grid_sample(
                    (x - 127.0).unsqueeze(0),
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False
                ).squeeze(0) + 127.0

            # 3. RGB scaling (30%)
            if C == 3 and torch.rand((), generator=g).item() < 0.30:
                factors = (
                    0.9 + 0.2 * torch.rand(
                        (3, 1, 1), generator=g
                    )
                )
                x = (x * factors).clamp(0, 255)

            # 4. Random erasure (20%)
            if torch.rand((), generator=g).item() < 0.0:
                fraction = (
                    0.02 + 0.06 * torch.rand(
                        (), generator=g
                    ).item()
                )

                side = max(
                    1, round((H * W * fraction) ** 0.5)
                )
                side = min(side, H, W)

                top = torch.randint(
                    0, H - side + 1, (1,), generator=g
                ).item()

                left = torch.randint(
                    0, W - side + 1, (1,), generator=g
                ).item()

                x[:, top:top + side, left:left + side] = 127.0

        # Normalize to approximately [-1, 1].
        x = x.clamp(0, 255) / 127.5 - 1.0

        return x, y



def train(task, X_train, y_train, seed=0, device="cuda", epoch_callback=None):
    """Build and train a new model from scratch."""

    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    np_rng = np.random.default_rng(seed)

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
        # 4. Training loop
    for epoch in range(EPOCHS[task]):

        model.train()
        dataset.set_epoch(epoch)

        total_loss = 0.0

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Apply CutMix to 50% of batches for color datasets.
            if (
                task in ("cifar10", "imagenet32")
                and torch.rand((), generator=g).item() < 0.50
            ):
                X_mixed, y_a, y_b, lam = cutmix(
                    X_batch, y_batch, g, np_rng, alpha=1.0
                )

                logits = model(X_mixed)

                loss = (
                    lam * loss_fn(logits, y_a)
                    + (1.0 - lam) * loss_fn(logits, y_b)
                )

            else:
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

    model.eval()
    return model


def predict(model, X):
    """Return class probabilities for X, one row per example, on the CPU."""
    device = next(model.parameters()).device
    X = prepare(X)
    # Predict in chunks so a large test set does not run out of GPU memory.
    chunks = [model.predict(X[i:i + 1024].to(device)).cpu() for i in range(0, len(X), 1024)]
    return torch.cat(chunks)
