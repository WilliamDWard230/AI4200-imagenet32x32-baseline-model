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

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, augment=False, seed=0):
        self.X = X
        self.y = y
        self.augment = augment
        self.seed = seed

        self.n_original = len(X)
        self.n_horizontal = int(self.n_original * 0.50) if augment else 0
        self.n_vertical = int(self.n_original * 0.50) if augment else 0

    def set_epoch(self, epoch):
        if self.augment:
            generator = torch.Generator().manual_seed(self.seed + epoch)

            self.horizontal_indices = torch.randperm(
                self.n_original,
                generator=generator
            )[:self.n_horizontal]

            self.vertical_indices = torch.randperm(
                self.n_original,
                generator=generator
            )[:self.n_vertical]

    def __len__(self):
        return self.n_original + self.n_horizontal + self.n_vertical

    def __getitem__(self, idx):
        if idx < self.n_original:
            x = self.X[idx]
            y = self.y[idx]

        elif idx < self.n_original + self.n_horizontal:
            original_idx = self.horizontal_indices[idx - self.n_original]
            x = torch.flip(self.X[original_idx], dims=[2])
            y = self.y[original_idx]

        else:
            original_idx = self.vertical_indices[
                idx - self.n_original - self.n_horizontal
            ]
            x = torch.flip(self.X[original_idx], dims=[1])
            y = self.y[original_idx]

        # Normalize only when an image is retrieved
        x = x.float() / 127.5 - 1.0

        return x, y

def train(task, X_train, y_train, seed=0, device="cuda"):
    """Build a new model and train it from scratch. Returns the trained model."""
    torch.manual_seed(seed)                   # makes the run reproducible

    # 1. Build the model for this task. Only input size and class count differ.
    model = ResidualMLP(**TASKS[task]).to(device)

    # 2. Wrap the data so we can loop over shuffled mini-batches.
    # Prepare original training data
    X = torch.as_tensor(X_train)
    y = torch.as_tensor(y_train).long()

    dataset = AugmentedDataset(
        X,
        y,
        augment=(task in ("cifar10", "imagenet32")),
        seed = seed
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # 3. Choose the loss and the optimizer.
    loss_fn = nn.CrossEntropyLoss()           # expects raw logits
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS[task], eta_min = 1e-5)

    # 4. The training loop.
    model.train()
    for epoch in range(EPOCHS[task]):
        dataset.set_epoch(epoch)
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)           # forward pass
            loss = loss_fn(logits, y_batch)   # how wrong are we?

            optimizer.zero_grad()             # clear old gradients
            loss.backward()                   # backward pass: compute gradients
            optimizer.step()                  # update the weights

            total_loss += loss.item() * len(X_batch)
        scheduler.step()

        print(f"epoch {epoch + 1}: train loss {total_loss / len(dataset):.4f} lr {optimizer.param_groups[0]['lr']:.6f}")

    return model


def predict(model, X):
    """Return class probabilities for X, one row per example, on the CPU."""
    device = next(model.parameters()).device
    X = prepare(X)
    # Predict in chunks so a large test set does not run out of GPU memory.
    chunks = [model.predict(X[i:i + 1024].to(device)).cpu() for i in range(0, len(X), 1024)]
    return torch.cat(chunks)
