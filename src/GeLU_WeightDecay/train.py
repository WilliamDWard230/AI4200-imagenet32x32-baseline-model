import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import TASKS, ResidualMLP


# Baseline settings. Deliberately plain so they are easy to beat:
# plain SGD, constant learning rate, no dropout, no schedule, no augmentation.
LEARNING_RATE = 0.05
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

def train(task, X_train, y_train, seed=0, device="cuda"):
    """Build a new model and train it from scratch. Returns the trained model."""
    torch.manual_seed(seed)                   # makes the run reproducible

    # 1. Build the model for this task. Only input size and class count differ.
    model = ResidualMLP(**TASKS[task]).to(device)

    # 2. Wrap the data so we can loop over shuffled mini-batches. (Added gaussian noise data aug)
    # Prepare original images and labels
    X = prepare(X_train)
    y = torch.as_tensor(y_train).long()

    # Select 5% of images for augmentation
    num_augmented = int(len(X) * 0.05)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(X), generator=generator)[:num_augmented]

    # Duplicate images and add Gaussian noise
    X_noisy = X[indices].clone()

    noise = torch.randn(
        X_noisy.shape,
        generator=generator,
        dtype=X_noisy.dtype
    ) * 0.05

    X_noisy = (X_noisy + noise).clamp(-1.0, 1.0)

    # Preserve original images and append noisy copies
    X = torch.cat([X, X_noisy], dim=0)
    y = torch.cat([y, y[indices].clone()], dim=0)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Choose the loss and the optimizer.
    loss_fn = nn.CrossEntropyLoss()           # expects raw logits
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

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
