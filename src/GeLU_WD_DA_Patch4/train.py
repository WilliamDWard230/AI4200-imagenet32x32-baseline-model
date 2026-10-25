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

    # Augmentation settings
    FLIP_FRACTION = 0.15
    NOISE_FRACTION = 0.15
    SOBEL_FRACTION = 0.15
    NOISE_STD = 0.05

    generator = torch.Generator().manual_seed(seed)
    n_original = len(X)

    # 1. Horizontal-flipped copies
    n_flip = int(n_original * FLIP_FRACTION)

    flip_indices = torch.randperm(
        n_original, generator=generator
    )[:n_flip]

    X_flipped = torch.flip(X[flip_indices], dims=[3])
    y_flipped = y[flip_indices].clone()

    # 2. Gaussian-noise copies
    n_noise = int(n_original * NOISE_FRACTION)

    noise_indices = torch.randperm(
        n_original, generator=generator
    )[:n_noise]

    X_noisy = X[noise_indices].clone()

    noise = torch.randn(
        X_noisy.shape,
        generator=generator,
        dtype=X_noisy.dtype
    ) * NOISE_STD

    X_noisy = (X_noisy + noise).clamp(-1.0, 1.0)
    y_noisy = y[noise_indices].clone()

    # 3. Sobel-filtered copies
    n_sobel = int(n_original * SOBEL_FRACTION)

    sobel_indices = torch.randperm(
        n_original, generator=generator
    )[:n_sobel]

    X_sobel = sobel_filter(X[sobel_indices].clone())
    y_sobel = y[sobel_indices].clone()

    # 4. Combine original and augmented images
    X = torch.cat(
        [X, X_flipped, X_noisy, X_sobel],
        dim=0
    )

    y = torch.cat(
        [y, y_flipped, y_noisy, y_sobel],
        dim=0
    )

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print(f"Original images: {n_original}")
    print(f"Horizontal-flipped copies: {n_flip}")
    print(f"Gaussian-noise copies: {n_noise}")
    print(f"Sobel-filtered copies: {n_sobel}")
    print(f"Total training examples: {len(dataset)}")

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
