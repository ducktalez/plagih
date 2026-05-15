"""
PyTorch MLP models for the NN+GP pipeline.

Two modes:
  - ``find_minimal_nn``: grid-search for the smallest architecture that reaches
    a target MSE (used to measure "how much the GP features help").
  - ``train_nn``: single training run with a fixed architecture.

All functions use MSE as the shared loss function (same as GP's error metric).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Simple fully-connected network with ReLU activations and a sigmoid output.

    The sigmoid output maps predictions to ``[0, 1]``, matching the normalised
    target range used throughout the pipeline.
    """

    def __init__(self, in_features: int, hidden_sizes: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_features
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Training result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    """Result of a single NN training run."""

    hidden_sizes: List[int]
    param_count: int
    final_mse: float
    loss_curve: List[float] = field(default_factory=list)

    def __str__(self) -> str:
        return f"MLP{self.hidden_sizes}  params={self.param_count:,}  mse={self.final_mse:.6f}"


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------


def train_nn(
    X: np.ndarray,
    y: np.ndarray,
    hidden_sizes: List[int],
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 30,
    device: Optional[str] = None,
) -> TrainResult:
    """Train a single MLP with early stopping.

    Args:
        X: Feature matrix ``[n_rows, n_features]``, float32.
        y: Target vector ``[n_rows]``, float32, values in ``[0, 1]``.
        hidden_sizes: List of hidden layer widths, e.g. ``[32, 16]``.
        epochs: Maximum training epochs.
        lr: Adam learning rate.
        batch_size: Mini-batch size.
        patience: Early-stopping patience (epochs without val improvement).
        device: ``'cpu'`` or ``'cuda'``. Auto-detected if None.

    Returns:
        TrainResult with metrics.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    X_t = torch.tensor(X, dtype=torch.float32, device=dev)
    y_t = torch.tensor(y, dtype=torch.float32, device=dev)

    # 80/20 train/val split (deterministic)
    n = len(X_t)
    n_train = int(n * 0.8)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    tr_idx, va_idx = idx[:n_train], idx[n_train:]

    ds_train = TensorDataset(X_t[tr_idx], y_t[tr_idx])
    ds_val = TensorDataset(X_t[va_idx], y_t[va_idx])
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    model = MLP(X.shape[1], hidden_sizes).to(dev)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_curve: List[float] = []
    best_val = float("inf")
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in dl_train:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            Xv, yv = ds_val.tensors
            val_loss = criterion(model(Xv), yv).item()
        loss_curve.append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Final MSE on full dataset
    model.eval()
    with torch.no_grad():
        full_mse = criterion(model(X_t), y_t).item()

    return TrainResult(
        hidden_sizes=hidden_sizes,
        param_count=model.param_count,
        final_mse=full_mse,
        loss_curve=loss_curve,
    )


# ---------------------------------------------------------------------------
# Grid search for minimal architecture
# ---------------------------------------------------------------------------

# Candidate architectures ordered from smallest to largest.
# Format: list of hidden-layer widths.
_ARCHITECTURE_GRID: List[List[int]] = [
    [8],
    [16],
    [32],
    [8, 8],
    [16, 8],
    [32, 16],
    [64],
    [32, 32],
    [64, 32],
    [64, 64],
    [128, 64],
    [128, 128],
]


def find_minimal_nn(
    X: np.ndarray,
    y: np.ndarray,
    target_mse: float,
    *,
    tolerance: float = 0.05,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 30,
    device: Optional[str] = None,
) -> Tuple[TrainResult, List[TrainResult]]:
    """Find the smallest MLP architecture that achieves ``target_mse * (1 + tolerance)``.

    Iterates over ``_ARCHITECTURE_GRID`` (smallest first) and returns the first
    architecture that satisfies the criterion.  If none does, returns the result
    for the largest architecture tried.

    Args:
        X: Feature matrix.
        y: Normalised target vector.
        target_mse: Reference MSE to beat (e.g. baseline NN on raw features).
        tolerance: Fractional tolerance (default 0.05 = 5%).
        epochs / lr / batch_size / patience / device: Passed to ``train_nn``.

    Returns:
        Tuple of (best_result, all_results) where all_results lists every
        architecture tried (useful for plotting the search trajectory).
    """
    threshold = target_mse * (1.0 + tolerance)
    all_results: List[TrainResult] = []

    for hidden_sizes in _ARCHITECTURE_GRID:
        result = train_nn(
            X,
            y,
            hidden_sizes,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
            device=device,
        )
        all_results.append(result)
        if result.final_mse <= threshold:
            return result, all_results

    # Return largest tried if nothing succeeded
    return all_results[-1], all_results
