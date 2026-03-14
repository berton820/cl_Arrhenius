"""
Neural network model (StressNet), feature preparation, training, and prediction.
Supports log(stress) target for relative-error-friendly training.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy
from typing import Optional, Tuple, Union

# Use same column names as data_loader
COL_TK = "T_K"
COL_SR = "StrainRate"
COL_STRAIN = "TrueStrain"
COL_STRESS = "FlowStress"

# Default device
def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StressNet(nn.Module):
    """MLP: input [1000/T, ln(strain_rate), strain], output stress (or log(stress))."""

    def __init__(self, hidden_dims=(128, 128, 64)):
        super().__init__()
        layers = []
        in_dim = 3
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.SiLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Normalizer:
    """Z-score normalizer for features and targets."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        data = np.asarray(data)
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        return self

    def transform(self, data):
        data = np.asarray(data)
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        data = np.asarray(data)
        return data * self.std + self.mean


def prepare_features(
    df_or_arrays: Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    is_arrays: bool = False,
) -> np.ndarray:
    """Build feature matrix [1000/T_K, ln(StrainRate), TrueStrain]."""
    import pandas as pd
    if is_arrays:
        T_K, SR, EPS = df_or_arrays
    else:
        T_K = df_or_arrays[COL_TK].values
        SR = df_or_arrays[COL_SR].values
        EPS = df_or_arrays[COL_STRAIN].values

    T_feat = 1000.0 / np.asarray(T_K, dtype=float)
    ln_sr = np.log(np.asarray(SR, dtype=float))
    eps = np.asarray(EPS, dtype=float)
    return np.column_stack([T_feat, ln_sr, eps])


def transform_target_for_training(y: np.ndarray, use_log_stress: bool = True) -> np.ndarray:
    """Convert stress to training target: log(stress) or stress."""
    y = np.asarray(y, dtype=float)
    if use_log_stress:
        return np.log(np.clip(y, 1e-6, None))
    return y


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int = 800,
    lr: float = 1e-3,
    batch_size: int = 256,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    verbose_every: Optional[int] = 200,
    patience: Optional[int] = 120,
    device: Optional[torch.device] = None,
) -> dict:
    """Train model with optional validation and early stopping. Returns history dict."""
    if device is None:
        device = _device()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": []}
    model.to(device)

    best_state = None
    best_val = float("inf")
    bad = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)
        scheduler.step()

        train_loss = total_loss / len(X_train)
        history["train_loss"].append(train_loss)

        val_loss = None
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                vp = model(torch.FloatTensor(X_val).to(device))
                val_loss = float(criterion(vp, torch.FloatTensor(y_val).reshape(-1, 1).to(device)).item())
            history["val_loss"].append(val_loss)

            if val_loss + 1e-10 < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                bad = 0
            else:
                bad += 1
                if patience and bad >= patience:
                    break

        if verbose_every and (epoch + 1) % verbose_every == 0:
            if val_loss is not None:
                print(f"  Epoch {epoch+1:>4d}/{epochs}: train={train_loss:.6e}, val={val_loss:.6e}")
            else:
                print(f"  Epoch {epoch+1:>4d}/{epochs}: train={train_loss:.6e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def predict_nn(
    model: nn.Module,
    X: np.ndarray,
    norm_X: Normalizer,
    norm_y: Normalizer,
    use_log_stress_target: bool = True,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Predict flow stress (MPa) from features. Inverse of log if trained on log(stress)."""
    if device is None:
        device = _device()
    Xn = norm_X.transform(X)
    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.FloatTensor(Xn).to(device))
    pred_norm_np = np.array(pred_norm.detach().cpu().reshape(-1).tolist(), dtype=float)
    pred_space = norm_y.inverse_transform(pred_norm_np)
    if use_log_stress_target:
        return np.exp(pred_space)
    return pred_space
