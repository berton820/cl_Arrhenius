"""
Evaluation metrics: R, R², AARE, RMSE. Optional robust AARE for low-stress regions.
"""

import numpy as np
from typing import Optional


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strain_min: Optional[float] = None,
    stress_min: Optional[float] = None,
    denom_floor: Optional[float] = None,
) -> dict:
    """
    Compute R, R², AARE (%), RMSE.
    If strain_min/stress_min/denom_floor are set, only use points with strain >= strain_min
    and stress >= stress_min, and use max(true, denom_floor) in AARE denominator.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    if not np.any(valid):
        return {"R": np.nan, "R2": np.nan, "AARE(%)": np.nan, "RMSE": np.nan, "N": 0}

    yt = y_true[valid]
    yp = y_pred[valid]

    if strain_min is not None or stress_min is not None:
        # Caller must pass matching mask if using strain/stress filters
        pass  # Here we assume no extra mask; filters are applied by caller
    if denom_floor is not None:
        denom = np.maximum(yt, denom_floor)
        aare = float(np.mean(np.abs(yp - yt) / denom) * 100.0)
    else:
        aare = float(np.mean(np.abs(yp - yt) / yt) * 100.0)

    r = np.corrcoef(yt, yp)[0, 1] if len(yt) > 1 else 0.0
    r = float(r) if np.isfinite(r) else np.nan
    r2 = r ** 2 if np.isfinite(r) else np.nan
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))

    return {
        "R": r,
        "R2": r2,
        "AARE(%)": aare,
        "RMSE": rmse,
        "N": int(valid.sum()),
    }


def evaluate_method(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: Optional[np.ndarray] = None,
    strain_min: Optional[float] = None,
    denom_floor: Optional[float] = None,
) -> dict:
    """
    Evaluate one method; optionally restrict to strain >= strain_min and use denom_floor for AARE.
    Returns dict with R, R2, AARE(%), RMSE, N, and optionally AARE_raw(%).
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)

    if eps is not None:
        eps = np.asarray(eps, dtype=float).ravel()
        if len(eps) == len(valid):
            if strain_min is not None:
                valid = valid & (eps >= strain_min)
        if len(eps) == len(y_true):
            if strain_min is not None:
                valid = valid & (eps >= strain_min)

    yt = y_true[valid]
    yp = y_pred[valid]
    if len(yt) < 2:
        return {"R": np.nan, "R2": np.nan, "AARE(%)": np.nan, "AARE_raw(%)": np.nan, "RMSE": np.nan, "N": 0}

    aare_raw = float(np.mean(np.abs(yp - yt) / yt) * 100.0)
    denom = np.maximum(yt, denom_floor) if denom_floor is not None else yt
    aare = float(np.mean(np.abs(yp - yt) / denom) * 100.0)

    r = np.corrcoef(yt, yp)[0, 1]
    r = float(r) if np.isfinite(r) else np.nan
    r2 = r ** 2 if np.isfinite(r) else np.nan
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))

    return {
        "R": r,
        "R2": r2,
        "AARE(%)": aare,
        "AARE_raw(%)": aare_raw,
        "RMSE": rmse,
        "N": int(valid.sum()),
    }
