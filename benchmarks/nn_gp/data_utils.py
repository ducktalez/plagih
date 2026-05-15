"""
Data utilities for the NN+GP pipeline.

Handles normalization, residual computation, and GP feature matrix construction.
All functions are stateless and work on plain numpy arrays / DataFrames.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

if TYPE_CHECKING:
    from plagih.trees._evolution import Candidate


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_dataset(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, MinMaxScaler, MinMaxScaler]:
    """Normalize all feature columns and the target column independently to [0, 1].

    For classification targets (integer-valued), the class labels are treated as
    ordinal floats and scaled to [0, 1].  This allows a single shared MSE loss
    for both GP and NN without any special-casing.

    Args:
        df: Raw DataFrame with feature columns and a target column.
        target_col: Name of the target column.

    Returns:
        df_norm: Normalized DataFrame (same column names).
        feature_scaler: Fitted MinMaxScaler for feature columns.
        target_scaler: Fitted MinMaxScaler for the target column.
    """
    feature_cols = [c for c in df.columns if c != target_col]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    df_norm = df.copy().astype(float)
    df_norm[feature_cols] = feature_scaler.fit_transform(df[feature_cols].astype(float))
    df_norm[[target_col]] = target_scaler.fit_transform(df[[target_col]].astype(float))

    return df_norm, feature_scaler, target_scaler


def denormalize_predictions(
    preds_norm: np.ndarray,
    target_scaler: MinMaxScaler,
) -> np.ndarray:
    """Inverse-transform normalized predictions back to the original scale."""
    return target_scaler.inverse_transform(preds_norm.reshape(-1, 1)).ravel()


# ---------------------------------------------------------------------------
# GP feature matrix
# ---------------------------------------------------------------------------


def build_gp_feature_matrix(
    candidates: List["Candidate"],
    df_norm: pd.DataFrame,
) -> np.ndarray:
    """Evaluate each GP candidate on the normalized DataFrame.

    Returns a feature matrix of shape ``[n_rows, n_candidates]`` where column ``i``
    is the output of Pareto candidate ``i`` on every training row.  Outputs are
    clipped to ``[0, 1]`` (same range as the normalized target) and NaN/Inf
    replaced with 0.5 (neutral mid-point) so the NN training does not see
    invalid values.

    Args:
        candidates: List of GP Candidate objects (typically the Pareto front).
        df_norm: Normalized DataFrame (features only; target column may be present
                 but is ignored for evaluation).

    Returns:
        Feature matrix of shape ``[n_rows, len(candidates)]``.
    """
    if not candidates:
        return np.empty((len(df_norm), 0), dtype=np.float32)

    columns = []
    for cand in candidates:
        raw = cand.tree.eval_predict_numpy_now(df_norm)
        arr = np.asarray(raw, dtype=np.float64).ravel()
        # Replace non-finite values with neutral 0.5
        arr = np.where(np.isfinite(arr), arr, 0.5)
        # Clip to [0, 1] (normalised output range)
        arr = np.clip(arr, 0.0, 1.0)
        columns.append(arr.astype(np.float32))

    return np.column_stack(columns)


# ---------------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------------


def compute_residual(
    gp_feature_matrix: np.ndarray,
    target_norm: np.ndarray,
) -> np.ndarray:
    """Compute the per-row residual that GP candidates could not explain.

    For each row, the "best GP prediction" is the candidate output nearest to
    the target.  The residual is ``target - best_gp_pred``, clipped back to
    ``[0, 1]`` so subsequent GP runs operate on a valid normalised target.

    If no candidates are available (empty feature matrix), returns the original
    target unchanged.

    Args:
        gp_feature_matrix: Shape ``[n_rows, n_candidates]``.  May be empty.
        target_norm: Normalised target array, shape ``[n_rows]``.

    Returns:
        Residual array of shape ``[n_rows]``, values in ``[0, 1]``.
    """
    if gp_feature_matrix.shape[1] == 0:
        return target_norm.copy()

    # Per-row absolute errors: shape [n_rows, n_candidates]
    errors = np.abs(gp_feature_matrix - target_norm[:, np.newaxis])
    best_idx = np.argmin(errors, axis=1)  # shape [n_rows]
    best_preds = gp_feature_matrix[np.arange(len(target_norm)), best_idx]

    residual = target_norm - best_preds
    # Shift & scale residual into [0, 1] for the next GP iteration
    r_min, r_max = residual.min(), residual.max()
    if np.isclose(r_min, r_max):
        # All residuals identical → constant residual, no information to fit
        return np.full_like(target_norm, 0.5)
    residual_norm = (residual - r_min) / (r_max - r_min)
    return residual_norm.astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience: build enriched feature array for NN training
# ---------------------------------------------------------------------------


def build_nn_input(
    df_norm: pd.DataFrame,
    target_col: str,
    gp_feature_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Concatenate raw normalised features with optional GP feature columns.

    Args:
        df_norm: Normalised DataFrame.
        target_col: Target column name (excluded from features).
        gp_feature_matrix: Optional ``[n_rows, n_candidates]`` GP features.

    Returns:
        Feature matrix ``[n_rows, n_raw_features (+ n_candidates)]``.
    """
    feature_cols = [c for c in df_norm.columns if c != target_col]
    X_raw = df_norm[feature_cols].to_numpy(dtype=np.float32)
    if gp_feature_matrix is None or gp_feature_matrix.shape[1] == 0:
        return X_raw
    return np.concatenate([X_raw, gp_feature_matrix.astype(np.float32)], axis=1)
