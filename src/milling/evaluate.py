"""Leave-one-case-out cross-validation and regression metrics.

Runs within a case share the same insert and are strongly autocorrelated, so a
random split would leak. Every fold holds out one full case (one tool's life).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def loco_splits(train_cases: np.ndarray, eval_cases: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    """Yield (train_idx, test_idx, case) with one case held out per fold.

    The training pool and the evaluation set may differ (e.g. training on interpolated
    labels while evaluating only on measured VB), so indices refer to each array separately.
    """
    for case in np.unique(eval_cases):
        yield np.flatnonzero(train_cases != case), np.flatnonzero(eval_cases == case), int(case)


def summarize(df: pd.DataFrame, pred_col: str = "pred") -> dict[str, float]:
    """Pooled out-of-fold metrics plus the mean of per-case MAE."""
    out = regression_metrics(df["target"], df[pred_col])
    out["case_MAE_mean"] = float(
        df.groupby("case").apply(lambda g: mean_absolute_error(g["target"], g[pred_col])).mean()
    )
    return out
