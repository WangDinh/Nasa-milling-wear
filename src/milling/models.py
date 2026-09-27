"""Model zoo: classical regressors on tabular features and a small 1D-CNN on raw signals."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVR
from torch import nn


def make_sklearn_model(name: str, seed: int = 42):
    """Factory for the classical baselines. Scaling is inside the pipeline so CV stays leak-free."""
    if name == "mean":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        # Quantile transform bounds out-of-range inputs: held-out case 1 has vibration levels far
        # outside the other cases, and a plain scaled ridge extrapolates to VB of +-14 mm there.
        return make_pipeline(
            QuantileTransformer(n_quantiles=100, output_distribution="normal"),
            RidgeCV(alphas=np.logspace(-2, 4, 13)),
        )
    if name == "svr":
        return make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.02))
    if name == "rf":
        return RandomForestRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    if name == "gbr":
        return GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=seed
        )
    raise ValueError(f"unknown model: {name}")


class WearCNN(nn.Module):
    """1D-CNN over the 6 sensor channels, optionally fused with process parameters."""

    def __init__(self, n_channels: int = 6, n_aux: int = 0, width: int = 32, dropout: float = 0.3):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=7, padding=3),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.MaxPool1d(4),
            )

        self.encoder = nn.Sequential(
            block(n_channels, width // 2),
            block(width // 2, width),
            block(width, width * 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(width * 2 + n_aux, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        z = self.encoder(x)
        if aux is not None and aux.shape[1] > 0:
            z = torch.cat([z, aux], dim=1)
        return self.head(z).squeeze(1)
