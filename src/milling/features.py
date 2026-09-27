"""Hand-crafted time- and frequency-domain features from the steady-state cut."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy import stats

from milling.data import FS, SIGNALS, SMC_DC_SATURATION

PROCESS_FEATURES = ["DOC", "feed", "material"]

# Frequency bands (Hz) for band-power features. Spindle speed is 826 rpm (13.8 Hz);
# with 6 inserts the tooth-passing frequency is ~82.6 Hz.
BANDS = [(0, 10), (10, 30), (30, 60), (60, 100), (100, 125)]


def crop(signals: np.ndarray, window: tuple[int, int]) -> np.ndarray:
    """Keep the steady-state cutting segment [start, end) along the last axis."""
    start, end = window
    return signals[..., start:end]


def _channel_features(x: np.ndarray) -> dict[str, float]:
    rms = np.sqrt(np.mean(x**2))
    flat = x.std() < 1e-6  # e.g. smcDC clipped for the whole window
    feats = {
        "mean": x.mean(),
        "std": x.std(),
        "rms": rms,
        "max": x.max(),
        "min": x.min(),
        "p2p": np.ptp(x),
        "skew": 0.0 if flat else stats.skew(x),
        "kurtosis": 0.0 if flat else stats.kurtosis(x),
        "crest": np.abs(x).max() / rms if rms > 0 else 0.0,
        "p95": np.percentile(x, 95),
    }
    freqs, psd = sps.welch(x - x.mean(), fs=FS, nperseg=512)
    total = psd.sum() + 1e-12
    feats["spec_centroid"] = (freqs * psd).sum() / total
    feats["spec_peak_hz"] = freqs[np.argmax(psd)]
    for lo, hi in BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        feats[f"band_{lo}_{hi}"] = psd[mask].sum() / total
    return feats


def extract_features(signals: np.ndarray, window: tuple[int, int]) -> pd.DataFrame:
    """Return one row per run with columns ``<signal>__<feature>``."""
    cropped = crop(signals, window)
    rows = []
    for run in cropped:
        row = {}
        if np.isnan(run).any():
            rows.append(row)
            continue
        for name, x in zip(SIGNALS, run):
            for k, v in _channel_features(x.astype(np.float64)).items():
                row[f"{name}__{k}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)


def saturation_fraction(
    signals: np.ndarray, channel: str = "smcDC", level: float = SMC_DC_SATURATION
) -> np.ndarray:
    """Fraction of samples per run at the clipping level of a channel."""
    ch = SIGNALS.index(channel)
    return (signals[:, ch, :] >= level).mean(axis=1)


def feature_columns(table: pd.DataFrame, parts: list[str]) -> list[str]:
    """Resolve feature-set parts ('sensor', 'process', 'time') to column names."""
    cols: list[str] = []
    for part in parts:
        if part == "sensor":
            cols += [c for c in table.columns if "__" in c]
        elif part == "process":
            cols += PROCESS_FEATURES
        elif part == "time":
            cols += ["time"]
        else:
            raise ValueError(f"unknown feature part: {part}")
    return cols
