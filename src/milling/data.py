"""Load and clean the NASA milling dataset (mill.mat).

The .mat file holds a 1x167 struct array; each element is one experimental run
with scalar metadata (case, run, VB, time, DOC, feed, material) and six sensor
signals sampled at 250 Hz (9000 samples = 36 s).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

SIGNALS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
META = ["case", "run", "VB", "time", "DOC", "feed", "material"]
MATERIALS = {1: "cast_iron", 2: "steel"}
FS = 250.0  # Hz, after the RMS devices (see docs/dataset_readme.pdf)
N_SAMPLES = 9000
SMC_DC_SATURATION = 9.99  # the DC current channel clips at the ADC ceiling (9.995 V)
MAX_ABS_VALUE = 100.0  # anything above this is a corrupted record, not a physical reading


def load_mill(path: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (meta, signals).

    meta    : one row per run (all 167), with a ``valid`` flag and ``vb_interp``.
    signals : float32 array (n_runs, n_signals, N_SAMPLES); corrupted runs are NaN.
    """
    mill = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)["mill"]

    rows, signals = [], np.full((len(mill), len(SIGNALS), N_SAMPLES), np.nan, dtype=np.float32)
    for i, rec in enumerate(mill):
        row = {k: float(getattr(rec, k)) for k in META}
        row["idx"] = i
        arrs = [np.asarray(getattr(rec, s), dtype=np.float64) for s in SIGNALS]
        lengths = {len(a) for a in arrs}
        reason = ""
        if lengths != {N_SAMPLES}:
            reason = f"length {sorted(lengths)} != {N_SAMPLES}"
        elif max(np.nanmax(np.abs(a)) for a in arrs) > MAX_ABS_VALUE:
            reason = "non-physical values (overflow)"
        row["valid"] = reason == ""
        row["invalid_reason"] = reason
        if row["valid"]:
            signals[i] = np.stack(arrs).astype(np.float32)
        rows.append(row)

    meta = pd.DataFrame(rows)
    for col in ["case", "run", "material", "idx"]:
        meta[col] = meta[col].astype(int)
    meta["material_name"] = meta["material"].map(MATERIALS)
    meta["vb_measured"] = meta["VB"].notna()
    meta["vb_interp"] = _interpolate_vb(meta)
    meta = meta.sort_values(["case", "run"]).reset_index(drop=True)
    signals = signals[meta["idx"].to_numpy()]
    return meta, signals


def _interpolate_vb(meta: pd.DataFrame) -> pd.Series:
    """Linearly interpolate VB over cutting time within each case.

    Only fills gaps *between* measurements; never extrapolates past the last one.
    """
    out = pd.Series(np.nan, index=meta.index)
    for _, g in meta.groupby("case"):
        known = g[g["VB"].notna()]
        if len(known) < 2:
            out.loc[known.index] = known["VB"]
            continue
        t = g["time"].to_numpy()
        inside = (t >= known["time"].min()) & (t <= known["time"].max())
        interp = np.interp(t, known["time"], known["VB"])
        out.loc[g.index[inside]] = interp[inside]
    return out


def labelled(meta: pd.DataFrame, use_interpolated: bool = False) -> pd.DataFrame:
    """Valid runs that have a (measured or interpolated) VB label."""
    col = "vb_interp" if use_interpolated else "VB"
    df = meta[meta["valid"] & meta[col].notna()].copy()
    df["target"] = df[col]
    return df


def load_processed(processed_dir: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the outputs of scripts/make_dataset.py: (runs merged with features, signals)."""
    processed_dir = Path(processed_dir)
    runs = pd.read_csv(processed_dir / "runs.csv")
    features = pd.read_csv(processed_dir / "features.csv")
    table = runs.merge(features, on=["case", "run"], how="left", validate="one_to_one")
    signals = np.load(processed_dir / "signals.npy")
    return table, signals
