"""Raw mill.mat -> data/processed/{runs.csv, signals.npy, features.csv}.

Usage: python scripts/make_dataset.py --config configs/baseline.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from milling.data import load_mill
from milling.features import extract_features, saturation_fraction
from milling.utils import load_config, resolve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    cfg = load_config(parser.parse_args().config)

    out_dir = resolve(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    meta, signals = load_mill(resolve(cfg["data"]["raw_path"]))
    meta["smcDC_saturation"] = saturation_fraction(np.nan_to_num(signals))
    print(f"runs: {len(meta)}  valid: {meta['valid'].sum()}  VB measured: {meta['vb_measured'].sum()}")
    for _, r in meta[~meta["valid"]].iterrows():
        print(f"  dropped case {r['case']} run {r['run']}: {r['invalid_reason']}")

    feats = extract_features(signals, tuple(cfg["features"]["window"]))
    features = pd.concat([meta[["case", "run"]], feats], axis=1)

    meta.to_csv(out_dir / "runs.csv", index=False)
    np.save(out_dir / "signals.npy", signals)
    features.to_csv(out_dir / "features.csv", index=False)
    print(f"wrote {out_dir}/runs.csv, signals.npy {signals.shape}, features.csv {feats.shape}")


if __name__ == "__main__":
    main()
