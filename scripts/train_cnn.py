"""1D-CNN on the raw steady-state signals, evaluated with leave-one-case-out CV.

Training uses random crops of the steady-state window as augmentation; a run's
prediction is the mean over sliding windows.

Usage: python scripts/train_cnn.py --config configs/baseline.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from torch import nn

from milling.data import labelled, load_processed
from milling.evaluate import loco_splits, summarize
from milling.features import PROCESS_FEATURES, crop
from milling.models import WearCNN
from milling.utils import load_config, resolve, seed_everything

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sliding_windows(x: np.ndarray, length: int, stride: int) -> np.ndarray:
    starts = range(0, x.shape[-1] - length + 1, stride)
    return np.stack([x[..., s : s + length] for s in starts], axis=1)  # (n, n_win, C, L)


def fit_predict_fold(X_tr, A_tr, y_tr, X_te, A_te, cc, seed):
    seed_everything(seed)
    # Per-channel normalisation from the training fold only.
    mu = X_tr.mean(axis=(0, 2), keepdims=True)
    sd = X_tr.std(axis=(0, 2), keepdims=True) + 1e-6
    X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd
    a_mu, a_sd = A_tr.mean(0), A_tr.std(0) + 1e-6
    A_tr, A_te = (A_tr - a_mu) / a_sd, (A_te - a_mu) / a_sd

    model = WearCNN(n_channels=X_tr.shape[1], n_aux=A_tr.shape[1], width=cc["width"], dropout=cc["dropout"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cc["lr"], weight_decay=cc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cc["epochs"])
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    At = torch.tensor(A_tr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    n, L, T = len(Xt), cc["crop_len"], Xt.shape[-1]
    rng = np.random.default_rng(seed)

    model.train()
    for _ in range(cc["epochs"]):
        for batch in np.array_split(rng.permutation(n), max(1, n // cc["batch_size"])):
            starts = torch.as_tensor(rng.integers(0, T - L + 1, size=len(batch)), device=DEVICE)
            idx = starts[:, None] + torch.arange(L, device=DEVICE)[None, :]
            xb = torch.gather(Xt[batch], 2, idx[:, None, :].expand(-1, Xt.shape[1], -1))
            loss = loss_fn(model(xb, At[batch]), yt[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    W = sliding_windows(X_te, L, cc["eval_stride"])  # (n_te, n_win, C, L)
    n_te, n_win = W.shape[:2]
    with torch.no_grad():
        xw = torch.tensor(W.reshape(n_te * n_win, *W.shape[2:]), dtype=torch.float32, device=DEVICE)
        aw = torch.tensor(np.repeat(A_te, n_win, axis=0), dtype=torch.float32, device=DEVICE)
        pred = model(xw, aw).cpu().numpy().reshape(n_te, n_win).mean(axis=1)
    return pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    cfg = load_config(parser.parse_args().config)
    cc = cfg["cnn"]
    res_dir = resolve(cfg["output"]["results_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)

    table, signals = load_processed(resolve(cfg["data"]["processed_dir"]))
    signals = crop(signals, tuple(cfg["features"]["window"]))
    train_df = labelled(table, cfg["data"]["use_interpolated_vb"])
    eval_df = labelled(table, use_interpolated=False)
    aux_cols = PROCESS_FEATURES if cc["use_process"] else []
    print(f"device: {DEVICE} | train pool {len(train_df)} | eval {len(eval_df)} | aux {aux_cols}")

    X_pool, X_eval = signals[train_df.index], signals[eval_df.index]
    A_pool = train_df[aux_cols].to_numpy(np.float32).reshape(len(train_df), -1)
    A_eval = eval_df[aux_cols].to_numpy(np.float32).reshape(len(eval_df), -1)
    y_pool = train_df["target"].to_numpy(np.float32)

    oof = eval_df[["case", "run", "time", "material_name", "target"]].copy()
    name = "sensor_process" if cc["use_process"] else "sensor_only"
    rows = []
    for seed in cc["seeds"]:
        pred = np.full(len(eval_df), np.nan)
        for tr, te, case in loco_splits(train_df["case"].to_numpy(), eval_df["case"].to_numpy()):
            pred[te] = fit_predict_fold(X_pool[tr], A_pool[tr], y_pool[tr], X_eval[te], A_eval[te], cc, 1000 * seed + case)
        oof[f"{name}/cnn_seed{seed}"] = pred
        m = summarize(oof.assign(pred=pred))
        rows.append({"seed": seed, **m})
        print(f"seed {seed}: MAE {m['MAE']:.3f}  RMSE {m['RMSE']:.3f}  R2 {m['R2']:+.3f}")

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(res_dir / "metrics_cnn_seeds.csv", index=False)
    # Headline numbers: mean over seeds (std in a separate column) - not the seed ensemble.
    summary = {"feature_set": f"{name} (raw signals)", "model": "cnn", "n_features": np.nan}
    for k in ["MAE", "RMSE", "R2", "case_MAE_mean"]:
        summary[k], summary[f"{k}_std"] = per_seed[k].mean(), per_seed[k].std()
    pd.DataFrame([summary]).to_csv(res_dir / "metrics_cnn.csv", index=False)
    print(f"{name}/cnn over {len(rows)} seeds: MAE {summary['MAE']:.3f} ± {summary['MAE_std']:.3f}, "
          f"R2 {summary['R2']:+.3f} ± {summary['R2_std']:.3f}")
    oof[f"{name}/cnn"] = oof[[f"{name}/cnn_seed{s}" for s in cc["seeds"]]].mean(axis=1)
    oof.to_csv(res_dir / "oof_cnn.csv", index=False)


if __name__ == "__main__":
    main()
