"""Combine classical + CNN results into reports/results/summary.md and headline figures.

Usage: python scripts/report.py --config configs/baseline.yaml   (after train.py and train_cnn.py)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from milling.utils import INK_MUTED, PALETTE, load_config, resolve, set_plot_style
from milling.viz import plot_pred_vs_true, plot_trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    cfg = load_config(parser.parse_args().config)
    set_plot_style()
    res_dir, fig_dir = resolve(cfg["output"]["results_dir"]), resolve(cfg["output"]["figures_dir"])

    classical = pd.read_csv(res_dir / "metrics_classical.csv")
    cnn = pd.read_csv(res_dir / "metrics_cnn.csv")
    oof_cnn = pd.read_csv(res_dir / "oof_cnn.csv")

    # Headline table: trivial baseline, best model per feature set, CNN.
    rows = [classical[classical.model == "mean"].iloc[0].to_dict() | {"feature_set": "— (predict mean)"}]
    rows += classical[classical.model != "mean"].sort_values("MAE").groupby("feature_set").head(1).to_dict("records")
    rows += cnn.to_dict("records")
    head = pd.DataFrame(rows)
    head["label"] = head.feature_set + " / " + head.model
    head = head.sort_values("MAE")

    fmt = lambda r, k: f"{r[k]:.3f}" + (f" ± {r[k + '_std']:.3f}" if pd.notna(r.get(k + "_std")) else "")
    lines = ["| Inputs | Model | MAE (mm) | RMSE (mm) | R² |", "|---|---|---|---|---|"]
    for _, r in head.iterrows():
        lines.append(f"| {r.feature_set} | {r.model} | {fmt(r, 'MAE')} | {fmt(r, 'RMSE')} | {fmt(r, 'R2')} |")
    full = ["| Feature set | Model | MAE | RMSE | R² |", "|---|---|---|---|---|"]
    for _, r in classical.sort_values(["feature_set", "MAE"]).iterrows():
        full.append(f"| {r.feature_set} | {r.model} | {r.MAE:.3f} | {r.RMSE:.3f} | {r.R2:+.3f} |")
    summary = (
        "# Results — VB regression, leave-one-case-out CV\n\n"
        "145 labelled runs, 16 folds (one held-out insert per fold). CNN: mean ± std over "
        f"{len(cfg['cnn']['seeds'])} seeds.\n\n## Headline\n\n" + "\n".join(lines)
        + "\n\n## All classical experiments\n\n" + "\n".join(full) + "\n"
    )
    (res_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)

    # Headline bar chart.
    fig, ax = plt.subplots(figsize=(8, 3.8))
    h = head.iloc[::-1]
    # Blue = uses sensor signals, orange = cutting-time baseline, grey = no sensor/time information.
    colors = [PALETTE[0] if "sensor" in fs else PALETTE[1] if "time" in fs else INK_MUTED for fs in h.feature_set]
    ax.barh(h.label, h.MAE, color=colors)
    if "MAE_std" in h:
        std = h["MAE_std"].to_numpy()
        has = ~np.isnan(std)
        ax.errorbar(h.MAE[has], np.flatnonzero(has), xerr=std[has], fmt="none", ecolor="#0b0b0b", capsize=3, lw=1)
    for y, v in enumerate(h.MAE):
        ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=8, color="#52514e")
    ax.set(xlabel="MAE (mm), leave-one-case-out — lower is better", title="Predicting flank wear VB from one run")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "headline_mae.png")

    key = [c for c in oof_cnn.columns if c.endswith("/cnn")][0]
    plot_pred_vs_true(oof_cnn, key, fig_dir / "pred_vs_true_cnn.png")
    plot_trajectories(oof_cnn, key, fig_dir / "trajectories_cnn.png")
    print(f"figures -> {fig_dir}")


if __name__ == "__main__":
    main()
