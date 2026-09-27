"""Result plots shared by the training and report scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from milling.utils import INK_MUTED, MATERIAL_COLORS, PALETTE


def plot_metrics(metrics: pd.DataFrame, out: Path, title: str = "VB regression: feature set × model") -> None:
    sets = list(dict.fromkeys(metrics.feature_set))
    sets.sort(key=lambda s: metrics[metrics.feature_set == s].MAE.min(), reverse=True)
    models = list(dict.fromkeys(metrics.sort_values("model").model))
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.8 / len(models)
    for k, m in enumerate(models):
        vals = [metrics[(metrics.feature_set == s) & (metrics.model == m)].MAE.iloc[0] for s in sets]
        ax.bar(np.arange(len(sets)) + (k - (len(models) - 1) / 2) * w, vals, width=w * 0.92,
               color=PALETTE[k], label=m)
    ax.set_xticks(range(len(sets)), sets)
    ax.set(ylabel="MAE (mm), leave-one-case-out", title=title)
    ax.legend(ncol=len(models), loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)


def plot_pred_vs_true(oof: pd.DataFrame, key: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    lim = [0, max(oof.target.max(), oof[key].max()) * 1.05]
    ax.plot(lim, lim, ls="--", color=INK_MUTED, lw=1)
    for mat, c in MATERIAL_COLORS.items():
        d = oof[oof.material_name == mat]
        ax.scatter(d.target, d[key], s=28, color=c, edgecolor="white", lw=0.8, label=mat.replace("_", " "))
    ax.set(xlim=lim, ylim=lim, xlabel="measured VB (mm)", ylabel="predicted VB (mm)", title=f"Out-of-fold: {key}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)


def plot_trajectories(oof: pd.DataFrame, key: str, out: Path) -> None:
    cases = sorted(oof.case.unique())
    ncol = 4
    fig, axes = plt.subplots(int(np.ceil(len(cases) / ncol)), ncol, figsize=(13, 9), sharey=True)
    for ax, case in zip(axes.flat, cases):
        g = oof[oof.case == case].sort_values("time")
        ax.plot(g.time, g.target, "-o", ms=3, color=PALETTE[0], label="measured")
        ax.plot(g.time, g[key], "-s", ms=3, color=PALETTE[1], label="predicted")
        ax.set_title(f"case {case} ({g.material_name.iloc[0].replace('_', ' ')})", fontsize=9)
    for ax in axes.flat[len(cases):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.supxlabel("cutting time (min)")
    fig.supylabel("VB (mm)")
    fig.suptitle(f"Held-out tool-wear trajectories — {key}", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(out)


def plot_importance(importance: pd.Series, key: str, out: Path, top: int = 15) -> None:
    s = importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(s.index, s.values, color=PALETTE[0])
    ax.set(xlabel="permutation importance (Δ MAE, mm)", title=f"Top-{top} features — {key} (refit on all runs, in-sample)")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out)
