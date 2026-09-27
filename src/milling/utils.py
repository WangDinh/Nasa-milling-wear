"""Config, paths, seeding and plotting style shared by scripts and notebooks."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]

# Categorical slots (fixed order) and neutrals, validated for colour-vision deficiency.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MATERIAL_COLORS = {"cast_iron": PALETTE[0], "steel": PALETTE[1]}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def load_config(path: str | Path) -> dict:
    with open(ROOT / path if not Path(path).is_absolute() else path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def set_plot_style() -> None:
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.prop_cycle": cycler(color=PALETTE),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK,
            "axes.titleweight": "semibold",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "lines.markersize": 5,
            "font.size": 10,
        }
    )
