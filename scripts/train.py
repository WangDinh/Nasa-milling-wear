"""Classical models on hand-crafted features, evaluated with leave-one-case-out CV.

Runs every (feature set x model) combination from the config, then refits the best
sensor-based model on all labelled runs and saves it.

Usage: python scripts/train.py --config configs/baseline.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from milling.data import labelled, load_processed
from milling.evaluate import loco_splits, summarize
from milling.features import feature_columns
from milling.models import make_sklearn_model
from milling.utils import load_config, resolve, seed_everything, set_plot_style
from milling.viz import plot_importance, plot_metrics, plot_pred_vs_true, plot_trajectories


def run_experiment(train_df, eval_df, cols, model_name, seed):
    Xtr, ytr = train_df[cols].to_numpy(float), train_df["target"].to_numpy(float)
    Xev = eval_df[cols].to_numpy(float)
    pred = np.full(len(eval_df), np.nan)
    for tr, te, _ in loco_splits(train_df["case"].to_numpy(), eval_df["case"].to_numpy()):
        model = make_sklearn_model(model_name, seed)
        model.fit(Xtr[tr], ytr[tr])
        pred[te] = model.predict(Xev[te])
    return pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    cfg = load_config(parser.parse_args().config)
    seed_everything(cfg["seed"])
    set_plot_style()

    res_dir, fig_dir, model_dir = (resolve(cfg["output"][k]) for k in ["results_dir", "figures_dir", "models_dir"])
    for d in (res_dir, fig_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    table, _ = load_processed(resolve(cfg["data"]["processed_dir"]))
    train_df = labelled(table, cfg["data"]["use_interpolated_vb"])
    eval_df = labelled(table, use_interpolated=False)
    print(f"train pool: {len(train_df)} runs | eval: {len(eval_df)} runs from {eval_df.case.nunique()} cases")

    rows, oof = [], eval_df[["case", "run", "time", "material_name", "target"]].copy()
    for fs_name, parts in cfg["feature_sets"].items():
        cols = feature_columns(table, parts)
        for model_name in cfg["models"]:
            key = f"{fs_name}/{model_name}"
            oof[key] = run_experiment(train_df, eval_df, cols, model_name, cfg["seed"])
            m = summarize(oof.rename(columns={key: "pred"}))
            rows.append({"feature_set": fs_name, "model": model_name, "n_features": len(cols), **m})
            print(f"{key:<28} MAE {m['MAE']:.3f}  RMSE {m['RMSE']:.3f}  R2 {m['R2']:+.3f}")

    metrics = pd.DataFrame(rows).sort_values("MAE")
    metrics.to_csv(res_dir / "metrics_classical.csv", index=False)
    oof.to_csv(res_dir / "oof_classical.csv", index=False)

    # Best model that uses sensor data (sensors are the point of the study).
    sensor_sets = [k for k, v in cfg["feature_sets"].items() if "sensor" in v]
    best = metrics[metrics.feature_set.isin(sensor_sets)].iloc[0]
    best_key = f"{best.feature_set}/{best.model}"
    print(f"\nbest sensor model: {best_key}  (MAE {best.MAE:.3f} mm)")

    plot_metrics(metrics, fig_dir / "model_mae_classical.png")
    plot_pred_vs_true(oof, best_key, fig_dir / "pred_vs_true_classical.png")
    plot_trajectories(oof, best_key, fig_dir / "trajectories_classical.png")

    # Refit on all labelled runs and save.
    cols = feature_columns(table, cfg["feature_sets"][best.feature_set])
    final = make_sklearn_model(best.model, cfg["seed"]).fit(train_df[cols].to_numpy(float), train_df["target"])
    joblib.dump({"model": final, "features": cols, "config": cfg}, model_dir / "best_classical.joblib")

    imp = permutation_importance(final, eval_df[cols].to_numpy(float), eval_df["target"], n_repeats=10,
                                 random_state=cfg["seed"], scoring="neg_mean_absolute_error")
    importance = pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False)
    importance.to_csv(res_dir / "feature_importance.csv", header=["importance"])
    plot_importance(importance, best_key, fig_dir / "feature_importance.png")
    print(f"saved {model_dir / 'best_classical.joblib'}")


if __name__ == "__main__":
    main()
