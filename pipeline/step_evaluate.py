"""Stage: evaluate a trained model on the forecast dataset.

Loads a saved model (``model.keras`` + ``model.json``) from ``pipeline/tmp/``
by default, or from any path given in the config (``overload`` option), and
measures its performance on the held-out test portion of ``dataset.csv``.

Outputs predictions and metrics as CSVs/JSON, plus plots:
  - all-horizon predictions (every forecast origin over the test period)
  - rolling forecast (one-step-ahead at the forecast horizon)
  - persistence baseline
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pipeline import common, config as config_module

STAGE = "evaluate"


def _plot_evaluation(roll_df, all_df, per_horizon, output_dir):
    """Generate the evaluation plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(roll_df["timestamp"], roll_df["actual"], label="actual", alpha=0.8)
    ax.plot(roll_df["timestamp"], roll_df["predicted"], label="predicted", alpha=0.8)
    ax.set_title("Rolling forecast vs actual (forecast horizon)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Wind speed (m/s)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "rolling_forecast_timeseries.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    per_horizon.plot(x="horizon", y="mae", ax=axes[0], marker="o", legend=False)
    axes[0].set_title("MAE by forecast horizon")
    axes[0].set_xlabel("Horizon (timesteps)")
    axes[0].grid(True)
    per_horizon.plot(x="horizon", y="rmse", ax=axes[1], marker="o", legend=False)
    axes[1].set_title("RMSE by forecast horizon")
    axes[1].set_xlabel("Horizon (timesteps)")
    axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "per_horizon_metrics.png", dpi=150)
    plt.close(fig)

    errors = all_df["predicted"] - all_df["actual"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=60, edgecolor="black")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_title("Prediction error distribution")
    ax.set_xlabel("Error (predicted - actual)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "error_histogram.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(all_df["actual"], all_df["predicted"], s=2, alpha=0.4)
    lims = [min(all_df[["actual", "predicted"]].min()), max(all_df[["actual", "predicted"]].max())]
    ax.plot(lims, lims, color="red", linestyle="--", label="perfect fit")
    ax.set_title("Predicted vs actual (all horizons)")
    ax.set_xlabel("Actual (m/s)")
    ax.set_ylabel("Predicted (m/s)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_actual_predicted.png", dpi=150)
    plt.close(fig)

    return [
        "rolling_forecast_timeseries.png",
        "per_horizon_metrics.png",
        "error_histogram.png",
        "scatter_actual_predicted.png",
    ]


def run(config):
    tf = common.setup_tensorflow(config["gpu"]["enabled"])
    print(f"TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")

    from keras import backend as K

    cfg = config[STAGE]
    tmp_dir = common.resolve(config["paths"]["tmp_dir"])

    model_path = common.resolve(cfg["model_path"] or tmp_dir / "model.keras")
    metadata_path = common.resolve(cfg["model_json_path"] or tmp_dir / "model.json")
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Model files not found ({model_path}, {metadata_path}). "
            "Run the train stage first or set evaluate.model_path / model_json_path."
        )

    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)

    wrapper_key = metadata["wrapper_key"]
    wrapper_class = common.wrapper_factory(wrapper_key)
    print(f"Evaluating {metadata['model_class']} ({model_path.name})")

    dataset = common.load_dataset(common.resolve(config["paths"]["dataset_csv"]))
    train_ratio = metadata["dataset"].get("train_ratio", 0.75)
    val_ratio = metadata["dataset"].get("val_ratio", 0.20)
    train_df, val_df, test_df = common.split_dataset(dataset, train_ratio, val_ratio)
    val_end = len(train_df) + len(val_df)
    print(f"Dataset: {len(dataset)} rows | train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    K.clear_session()
    wrapper = wrapper_class()
    wrapper.prepare(
        train_df,
        val_df,
        input_steps=metadata["input_steps"],
        output_steps=metadata["output_steps"],
        target_col=metadata["target_col"],
        denoise=metadata["denoise"],
        denoise_level=metadata["denoise_level"],
    )
    wrapper.build(common.FixedHyperParameters(metadata["hyperparameters"]))
    wrapper.model.load_weights(model_path)

    all_df = common.predict_all_horizons(wrapper, test_df)
    with contextlib.redirect_stdout(io.StringIO()):
        roll_pred, roll_actual = wrapper.rolling_forecast(dataset, test_start=val_end)
    roll_df = pd.DataFrame(
        {
            "timestamp": dataset.index[val_end : val_end + len(roll_actual)],
            "actual": roll_actual.ravel(),
            "predicted": roll_pred.ravel(),
        }
    )

    metrics = {"model": metadata}
    metrics["all_horizons"] = common.compute_metrics(all_df["actual"], all_df["predicted"])
    metrics["persistence"] = common.compute_metrics(all_df["actual"], all_df["persistence"])
    metrics["rolling"] = common.compute_metrics(roll_df["actual"], roll_df["predicted"])

    per_horizon_df = common.horizon_metrics(all_df)
    metrics["by_horizon"] = per_horizon_df.to_dict(orient="records")

    out_dir = common.resolve(cfg["output_dir"] or tmp_dir / "evaluate" / wrapper.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_df.to_csv(out_dir / "predictions_all_horizons.csv", index=False)
    roll_df.to_csv(out_dir / "predictions_rolling.csv", index=False)
    per_horizon_df.to_csv(out_dir / "per_horizon_metrics.csv", index=False)
    common.write_json(out_dir / "metrics.json", metrics)

    plots = _plot_evaluation(roll_df, all_df, per_horizon_df, out_dir)

    print(
        f"[{wrapper_key}] rolling MAE={metrics['rolling']['mae']:.4f} "
        f"RMSE={metrics['rolling']['rmse']:.4f}"
    )
    print(
        f"[{wrapper_key}] all-horizon MAE={metrics['all_horizons']['mae']:.4f} "
        f"RMSE={metrics['all_horizons']['rmse']:.4f}"
    )
    print(f"Results saved to {out_dir}")

    return {
        "artifacts": {
            "predictions_all_horizons_csv": str(out_dir / "predictions_all_horizons.csv"),
            "predictions_rolling_csv": str(out_dir / "predictions_rolling.csv"),
            "per_horizon_metrics_csv": str(out_dir / "per_horizon_metrics.csv"),
            "metrics_json": str(out_dir / "metrics.json"),
            "plots": [str(out_dir / p) for p in plots],
        },
        "metrics": {
            "rolling": metrics["rolling"],
            "all_horizons": metrics["all_horizons"],
            "persistence": metrics["persistence"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluation stage.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to a pipeline config file.")
    args = parser.parse_args()

    common.ensure_project_root_on_path()
    config = config_module.load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()