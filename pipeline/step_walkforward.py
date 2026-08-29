"""Stage: walk-forward validation with periodic retraining.

Sliding-window simulation: for each window the model is retrained on the last
``train_window_days`` of data (default 60 = 2 months) and evaluated on the
following ``test_window_days`` (default 30 = 1 month). The window advances by
``step_days``, which may be smaller than the test window so the evaluation
periods can overlap.

By default the input is the last imputed dataset produced by the impute stage;
the pipeline falls back to ``data/dataset.csv`` when no imputed dataset exists.
"""

import argparse
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from pipeline import common, config as config_module

STAGE = "walkforward"

ROWS_PER_DAY = 24 * 6  # 10-minute data


def load_input_frame(path):
    """Load an input CSV and normalise it to a DatetimeIndex."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    return df


def resolve_input_path(config, cfg):
    """Pick the input file: explicit path > last imputed dataset > dataset.csv."""
    if cfg.get("input"):
        return common.resolve(cfg["input"])

    tmp_dir = common.resolve(config["paths"]["tmp_dir"])
    impute_cfg = config.get("impute", {})
    for method in impute_cfg.get("methods", []):
        candidate = tmp_dir / "imputed" / method / "imputed.csv"
        if candidate.is_file():
            return candidate

    return common.resolve(config["paths"]["dataset_csv"])


def _train_and_predict(wrapper, train_df, eval_df, cfg, hyperparameters):
    """Train on *train_df* (split internally into train/val) and predict *eval_df*.

    Returns (predictions_df, window_metrics_dict).
    """
    from keras import backend as K
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau

    train_split = int(len(train_df) * cfg.get("train_val_ratio", 0.8))
    train_part = train_df.iloc[:train_split].copy()
    val_part = train_df.iloc[train_split:].copy()

    K.clear_session()
    wrapper.prepare(
        train_part,
        val_part,
        input_steps=cfg["input_steps"],
        output_steps=cfg["output_steps"],
        target_col=cfg["target_col"],
        denoise=cfg["denoise"],
        denoise_level=cfg["denoise_level"],
    )
    if hasattr(wrapper, "schedule_total_steps"):
        steps_per_epoch = int((len(train_part) + cfg["batch_size"] - 1) // cfg["batch_size"])
        wrapper.schedule_total_steps = steps_per_epoch * cfg["epochs"]

    wrapper.build(common.FixedHyperParameters(hyperparameters))
    history = wrapper.fit(
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        verbose=cfg.get("verbose", 0),
        callbacks=[
            EarlyStopping(
                monitor="val_loss", patience=cfg["patience"], restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=max(2, cfg["patience"] // 2),
                min_lr=1e-6,
            ),
        ],
        use_validation=True,
    )

    eval_window = pd.concat([train_df.iloc[-cfg["input_steps"]:], eval_df])
    predictions = common.predict_all_horizons(wrapper, eval_window)

    metrics = common.compute_metrics(predictions["actual"], predictions["predicted"])
    metrics["persistence_mae"] = common.compute_metrics(
        predictions["actual"], predictions["persistence"]
    )["mae"]
    metrics["train_epochs"] = len(history.history.get("loss", []))
    return predictions, metrics


def _evaluate_wrapper(wrapper_key, df, cfg, config, out_dir):
    """Run the full walk-forward for one model and save per-window artifacts."""
    hyperparameters, params_source = common.resolve_hyperparameters(wrapper_key, cfg, config)
    wrapper_class = common.wrapper_factory(wrapper_key)
    wrapper_name = wrapper_class().name

    model_dir = out_dir / wrapper_name
    model_dir.mkdir(parents=True, exist_ok=True)

    train_rows = cfg["train_window_days"] * ROWS_PER_DAY
    test_rows = cfg["test_window_days"] * ROWS_PER_DAY
    step_rows = cfg["step_days"] * ROWS_PER_DAY
    n = len(df)

    all_predictions = []
    window_records = []
    test_start = train_rows
    window_idx = 0

    while test_start + test_rows <= n:
        if cfg.get("max_windows") is not None and window_idx >= cfg["max_windows"]:
            break
        train_start = max(0, test_start - train_rows)
        train_df = df.iloc[train_start:test_start].copy()
        test_df = df.iloc[test_start:test_start + test_rows].copy()

        print(
            f"[{wrapper_name}] window {window_idx}: train "
            f"{df.index[train_start]}->{df.index[test_start - 1]}, "
            f"test {df.index[test_start]}->{df.index[test_start + test_rows - 1]}"
        )

        import time
        started_at = time.perf_counter()
        wrapper = wrapper_class()
        pred_df, metrics = _train_and_predict(wrapper, train_df, test_df, cfg, hyperparameters)
        elapsed = time.perf_counter() - started_at

        metrics.update(
            {
                "window": window_idx,
                "train_start": str(df.index[train_start]),
                "train_end": str(df.index[test_start - 1]),
                "test_start": str(df.index[test_start]),
                "test_end": str(df.index[test_start + test_rows - 1]),
                "elapsed_sec": round(elapsed, 3),
                "params_source": params_source,
            }
        )
        window_records.append(metrics)
        pred_df.to_csv(model_dir / f"predictions_window_{window_idx:04d}.csv", index=False)
        print(
            f"[{wrapper_name}] window {window_idx}: MAE={metrics['mae']:.4f} "
            f"RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f} elapsed={elapsed:.1f}s"
        )
        all_predictions.append(pred_df)

        window_idx += 1
        test_start += step_rows

    if not window_records:
        print(
            f"[{wrapper_name}] no windows generated. Dataset has {n} rows; "
            f"need at least {train_rows + test_rows}. Reduce train/test window days "
            "or provide a longer dataset."
        )
        return {}

    pd.DataFrame(window_records).to_csv(model_dir / "metrics.csv", index=False)

    all_df = pd.concat(all_predictions, ignore_index=True)
    all_df.to_csv(model_dir / "predictions_all.csv", index=False)
    overall = common.compute_metrics(all_df["actual"], all_df["predicted"])
    overall["persistence_mae"] = common.compute_metrics(
        all_df["actual"], all_df["persistence"]
    )["mae"]
    overall["by_horizon"] = common.horizon_metrics(all_df).to_dict(orient="records")

    summary = {
        "settings": {
            "wrapper": wrapper_name,
            "wrapper_key": wrapper_key,
            "input_file": str(cfg["_input_path"]),
            "train_window_days": cfg["train_window_days"],
            "test_window_days": cfg["test_window_days"],
            "step_days": cfg["step_days"],
            "input_steps": cfg["input_steps"],
            "output_steps": cfg["output_steps"],
            "target_col": cfg["target_col"],
            "denoise_level": cfg["denoise_level"],
            "epochs": cfg["epochs"],
            "batch_size": cfg["batch_size"],
            "patience": cfg["patience"],
            "params_source": params_source,
            "hyperparameters": hyperparameters,
        },
        "windows": window_records,
        "overall": overall,
    }
    common.write_json(model_dir / "metrics.json", summary)

    print(f"[{wrapper_name}] {len(window_records)} windows | "
          f"overall MAE={overall['mae']:.4f} RMSE={overall['rmse']:.4f}")
    return {
        "predictions_all_csv": str(model_dir / "predictions_all.csv"),
        "metrics_csv": str(model_dir / "metrics.csv"),
        "metrics_json": str(model_dir / "metrics.json"),
        "overall_metrics": overall,
    }


def run(config):
    tf = common.setup_tensorflow(config["gpu"]["enabled"])
    print(f"TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")

    cfg = config[STAGE]
    input_path = resolve_input_path(config, cfg)
    print(f"Input dataset: {input_path}")
    df = load_input_frame(input_path)

    metadata_columns = [c for c in ("imputed", "_target_imputed") if c in df.columns]
    if metadata_columns:
        df = df.drop(columns=metadata_columns)
    print(f"Dataset shape: {df.shape} | range {df.index.min()} -> {df.index.max()}")
    if cfg.get("seed") is not None:
        tf.keras.utils.set_random_seed(cfg["seed"])

    cfg["_input_path"] = input_path
    out_dir = common.resolve(
        cfg["output_dir"] or (Path(config["paths"]["tmp_dir"]) / "walkforward")
    )

    results = {}
    for wrapper_key in cfg["wrappers"]:
        print(f"\n=== Walk-forward: {wrapper_key} ===")
        results[wrapper_key] = _evaluate_wrapper(wrapper_key, df, cfg, config, out_dir)

    return {"artifacts": results}


def main():
    parser = argparse.ArgumentParser(description="Walk-forward validation stage.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to a pipeline config file.")
    args = parser.parse_args()

    common.ensure_project_root_on_path()
    config = config_module.load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()