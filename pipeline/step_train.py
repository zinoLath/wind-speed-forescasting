"""Stage: train the best model and persist it.

Trains one wrapper on the forecast dataset and saves the model to
``models/best_model/model.keras`` together with a ``model.json`` that
describes the model context (class, hyperparameters, training metrics,
timestamps). A copy of both files is mirrored to ``pipeline/tmp/`` so the
later stages can consume them.
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import common, config as config_module

STAGE = "train"


def run(config):
    tf = common.setup_tensorflow(config["gpu"]["enabled"])
    print(f"TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")

    from keras import backend as K
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau

    cfg = config[STAGE]
    wrapper_key = cfg["wrapper"]
    wrapper_class = common.wrapper_factory(wrapper_key)

    hyperparameters, params_source = common.resolve_hyperparameters(wrapper_key, cfg, config)
    print(f"[{wrapper_key}] hyperparameters source: {params_source}")

    dataset = common.load_dataset(common.resolve(config["paths"]["dataset_csv"]))
    train_ratio = cfg.get("train_ratio", 0.75)
    val_ratio = cfg.get("val_ratio", 0.20)
    train_df, val_df, test_df = common.split_dataset(dataset, train_ratio, val_ratio)
    print(
        f"Dataset: {len(dataset)} rows | train={len(train_df)} "
        f"val={len(val_df)} test={len(test_df)}"
    )

    tf.keras.utils.set_random_seed(cfg["seed"])
    K.clear_session()
    wrapper = wrapper_class()
    wrapper.prepare(
        train_df,
        val_df,
        input_steps=cfg["input_steps"],
        output_steps=cfg["output_steps"],
        target_col=cfg["target_col"],
        denoise=cfg["denoise"],
        denoise_level=cfg["denoise_level"],
    )
    if hasattr(wrapper, "schedule_total_steps"):
        steps_per_epoch = int((len(train_df) + cfg["batch_size"] - 1) // cfg["batch_size"])
        wrapper.schedule_total_steps = steps_per_epoch * cfg["epochs"]

    wrapper.build(common.FixedHyperParameters(hyperparameters))
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=cfg["patience"], restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, cfg["patience"] // 2),
            min_lr=1e-6,
        ),
    ]

    started_at = time.perf_counter()
    history = wrapper.fit(
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        verbose=cfg.get("verbose", 1),
        callbacks=callbacks,
        use_validation=True,
    )
    training_time_sec = time.perf_counter() - started_at

    losses = history.history.get("loss", [])
    val_losses = history.history.get("val_loss", [])
    model_metadata = {
        "model_class": wrapper_class.__name__,
        "wrapper_key": wrapper_key,
        "name": wrapper.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": hyperparameters,
        "params_source": params_source,
        "input_steps": cfg["input_steps"],
        "output_steps": cfg["output_steps"],
        "target_col": cfg["target_col"],
        "denoise": cfg["denoise"],
        "denoise_level": cfg["denoise_level"],
        "decoder_mode": getattr(wrapper, "decoder_mode", "teacher_forcing"),
        "target_mode": getattr(wrapper, "target_mode", "absolute"),
        "training": {
            "epochs_requested": cfg["epochs"],
            "epochs_run": len(losses),
            "batch_size": cfg["batch_size"],
            "patience": cfg["patience"],
            "training_time_sec": round(training_time_sec, 3),
            "final_loss": float(losses[-1]) if losses else None,
            "final_val_loss": float(val_losses[-1]) if val_losses else None,
            "best_val_loss": float(min(val_losses)) if val_losses else None,
            "seed": cfg["seed"],
        },
        "dataset": {
            "source": str(config["paths"]["dataset_csv"]),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
        },
    }

    models_dir = common.resolve(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "model.keras"
    metadata_path = models_dir / "model.json"
    wrapper.model.save(model_path)
    common.write_json(metadata_path, model_metadata)

    tmp_dir = common.resolve(config["paths"]["tmp_dir"])
    tmp_dir.mkdir(parents=True, exist_ok=True)
    common.write_json(tmp_dir / "model.json", model_metadata)
    wrapper.model.save(tmp_dir / "model.keras")

    print(
        f"[{wrapper_key}] trained in {training_time_sec:.1f}s over "
        f"{len(losses)} epochs | best val_loss={model_metadata['training']['best_val_loss']:.6f}"
    )
    print(f"Saved model -> {model_path}")
    print(f"Saved metadata -> {metadata_path}")

    return {
        "artifacts": {
            "model_keras": str(model_path),
            "model_json": str(metadata_path),
            "tmp_model_keras": str(tmp_dir / "model.keras"),
            "tmp_model_json": str(tmp_dir / "model.json"),
        },
        "metadata": model_metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Training stage.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to a pipeline config file.")
    args = parser.parse_args()

    common.ensure_project_root_on_path()
    config = config_module.load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()