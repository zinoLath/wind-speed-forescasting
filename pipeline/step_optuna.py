"""Stage: hyperparameter optimization with Optuna.

Trains every configured wrapper for ``n_trials`` trials and saves the best
trial for each wrapper under ``pipeline/tmp/optuna/<WrapperName>/``.
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import optuna
import pandas as pd

from src import common
from pipeline import config as config_module

STAGE = "optuna"


def _run_search(wrapper_key, train_df, val_df, cfg, out_dir):
    from keras import backend as K
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau

    # Data preparation (wavelet denoising, scalers, sequences) does not depend
    # on the trial hyperparameters, so it runs once and is reused by every
    # trial through the same wrapper instance (build() replaces only the model).
    wrapper = common.wrapper_factory(wrapper_key)()
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
        steps_per_epoch = int(np.ceil(len(train_df) / cfg["batch_size"]))
        wrapper.schedule_total_steps = steps_per_epoch * cfg["epochs"]
    wrapper.loss = cfg.get("loss", "mse")

    def objective(trial):
        K.clear_session()
        wrapper.build(common.OptunaHyperParameters(trial))
        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=cfg["patience"], restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=max(2, cfg["patience"] // 2),
                min_lr=1e-6,
            ),
        ]
        if cfg.get("pruning", True):
            callbacks.append(common.optuna_pruning_callback(trial))

        started_at = time.perf_counter()
        history = wrapper.fit(
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            verbose=0,
            callbacks=callbacks,
            use_validation=True,
        )
        elapsed = time.perf_counter() - started_at

        val_losses = history.history.get("val_loss", [])
        if not val_losses:
            raise RuntimeError("Training returned no val_loss.")

        best_val_loss = float(np.min(val_losses))

        # Optuna selects on validation RMSE measured under the inference
        # decoder convention (prepare() already rewrites the val decoder
        # inputs), because teacher-forced val_loss is a weak proxy for
        # forecast error (see docs/relatorio_transformer_improvements.md).
        # RMSE (not MAE) is the selection metric so trials with large errors
        # are punished harder.
        val = wrapper.val
        predicted_scaled = wrapper.model.predict(
            [val["X_encoder"], val["X_decoder"]], batch_size=256, verbose=0
        )[:, :, 0]
        actual_scaled = val["y_decoder"][:, :, 0]
        if wrapper.target_mode == "residual":
            persistence = val["X_encoder"][:, -1, wrapper.target_col_index]
            predicted_scaled = predicted_scaled + persistence
            actual_scaled = actual_scaled + persistence
        predictions = wrapper.scaler_target.inverse_transform(
            predicted_scaled.reshape(-1, 1)
        ).ravel()
        actuals = wrapper.scaler_target.inverse_transform(
            actual_scaled.reshape(-1, 1)
        ).ravel()
        val_rmse = float(np.sqrt(np.mean((actuals - predictions) ** 2)))
        val_mae = float(np.mean(np.abs(actuals - predictions)))

        trial.set_user_attr("train_epochs", len(history.history.get("loss", [])))
        trial.set_user_attr("elapsed_sec", elapsed)
        trial.set_user_attr("val_loss", best_val_loss)
        trial.set_user_attr("val_mae", val_mae)
        print(
            f"[{wrapper_key}] trial {trial.number} val_rmse={val_rmse:.6f} "
            f"val_mae={val_mae:.6f} val_loss={best_val_loss:.6f} "
            f"epochs={trial.user_attrs['train_epochs']} elapsed={elapsed:.1f}s"
        )
        return val_rmse

    sampler = optuna.samplers.TPESampler(
        seed=cfg["seed"],
        multivariate=True,
        group=True,
        n_startup_trials=min(20, cfg["n_trials"] // 2),
        n_ei_candidates=64,
    )
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=8)
    storage = f"sqlite:///{out_dir / 'optuna.db'}"
    study = optuna.create_study(
        direction="minimize", sampler=sampler, pruner=pruner,
        storage=storage, study_name=wrapper_key, load_if_exists=True,
    )

    started_at = time.perf_counter()
    study.optimize(
        objective,
        n_trials=cfg["n_trials"],
        timeout=cfg.get("timeout_sec"),
        gc_after_trial=True,
        show_progress_bar=True,
    )
    elapsed = time.perf_counter() - started_at

    pd.DataFrame(study.trials_dataframe()).to_csv(out_dir / "trials.csv", index=False)
    common.write_json(
        out_dir / "best_trial.json",
        {
            "wrapper": wrapper_key,
            "objective": "val_rmse",
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "elapsed_sec": elapsed,
            "seed": cfg["seed"],
            "timeout_sec": cfg.get("timeout_sec"),
        },
    )
    print(f"[{wrapper_key}] best val_rmse={study.best_value:.6f}")
    return {
        "best_trial_json": str(out_dir / "best_trial.json"),
        "trials_csv": str(out_dir / "trials.csv"),
    }


def run(config):
    tf = common.setup_tensorflow(config["gpu"]["enabled"])
    print(f"TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")
    tf.keras.utils.set_random_seed(config[STAGE]["seed"])

    cfg = config[STAGE]
    dataset = common.load_dataset(common.resolve(config["paths"]["dataset_csv"]))
    train_df, val_df, _ = common.split_dataset(
        dataset, cfg.get("train_ratio", 0.75), cfg.get("val_ratio", 0.20)
    )
    print(f"Dataset: {len(dataset)} rows | train={len(train_df)} val={len(val_df)}")

    results = {}
    for wrapper_key in cfg["wrappers"]:
        wrapper_name = common.wrapper_factory(wrapper_key)().name
        out_dir = common.resolve(
            Path(config["paths"]["tmp_dir"]) / "optuna" / wrapper_name
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOptimizing {wrapper_key} ({wrapper_name})...")
        results[wrapper_key] = _run_search(wrapper_key, train_df, val_df, cfg, out_dir)

    return {"artifacts": results}


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization stage.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to a pipeline config file.")
    args = parser.parse_args()

    common.ensure_project_root_on_path()
    config = config_module.load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()