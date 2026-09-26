"""Stage: hyperparameter optimization with Optuna.

Trains every configured wrapper for ``n_trials`` trials and saves the best
trial for each wrapper under ``pipeline/tmp/optuna/<WrapperName>/``.
"""

import argparse
import gc
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
    # TF is already configured (CUDA runtime, memory growth) by the time this
    # stage runs; importing here keeps the module import side-effect free.
    import tensorflow as tf

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
        features=cfg.get("features"),
        decoder_mode=cfg.get("decoder_mode", "direct"),
        persistence_gate=cfg.get("persistence_gate", False),
    )
    wrapper.loss = cfg.get("loss", "mse")
    wrapper.context_mode = cfg.get("context_mode", "repeat")

    objective_cfg = cfg.get("objective") or {}
    val_blocks = max(1, int(objective_cfg.get("val_blocks", 4)))

    def objective(trial):
        K.clear_session()
        hp = common.OptunaHyperParameters(trial)
        # Fixed batch size (search space focus); LR schedules are fixed to
        # constant inside the wrappers.
        batch_size = int(cfg.get("batch_size", 32))
        wrapper.build(hp)
        callbacks = common.default_callbacks(wrapper, cfg["patience"])
        if cfg.get("pruning", True):
            callbacks.append(common.optuna_pruning_callback(trial))

        started_at = time.perf_counter()
        try:
            history = wrapper.fit(
                epochs=cfg["epochs"],
                batch_size=batch_size,
                verbose=0,
                callbacks=callbacks,
                use_validation=True,
            )
        except (tf.errors.ResourceExhaustedError, tf.errors.InternalError,
                MemoryError) as error:
            # Large sampled configs can exceed the GPU budget mid-training.
            # Discard the trial instead of killing the whole study.
            trial.set_user_attr("oom", str(error)[:200])
            wrapper.model = None
            gc.collect()
            K.clear_session()
            raise optuna.TrialPruned("OOM") from error
        elapsed = time.perf_counter() - started_at

        val_losses = history.history.get("val_loss", [])
        if not val_losses:
            raise RuntimeError("Training returned no val_loss.")

        # Optuna selects on validation error measured under the inference
        # decoder convention (prepare() already rewrites the val decoder
        # inputs), because teacher-forced val_loss is a weak proxy for
        # forecast error (see docs/relatorio_transformer_improvements.md).
        # The objective is the mean MSE over ``val_blocks`` temporal blocks of
        # the validation slice: a single contiguous block sits in one wind
        # regime, and docs/estudo_transformer.md shows that regime-shift
        # robustness is what actually predicts test error.
        val = wrapper.val
        try:
            predicted_scaled = wrapper.model.predict(
                [val["X_encoder"], val["X_decoder"]], batch_size=128, verbose=0
            )[:, :, 0]
        except (tf.errors.ResourceExhaustedError, tf.errors.InternalError,
                MemoryError) as error:
            # A inferencia de validacao (batch 128) pode estourar mesmo quando o
            # treino (batch menor) coube; mesmo remedio do treino: descarta o trial.
            trial.set_user_attr("oom", f"eval: {str(error)[:200]}")
            wrapper.model = None
            gc.collect()
            K.clear_session()
            raise optuna.TrialPruned("OOM (evaluation)") from error
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

        errors = (actuals - predictions).reshape(-1, cfg["output_steps"])
        sequence_indices = np.arange(len(errors))
        # Horizon-weighted MSE: longer horizons count more (matches the
        # horizon_weighted_mse training loss), so Optuna prefers trials that
        # are accurate at the far end of the forecast window.
        from src.models.losses import horizon_weights
        hw = horizon_weights(cfg["output_steps"])
        block_hw_mses = [
            float(np.mean(hw * (errors[block] ** 2)))
            for block in np.array_split(sequence_indices, val_blocks)
        ]
        val_hw_mse = float(np.mean(hw * (errors ** 2)))
        val_mse = float(np.mean(errors ** 2))
        val_mae = float(np.mean(np.abs(errors)))
        val_rmse = float(np.sqrt(val_mse))

        trial.set_user_attr("train_epochs", len(history.history.get("loss", [])))
        trial.set_user_attr("elapsed_sec", elapsed)
        trial.set_user_attr("val_loss", float(np.min(val_losses)))
        trial.set_user_attr("val_mae", val_mae)
        trial.set_user_attr("val_rmse", val_rmse)
        trial.set_user_attr("val_block_mse", block_hw_mses)
        trial.set_user_attr("val_mse", val_mse)
        trial.set_user_attr("val_hw_mse", val_hw_mse)
        print(
            f"[{wrapper_key}] trial {trial.number} block_hw_mse={np.mean(block_hw_mses):.6f} "
            f"val_hw_mse={val_hw_mse:.6f} val_rmse={val_rmse:.6f} "
            f"epochs={trial.user_attrs['train_epochs']} elapsed={elapsed:.1f}s"
        )
        try:
            return float(np.mean(block_hw_mses))
        finally:
            # The wrapper is reused across trials and is the only persistent
            # reference to the trained model, so gc_after_trial alone cannot
            # free it. Drop the reference and tear down the TF graph right
            # after scoring so CPU/GPU memory does not accumulate per trial.
            wrapper.model = None
            gc.collect()
            K.clear_session()

    sampler = optuna.samplers.TPESampler(
        seed=cfg["seed"],
        multivariate=True,
        group=True,
        n_startup_trials=min(20, cfg["n_trials"] // 2),
        n_ei_candidates=64,
    )
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=8)
    storage = f"sqlite:///{out_dir / 'optuna.db'}"
    study_name = cfg.get("study_name", wrapper_key)
    study = optuna.create_study(
        direction="minimize", sampler=sampler, pruner=pruner,
        storage=storage, study_name=study_name, load_if_exists=True,
    )

    started_at = time.perf_counter()
    # Trials persist in the sqlite storage, so a rerun after Ctrl+C / OOM
    # resumes the study where it left off. optuna's n_trials counts *new*
    # trials per optimize() call, so subtract the trials already recorded.
    n_remaining = max(0, cfg["n_trials"] - len(study.trials))
    if n_remaining:
        study.optimize(
            objective,
            n_trials=n_remaining,
            timeout=cfg.get("timeout_sec"),
            gc_after_trial=True,
            show_progress_bar=True,
        )
    else:
        print(
            f"[{wrapper_key}] study already has {len(study.trials)} trials "
            f"(budget {cfg['n_trials']}); nothing left to run."
        )
    elapsed = time.perf_counter() - started_at

    pd.DataFrame(study.trials_dataframe()).to_csv(out_dir / "trials.csv", index=False)
    common.write_json(
        out_dir / "best_trial.json",
        {
            "wrapper": wrapper_key,
            "objective": f"val_block_hw_mse(k={val_blocks})",
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "elapsed_sec": elapsed,
            "seed": cfg["seed"],
            "timeout_sec": cfg.get("timeout_sec"),
        },
    )
    print(f"[{wrapper_key}] best block_mse={study.best_value:.6f}")
    return {
        "best_trial_json": str(out_dir / "best_trial.json"),
        "trials_csv": str(out_dir / "trials.csv"),
    }


def run(config):
    tf = common.setup_tensorflow(config["gpu"]["enabled"])
    print(f"TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")
    tf.keras.utils.set_random_seed(config[STAGE]["seed"])

    cfg = config[STAGE]
    dataset = common.load_dataset(common.resolve(config["paths"]["dataset_csv"]),
                                  keep_raw=tuple(cfg.get("denoise", ())))
    train_df, val_df, _ = common.split_dataset(
        dataset, cfg.get("train_ratio", 0.75), cfg.get("val_ratio", 0.20)
    )
    print(f"Dataset: {len(dataset)} rows | train={len(train_df)} val={len(val_df)}")

    results = {}
    for wrapper_key in cfg["wrappers"]:
        wrapper_name = common.wrapper_factory(wrapper_key)().name
        out_dir = common.resolve(
            Path(config["paths"]["tmp_dir"])
            / cfg.get("storage_subdir", "optuna")
            / wrapper_name
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOptimizing {wrapper_key} ({wrapper_name})...")
        results[wrapper_key] = _run_search(wrapper_key, train_df, val_df, cfg, out_dir)

    return {"artifacts": results}


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization stage.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to a pipeline config file.")
    parser.add_argument("--wrappers", nargs="*", default=None,
                        help="Subset of wrappers to optimize (overrides config).")
    args = parser.parse_args()

    common.ensure_project_root_on_path()
    config = config_module.load_config(args.config)
    if args.wrappers:
        unknown = set(args.wrappers) - set(config[STAGE]["wrappers"])
        if unknown:
            raise SystemExit(f"Unknown wrappers: {sorted(unknown)}")
        config[STAGE]["wrappers"] = args.wrappers
    run(config)


if __name__ == "__main__":
    main()