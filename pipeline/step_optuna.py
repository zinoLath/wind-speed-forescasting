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

from pipeline import common, config as config_module

STAGE = "optuna"


def _run_search(wrapper_key, train_df, val_df, cfg, out_dir):
    from keras import backend as K
    from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau

    class PruningCallback(Callback):
        """Keras callback that reports val_loss to Optuna for trial pruning."""

        def __init__(self, trial):
            super().__init__()
            self.trial = trial

        def on_epoch_end(self, epoch, logs=None):
            val_loss = (logs or {}).get("val_loss")
            if val_loss is None:
                return
            self.trial.report(float(val_loss), step=epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned(f"Trial pruned at epoch {epoch}")

    def objective(trial):
        K.clear_session()
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
            callbacks.append(PruningCallback(trial))

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
        trial.set_user_attr("train_epochs", len(history.history.get("loss", [])))
        trial.set_user_attr("elapsed_sec", elapsed)
        print(
            f"[{wrapper_key}] trial {trial.number} best_val={best_val_loss:.6f} "
            f"epochs={trial.user_attrs['train_epochs']} elapsed={elapsed:.1f}s"
        )
        return best_val_loss

    sampler = optuna.samplers.TPESampler(
        seed=cfg["seed"],
        multivariate=True,
        group=True,
        n_startup_trials=min(20, cfg["n_trials"] // 2),
        n_ei_candidates=64,
    )
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=8)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

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
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "elapsed_sec": elapsed,
            "seed": cfg["seed"],
            "timeout_sec": cfg.get("timeout_sec"),
        },
    )
    print(f"[{wrapper_key}] best val_loss={study.best_value:.6f}")
    return {
        "best_trial_json": str(out_dir / "best_trial.json"),
        "trials_csv": str(out_dir / "trials.csv"),
    }


def run(config):
    tf = common.setup_tensorflow(config["gpu"]["enabled"])
    print(f"TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")

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