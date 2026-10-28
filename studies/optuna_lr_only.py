"""Optuna focado apenas em LR: learning_rate (log) + schedule (constant|warmup_cosine).

Os demais hiperparâmetros ficam fixos nos vencedores do round 3
(``pipeline/tmp/optuna/<name>/best_trial.json``). Objetivo idêntico ao
step_optuna: MSE por horizonte ponderado, média sobre blocos temporais de
validação, decoder de inferência. Estudos por wrapper em sqlite (retomável).

Uso: python tests/optuna_lr_only.py [--wrappers ...] [--n-trials 30]
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import optuna

from src import common
from src.models.losses import horizon_weights

EPOCHS = 150
BATCH = 32
PATIENCE = 12
VAL_BLOCKS = 4
STORAGE_SUBDIR = "optuna_lr"


class FixedWithTrialLR:
    """HP adapter: tudo fixo, exceto learning_rate (vem do trial)."""

    def __init__(self, fixed, trial):
        self.fixed = fixed
        self.trial = trial

    def Float(self, name, min_value, max_value, step=None, sampling=None, default=None):
        if name == "learning_rate":
            if sampling == "LOG":
                return self.trial.suggest_float(name, min_value, max_value, log=True)
            return self.trial.suggest_float(name, min_value, max_value, step=step)
        return float(self.fixed.get(name, default if default is not None else min_value))

    def Int(self, name, min_value, max_value, step=1, default=None):
        return int(self.fixed.get(name, default if default is not None else min_value))

    def Choice(self, name, values, default=None):
        return self.fixed.get(name, values[0])


def objective_factory(wrapper, train_df, val_df, fixed, n_train_sequences, cfg):
    from keras import backend as K
    import tensorflow as tf

    def objective(trial):
        K.clear_session()
        sched = trial.suggest_categorical("schedule", ["constant", "warmup_cosine"])
        wrapper.schedule_mode_override = sched if sched != "constant" else None
        batch_size = int(cfg.get("batch_size", BATCH))
        if sched == "warmup_cosine":
            steps_per_epoch = int(np.ceil(n_train_sequences / batch_size))
            wrapper.schedule_total_steps = steps_per_epoch * EPOCHS
        else:
            wrapper.schedule_total_steps = None

        wrapper.build(FixedWithTrialLR(fixed, trial))
        callbacks = common.default_callbacks(wrapper, cfg["patience"])
        if cfg.get("pruning", True):
            callbacks.append(common.optuna_pruning_callback(trial))

        started = time.perf_counter()
        try:
            history = wrapper.fit(epochs=EPOCHS, batch_size=batch_size, verbose=0,
                                  callbacks=callbacks, use_validation=True)
        except (tf.errors.ResourceExhaustedError, tf.errors.InternalError,
                MemoryError) as error:
            trial.set_user_attr("oom", str(error)[:200])
            wrapper.model = None
            gc.collect()
            K.clear_session()
            raise optuna.TrialPruned("OOM") from error
        elapsed = time.perf_counter() - started

        val_losses = history.history.get("val_loss", [])
        if not val_losses:
            raise RuntimeError("Treino sem val_loss.")

        val = wrapper.val
        try:
            predicted_scaled = wrapper.model.predict(
                [val["X_encoder"], val["X_decoder"]], batch_size=128, verbose=0
            )[:, :, 0]
        except (tf.errors.ResourceExhaustedError, tf.errors.InternalError,
                MemoryError) as error:
            trial.set_user_attr("oom", f"eval: {str(error)[:200]}")
            wrapper.model = None
            gc.collect()
            K.clear_session()
            raise optuna.TrialPruned("OOM (evaluation)") from error

        actual_scaled = val["y_decoder"][:, :, 0]
        predictions = wrapper.scaler_target.inverse_transform(
            predicted_scaled.reshape(-1, 1)
        ).ravel()
        actuals = wrapper.scaler_target.inverse_transform(
            actual_scaled.reshape(-1, 1)
        ).ravel()

        errors = (actuals - predictions).reshape(-1, cfg["output_steps"])
        hw = horizon_weights(cfg["output_steps"])
        sequence_indices = np.arange(len(errors))
        block_scores = [
            float(np.mean(hw * (errors[block] ** 2)))
            for block in np.array_split(sequence_indices, VAL_BLOCKS)
        ]
        val_mae = float(np.mean(np.abs(errors)))

        trial.set_user_attr("train_epochs", len(history.history.get("loss", [])))
        trial.set_user_attr("elapsed_sec", elapsed)
        trial.set_user_attr("schedule", sched)
        trial.set_user_attr("val_mae", val_mae)
        print(f"trial {trial.number} sched={sched} "
              f"lr={trial.params.get('learning_rate'):.2e} "
              f"block_hw_mse={np.mean(block_scores):.6f} "
              f"épocas={trial.user_attrs['train_epochs']} {elapsed:.0f}s", flush=True)
        try:
            return float(np.mean(block_scores))
        finally:
            wrapper.model = None
            gc.collect()
            K.clear_session()

    return objective


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wrappers", nargs="*", default=["lstm", "lstm_bi", "tcn", "tcn_bi"])
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()

    tf = common.setup_tensorflow(True)
    print(f"TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")
    tf.keras.utils.set_random_seed(42)

    dataset = common.load_dataset(common.resolve("data/dataset.csv"),
                                  keep_raw=("ws40", "ws100", "v40"))
    train_df, val_df, _ = common.split_dataset(dataset, 0.75, 0.20)
    cfg = {"batch_size": BATCH, "patience": PATIENCE, "pruning": True,
           "output_steps": 36, "epochs": EPOCHS}

    for wrapper_key in args.wrappers:
        name = common.wrapper_factory(wrapper_key)().name
        best_path = common.find_optuna_result(wrapper_key)
        if best_path is None:
            raise SystemExit(f"Sem best_trial.json para {wrapper_key}")
        fixed = json.load(open(best_path, encoding="utf-8"))["best_params"]

        out_dir = common.resolve(Path("pipeline/tmp") / STORAGE_SUBDIR / name)
        out_dir.mkdir(parents=True, exist_ok=True)

        wrapper = common.wrapper_factory(wrapper_key)()
        wrapper.prepare(
            train_df, val_df,
            input_steps=72, output_steps=36,
            target_col="ws100_wavelet",
            denoise=("ws100", "ws40", "v40"), denoise_level=1,
            decoder_mode="direct", persistence_gate=False,
        )
        n_train_sequences = wrapper.train["X_encoder"].shape[0]

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, group=True,
                                               n_startup_trials=8, n_ei_candidates=32),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=8),
            storage=f"sqlite:///{out_dir / 'optuna.db'}",
            study_name="lr_only", load_if_exists=True,
        )
        n_remaining = max(0, args.n_trials - len(study.trials))
        if n_remaining:
            study.optimize(
                objective_factory(wrapper, train_df, val_df, fixed, n_train_sequences, cfg),
                n_trials=n_remaining, gc_after_trial=True, show_progress_bar=True,
            )

        best = study.best_trial
        common.write_json(out_dir / "best_trial.json", {
            "wrapper": wrapper_key,
            "objective": f"val_block_hw_mse(k={VAL_BLOCKS})",
            "best_value": study.best_value,
            "best_params": {
                **fixed,
                "learning_rate": best.params["learning_rate"],
                "schedule": best.user_attrs.get("schedule", best.params.get("schedule")),
            },
            "n_trials": len(study.trials),
            "seed": 42,
        })
        print(f"[{wrapper_key}] melhor LR-only: sched={best.user_attrs.get('schedule')} "
              f"lr={best.params['learning_rate']:.2e} "
              f"block_hw_mse={study.best_value:.6f} ({len(study.trials)} trials)")


if __name__ == "__main__":
    main()