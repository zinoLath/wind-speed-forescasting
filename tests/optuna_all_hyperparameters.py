import argparse
import ctypes
import glob
import json
import os
import site
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def configure_tensorflow_gpu_runtime():
    """Configura as bibliotecas CUDA do ambiente antes de importar o TensorFlow."""
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    candidate_dirs = []

    for site_packages in site.getsitepackages():
        candidate_dirs.extend(glob.glob(os.path.join(site_packages, "nvidia", "*", "lib")))

    valid_dirs = [path for path in dict.fromkeys(candidate_dirs) if os.path.isdir(path)]
    if not valid_dirs:
        return []

    os.environ["LD_LIBRARY_PATH"] = ":".join(
        valid_dirs + ([current_ld] if current_ld else [])
    )

    for library_name in (
        "libcudart.so.12",
        "libcublas.so.12",
        "libcudnn.so.9",
        "libcusolver.so.11",
    ):
        for library_dir in valid_dirs:
            library_path = os.path.join(library_dir, library_name)
            if os.path.exists(library_path):
                try:
                    ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break

    return valid_dirs


cuda_lib_dirs = configure_tensorflow_gpu_runtime()

import tensorflow as tf

# Allow GPU memory to grow so that large TCN/attention graphs do not fail to
# allocate on the first attempt. This must be set before any GPU allocation.
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except (RuntimeError, ValueError):
        pass

import keras

from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.models.s2s_lstm_bi_wrapper import S2SLSTMBidirectionalWrapper
from src.models.s2s_lstm_wrapper import S2SLSTMWrapper
from src.models.s2s_tcn_bi_wrapper import S2STCNBidirectionalWrapper
from src.models.s2s_tcn_lstm_wrapper import S2STCNLSTMWrapper
from src.models.s2s_tcn_wrapper import S2STCNWrapper
from src.models.s2s_transformer_preln_wrapper import S2STransformerPrelnWrapper


def verify_cuda_gpu():
    """Confirma que o TensorFlow consegue executar uma operação real na GPU."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError(
            "Nenhuma GPU CUDA foi detectada pelo TensorFlow; busca Optuna cancelada."
        )

    try:
        with tf.device("/GPU:0"):
            result = tf.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
        result.numpy()
    except (tf.errors.OpError, RuntimeError) as exc:
        raise RuntimeError(
            f"A GPU foi detectada, mas uma operação CUDA falhou; busca Optuna cancelada: {exc}"
        ) from exc

    print(f"CUDA lib dirs found: {len(cuda_lib_dirs)}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPUs: {gpus}")


class OptunaHyperParameters:
    def __init__(self, trial: optuna.trial.Trial):
        self.trial = trial

    def Float(
        self,
        name,
        min_value,
        max_value,
        step=None,
        sampling=None,
        default=None,
    ):
        if sampling == "LOG":
            return self.trial.suggest_float(name, min_value, max_value, log=True)

        if step is not None:
            return self.trial.suggest_float(name, min_value, max_value, step=step)

        return self.trial.suggest_float(name, min_value, max_value)

    def Int(self, name, min_value, max_value, step=1, default=None):
        return self.trial.suggest_int(name, min_value, max_value, step=step)

    def Choice(self, name, values, default=None):
        return self.trial.suggest_categorical(name, values)


class OptunaPruningCallback(keras.callbacks.Callback):
    def __init__(self, trial: optuna.trial.Trial):
        super().__init__()
        self.trial = trial

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return
        self.trial.report(float(val_loss), step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Trial pruned at epoch {epoch}")


class TrialTrainingProgressCallback(keras.callbacks.Callback):
    def __init__(self, trial_number: int, epoch_log_interval: int = 5):
        super().__init__()
        self.trial_number = trial_number
        self.epoch_log_interval = max(0, int(epoch_log_interval))
        self.best_val_loss = None

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_log_interval == 0:
            return
        if (epoch + 1) % self.epoch_log_interval != 0:
            return

        logs = logs or {}
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        if val_loss is not None:
            self.best_val_loss = (
                float(val_loss)
                if self.best_val_loss is None
                else min(self.best_val_loss, float(val_loss))
            )

        print(
            f"[trial {self.trial_number}] epoch={epoch + 1} "
            f"loss={float(loss):.6f} val_loss={float(val_loss):.6f} "
            f"best_val={float(self.best_val_loss):.6f}"
        )


def load_dataset() -> pd.DataFrame:
    dataset = pd.read_csv(PROJECT_ROOT / "data" / "dataset.csv")

    for col in dataset.columns:
        if col in {"year", "month", "day", "hour", "minute", "id"}:
            continue
        if pd.api.types.is_numeric_dtype(dataset[col]):
            continue
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce")
        median_value = dataset[col].median()
        if pd.notna(median_value):
            dataset[col] = dataset[col].fillna(median_value)

    dataset["timestamp"] = pd.to_datetime(dataset["id"], format="mixed")
    dataset = dataset.sort_values("timestamp").set_index("timestamp")

    cols_to_drop = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "press",
        "humid",
        "temp",
        "id",
        "cis1",
        "cis2",
        "cis3",
        "cis4",
        "cis5",
        "cis6",
        "cis7",
        "cis8",
        "cis9",
        "cis10",
        "cis11",
        "cis12",
        "cis13",
        "cis14",
        "cis15",
        "cis16",
        "cis17",
        "cis18",
        "cis19",
        "wdisp40",
        "wdisp50",
        "wdisp60",
        "wdisp70",
        "wdisp80",
        "wdisp90",
        "wdisp100",
        "wdisp110",
        "wdisp120",
        "wdisp130",
        "wdisp140",
        "wdisp150",
        "wdisp160",
        "wdisp170",
        "wdisp180",
        "wdisp190",
        "wdisp200",
        "wdisp220",
        "wdisp240",
        "wdisp260",
        "vertdisp40",
        "vertdisp50",
        "vertdisp60",
        "vertdisp70",
        "vertdisp80",
        "vertdisp90",
        "vertdisp100",
        "vertdisp110",
        "vertdisp120",
        "vertdisp130",
        "vertdisp140",
        "vertdisp150",
        "vertdisp160",
        "vertdisp170",
        "vertdisp180",
        "vertdisp190",
        "vertdisp200",
        "vertdisp220",
        "vertdisp240",
        "vertdisp260",
        "wdir150",
        "wdir160",
        "wdir170",
        "wdir180",
        "wdir190",
        "wdir200",
        "wdir220",
        "wdir240",
        "wdir260",
        "verts150",
        "verts160",
        "verts170",
        "verts180",
        "verts190",
        "verts200",
        "verts220",
        "verts240",
        "verts260",
        "ws150",
        "ws160",
        "ws170",
        "ws180",
        "ws190",
        "ws200",
        "ws220",
        "ws240",
        "ws260",
    ]
    dataset = dataset.drop(columns=[c for c in cols_to_drop if c in dataset.columns], errors="ignore")

    heights = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 220, 240, 260]
    cols_to_rename = {}
    for height in heights:
        cols_to_rename[f"wdir{height}"] = f"dir{height}"
        cols_to_rename[f"verts{height}"] = f"v{height}"

    dataset = dataset.rename(columns=cols_to_rename)
    dataset = dataset.apply(pd.to_numeric, errors="coerce")
    dataset = dataset.interpolate(limit_direction="both").ffill().bfill()

    return dataset


def split_dataset(dataset: pd.DataFrame):
    train_ratio = 0.75
    val_ratio = 0.20
    train_end = int(len(dataset) * train_ratio)
    val_end = train_end + int(len(dataset) * val_ratio)
    return dataset.iloc[:train_end].copy(), dataset.iloc[train_end:val_end].copy()


def build_objective(
    wrapper_factory,
    train_df,
    val_df,
    input_steps,
    output_steps,
    denoise_level,
    epochs,
    epoch_log_interval,
):
    def objective(trial):
        trial_started_at = time.perf_counter()
        K.clear_session()
        wrapper = wrapper_factory()
        wrapper.prepare(
            train_df,
            val_df,
            input_steps=input_steps,
            output_steps=output_steps,
            target_col="ws100_wavelet",
            denoise=["ws100"],
            denoise_level=denoise_level,
        )

        if hasattr(wrapper, "schedule_total_steps"):
            # Transformer wrappers train better with warmup + cosine decay.
            n_train = len(train_df)
            steps_per_epoch = int(np.ceil(n_train / 32))
            wrapper.schedule_total_steps = steps_per_epoch * epochs

        wrapper.build(OptunaHyperParameters(trial))
        print(f"[trial {trial.number}] params={trial.params}")
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
            OptunaPruningCallback(trial),
            TrialTrainingProgressCallback(
                trial_number=trial.number,
                epoch_log_interval=epoch_log_interval,
            ),
        ]

        history = wrapper.fit(
            epochs=epochs,
            batch_size=32,
            verbose=0,
            callbacks=callbacks,
            use_validation=True,
        )

        val_losses = history.history.get("val_loss", [])
        if not val_losses:
            raise RuntimeError("O treinamento não retornou val_loss.")

        best_val_loss = float(np.min(val_losses))
        elapsed = time.perf_counter() - trial_started_at
        trial.set_user_attr("train_epochs", len(history.history.get("loss", [])))
        trial.set_user_attr("elapsed_sec", elapsed)
        print(
            f"[trial {trial.number}] complete best_val={best_val_loss:.6f} "
            f"epochs={len(history.history.get('loss', []))} elapsed={elapsed:.1f}s"
        )

        return best_val_loss

    return objective


def run_search(
    wrapper_factory,
    dataset,
    n_trials,
    epochs,
    input_steps,
    output_steps,
    denoise_level,
    seed,
    timeout,
    epoch_log_interval,
):
    train_df, val_df = split_dataset(dataset)
    wrapper = wrapper_factory()
    results_dir = PROJECT_ROOT / "data" / "results" / "optuna" / wrapper.name
    results_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(
        seed=seed,
        multivariate=True,
        group=True,
        n_startup_trials=20,
        n_ei_candidates=64,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=8,
        interval_steps=2,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    objective = build_objective(
        wrapper_factory=wrapper_factory,
        train_df=train_df,
        val_df=val_df,
        input_steps=input_steps,
        output_steps=output_steps,
        denoise_level=denoise_level,
        epochs=epochs,
        epoch_log_interval=epoch_log_interval,
    )

    print(
        f"[{wrapper.name}] search config: trials={n_trials}, epochs={epochs}, "
        f"input_steps={input_steps}, output_steps={output_steps}, "
        f"denoise_level={denoise_level}, timeout={timeout}, seed={seed}"
    )
    print(
        f"[{wrapper.name}] dataset split: train={len(train_df)} rows, val={len(val_df)} rows"
    )

    def on_trial_finished(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        duration_sec = (
            trial.duration.total_seconds() if trial.duration is not None else None
        )
        duration_txt = f"{duration_sec:.1f}s" if duration_sec is not None else "n/a"
        if trial.state == optuna.trial.TrialState.COMPLETE:
            best_trial = study.best_trial
            marker = " NEW_BEST" if trial.number == best_trial.number else ""
            print(
                f"[{wrapper.name}] trial {trial.number} complete "
                f"value={trial.value:.6f} duration={duration_txt} "
                f"best={best_trial.value:.6f} (trial={best_trial.number}){marker}"
            )
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(
                f"[{wrapper.name}] trial {trial.number} pruned "
                f"duration={duration_txt}"
            )
        else:
            print(
                f"[{wrapper.name}] trial {trial.number} state={trial.state.name} "
                f"duration={duration_txt}"
            )

    started_at = time.perf_counter()
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        gc_after_trial=True,
        show_progress_bar=True,
        callbacks=[on_trial_finished],
    )
    elapsed = time.perf_counter() - started_at

    pd.DataFrame(study.trials_dataframe()).to_csv(results_dir / "trials.csv", index=False)
    with open(results_dir / "best_trial.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "wrapper": wrapper.name,
                "best_value": study.best_value,
                "best_params": study.best_params,
                "n_trials": len(study.trials),
                "elapsed_sec": elapsed,
                "seed": seed,
                "timeout_sec": timeout,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"{wrapper.name}: melhor val_loss = {study.best_value:.6f}")
    print(f"{wrapper.name}: melhores hiperparâmetros = {study.best_params}")
    print(f"{wrapper.name}: resultados salvos em {results_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Otimização de hiperparâmetros com Optuna para os wrappers Seq2Seq.")
    parser.add_argument("--n-trials", type=int, default=100, help="Quantidade de trials por modelo.")
    parser.add_argument("--epochs", type=int, default=60, help="Quantidade de épocas por trial.")
    parser.add_argument("--input-steps", type=int, default=72, help="Tamanho da janela de entrada.")
    parser.add_argument("--output-steps", type=int, default=36, help="Horizonte de saída.")
    parser.add_argument("--denoise-level", type=int, default=2, help="Nível de wavelet denoising.")
    parser.add_argument("--timeout", type=int, default=None, help="Tempo máximo por modelo (segundos).")
    parser.add_argument("--seed", type=int, default=42, help="Seed para o sampler do Optuna.")
    parser.add_argument(
        "--epoch-log-interval",
        type=int,
        default=5,
        help="Loga métricas a cada N épocas dentro de cada trial (0 desativa).",
    )
    parser.add_argument(
        "--wrappers",
        nargs="*",
        default=["lstm", "lstm_bi", "tcn", "tcn_bi", "tcn_lstm", "transformer"],
        choices=["lstm", "lstm_bi", "tcn", "tcn_bi", "tcn_lstm", "transformer"],
        help="Wrappers a serem otimizados.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    verify_cuda_gpu()
    dataset = load_dataset()

    wrapper_factories = {
        "lstm": S2SLSTMWrapper,
        "lstm_bi": S2SLSTMBidirectionalWrapper,
        "tcn": S2STCNWrapper,
        "tcn_bi": S2STCNBidirectionalWrapper,
        "tcn_lstm": S2STCNLSTMWrapper,
        "transformer": S2STransformerPrelnWrapper,
    }

    for wrapper_key in args.wrappers:
        print(f"Iniciando Optuna para {wrapper_key}...")
        run_search(
            wrapper_factory=wrapper_factories[wrapper_key],
            dataset=dataset,
            n_trials=args.n_trials,
            epochs=args.epochs,
            input_steps=args.input_steps,
            output_steps=args.output_steps,
            denoise_level=args.denoise_level,
            seed=args.seed,
            timeout=args.timeout,
            epoch_log_interval=args.epoch_log_interval,
        )


if __name__ == "__main__":
    main()