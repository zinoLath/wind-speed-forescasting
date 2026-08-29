"""Fair benchmark harness for the Seq2Seq wrappers.

Trains a wrapper with fixed hyperparameters under a configurable budget and
reports val_loss plus (optionally) walk-forward MAE/RMSE/R2 metrics on the same
test region used by wfo_optuna.py. Results are appended to a JSONL log so that
experiments can be tracked across cycles.
"""

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
    os.environ["LD_LIBRARY_PATH"] = ":".join(valid_dirs + ([current_ld] if current_ld else []))
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

from keras import backend as K  # noqa: E402
from keras.callbacks import EarlyStopping, ReduceLROnPlateau  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

from src.models.s2s_lstm_bi_wrapper import S2SLSTMBidirectionalWrapper  # noqa: E402
from src.models.s2s_lstm_wrapper import S2SLSTMWrapper  # noqa: E402
from src.models.s2s_tcn_bi_wrapper import S2STCNBidirectionalWrapper  # noqa: E402
from src.models.s2s_tcn_wrapper import S2STCNWrapper  # noqa: E402
from src.models.s2s_transformer_autoreg_wrapper import S2STransformerAutoregressiveWrapper  # noqa: E402
from src.models.s2s_transformer_preln_wrapper import S2STransformerPrelnWrapper  # noqa: E402
from src.models.s2s_transformer_wrapper import S2STransformerWrapper  # noqa: E402
from tests.optuna_all_hyperparameters import load_dataset, split_dataset  # noqa: E402
from tests.wfo_optuna import FixedHyperParameters, load_best_params  # noqa: E402


WRAPPERS = {
    "lstm": S2SLSTMWrapper,
    "lstm_bi": S2SLSTMBidirectionalWrapper,
    "tcn": S2STCNWrapper,
    "tcn_bi": S2STCNBidirectionalWrapper,
    "transformer": S2STransformerWrapper,
    "transformer_preln": S2STransformerPrelnWrapper,
    "transformer_autoreg": S2STransformerAutoregressiveWrapper,
}


def build_wrapper(wrapper_key, dataset, params, input_steps, output_steps, epochs, batch_size,
                  use_schedule, decoder_mode, target_mode, seed, weight_decay=None, clipnorm=None,
                  loss="mse"):
    train_df, val_df = split_dataset(dataset)
    train_end = len(train_df)
    val_end = train_end + len(val_df)

    K.clear_session()
    import tensorflow as tf
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    wrapper = WRAPPERS[wrapper_key]()
    wrapper.prepare(
        train_df,
        val_df,
        input_steps=input_steps,
        output_steps=output_steps,
        target_col="ws100_wavelet",
        denoise=["ws100"],
        denoise_level=2,
        decoder_mode=decoder_mode,
        target_mode=target_mode,
    )
    if use_schedule and hasattr(wrapper, "schedule_total_steps"):
        n_train = len(train_df)
        steps_per_epoch = int(np.ceil(n_train / batch_size))
        wrapper.schedule_total_steps = steps_per_epoch * epochs
    if weight_decay:
        wrapper.weight_decay = weight_decay
    if clipnorm:
        wrapper.clipnorm = clipnorm
    if loss != "mse":
        wrapper.loss = loss

    wrapper.build(FixedHyperParameters(params))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ]
    if not use_schedule:
        # With a warmup+cosine schedule the LR annealing is already handled;
        # ReduceLROnPlateau would fight the schedule.
        callbacks.append(
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)
        )

    history = wrapper.fit(
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
        use_validation=True,
    )
    return wrapper, train_end, val_end, history


def evaluate_wfo(wrapper, dataset, test_start):
    started_at = time.perf_counter()
    predictions, actuals = wrapper.rolling_forecast(dataset, test_start=test_start)
    elapsed = time.perf_counter() - started_at
    predictions = predictions.ravel()
    actuals = actuals.ravel()
    metrics = {
        "test_samples": len(actuals),
        "mae": mean_absolute_error(actuals, predictions),
        "mse": mean_squared_error(actuals, predictions),
        "rmse": np.sqrt(mean_squared_error(actuals, predictions)),
        "r2": r2_score(actuals, predictions),
        "prediction_time_sec": elapsed,
    }
    return metrics, predictions, actuals


def parse_args():
    parser = argparse.ArgumentParser(description="Fair benchmark of Seq2Seq wrappers.")
    parser.add_argument("--wrapper", choices=WRAPPERS, required=True)
    parser.add_argument("--params", type=str, default=None,
                        help='JSON dict of fixed hyperparameters.')
    parser.add_argument("--from-optuna", action="store_true",
                        help="Load best params from data/results/optuna for this wrapper.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-steps", type=int, default=72)
    parser.add_argument("--output-steps", type=int, default=36)
    parser.add_argument("--schedule", action="store_true",
                        help="Enable warmup+cosine LR schedule for transformer wrappers.")
    parser.add_argument("--no-wfo", action="store_true", help="Skip rolling forecast.")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--tag", type=str, default="", help="Short label for the log.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decoder-mode", choices=["teacher_forcing", "direct"], default="teacher_forcing")
    parser.add_argument("--target-mode", choices=["absolute", "residual"], default="absolute")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    parser.add_argument("--clipnorm", type=float, default=None, help="Global gradient norm clipping.")
    parser.add_argument("--loss", choices=["mse", "mae", "huber"], default="mse")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.params and args.from_optuna:
        raise ValueError("Use either --params or --from-optuna, not both.")

    if args.params:
        params = json.loads(args.params)
    else:
        _, params = load_best_params(args.wrapper)

    dataset = load_dataset()
    wrapper, train_end, val_end, history = build_wrapper(
        args.wrapper, dataset, params, args.input_steps, args.output_steps,
        args.epochs, args.batch_size, args.schedule, args.decoder_mode,
        args.target_mode, args.seed, args.weight_decay, args.clipnorm, args.loss,
    )

    best_val_loss = float(np.min(history.history["val_loss"]))
    train_epochs = len(history.history.get("loss", []))
    record = {
        "wrapper": args.wrapper,
        "name": wrapper.name,
        "tag": args.tag,
        "params": params,
        "epochs_budget": args.epochs,
        "train_epochs": train_epochs,
        "best_val_loss": best_val_loss,
        "schedule": args.schedule,
        "decoder_mode": args.decoder_mode,
        "target_mode": args.target_mode,
        "seed": args.seed,
        "loss": args.loss,
        "weight_decay": args.weight_decay,
    }

    if not args.no_wfo:
        record["wfo"], preds, act = evaluate_wfo(wrapper, dataset, test_start=val_end)

    log_path = PROJECT_ROOT / "data" / "results" / "benchmark_transformer.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if not args.no_wfo and args.save_predictions:
        import pandas as pd
        pred_path = PROJECT_ROOT / "data" / "results" / "benchmark_preds" / (
            f"{args.wrapper}_{args.tag or 'run'}.csv"
        )
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"prediction": preds, "actual": act}).to_csv(
            pred_path, index=False
        )
        print(f"predictions saved to {pred_path}")

    wfo_txt = (
        f" | MAE={record['wfo']['mae']:.4f} RMSE={record['wfo']['rmse']:.4f} "
        f"R2={record['wfo']['r2']:.4f}"
        if "wfo" in record
        else ""
    )
    print(f"[{wrapper.name}] tag={args.tag} val_loss={best_val_loss:.6f} "
          f"epochs={train_epochs}/{args.epochs}{wfo_txt}")


if __name__ == "__main__":
    main()