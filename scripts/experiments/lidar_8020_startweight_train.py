"""80/20 com loss horizon-weighted INVERTIDA (ênfase no início do horizonte).

Mesmos hp canonicos, mesmo protocolo (60 epocas fixas, seed 42, split 80/20);
a unica diferenca e a funcao de perda: pesos decrescentes com o horizonte
(espelho da horizon-weighted canonica — passo 1 pesa ~1.3x, passo 36 ~0.7x).
O patch e em runtime (simbolos dos modulos dos wrappers), codigo canonical
intocado.

    python scripts/experiments/lidar_8020_startweight_train.py                # treina as LSTMs
    python scripts/experiments/lidar_8020_startweight_train.py \
        --wrappers lstm lstm_bi tcn tcn_bi                        # todas
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from src import common

SEED = 42
WRAPPERS = ("lstm", "lstm_bi")
OUTPUT_DIR = common.resolve("pipeline/tmp/trained_8020_startweight")
CANONICAL_DIR = common.resolve("pipeline/tmp/trained_8020")
LOSS_NAME = "start_weighted_mse"

WRAPPER_MODULES = {
    "lstm": "src.models.s2s_lstm_wrapper",
    "lstm_bi": "src.models.s2s_lstm_bi_wrapper",
    "tcn": "src.models.s2s_tcn_wrapper",
    "tcn_bi": "src.models.s2s_tcn_bi_wrapper",
}


def start_weighted_mse(output_steps):
    """Keras loss: MSE com pesos decrescentes (inicio do horizonte ~1.3x)."""
    steps = np.arange(1, output_steps + 1, dtype=np.float32)
    ramp = 1.0 + (output_steps + 1 - steps) / float(output_steps)
    weights = (ramp / ramp.mean()).astype(np.float32)[None, :, None]

    def loss(y_true, y_pred):
        return tf.reduce_mean(weights * tf.square(y_pred - y_true))

    return loss


def apply_start_weighted_loss(wrapper_key):
    """Substitui horizon_weighted_mse pela versao start-weighted no modulo."""
    import importlib

    module = importlib.import_module(WRAPPER_MODULES[wrapper_key])
    module.horizon_weighted_mse = start_weighted_mse
    module.LOSS_NAME = LOSS_NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--wrappers", nargs="+", default=list(WRAPPERS),
                        choices=WRAPPER_MODULES)
    args = parser.parse_args()

    tf.config.run_functions_eagerly(False)
    print(f"TensorFlow {tf.__version__} | loss = {LOSS_NAME} "
          "(pesos decrescentes: inicio do horizonte ~1.3x)")

    dataset = common.load_dataset(common.resolve("data/dataset.csv"))
    train_df, val_df, test_df = common.split_dataset(dataset, 0.80, 0.0)
    print(f"Dataset {len(dataset)} | treino {len(train_df)} | teste {len(test_df)}")

    for wrapper_key in args.wrappers:
        meta = json.load(open(CANONICAL_DIR / wrapper_key / "model.json"))
        hyperparameters = meta["hyperparameters"]
        apply_start_weighted_loss(wrapper_key)

        from keras import backend as K

        K.clear_session()
        tf.keras.utils.set_random_seed(args.seed)
        wrapper = common.wrapper_factory(wrapper_key)()
        wrapper.prepare(
            train_df,
            train_df.tail(400),
            input_steps=72,
            output_steps=36,
            target_col="ws100_wavelet",
            denoise=("ws100",),
            denoise_level=2,
        )
        batch_size = int(hyperparameters.get("batch_size", 32))
        if hasattr(wrapper, "schedule_total_steps"):
            steps = int((len(train_df) + batch_size - 1) // batch_size)
            wrapper.schedule_total_steps = steps * args.epochs
        wrapper.build(common.FixedHyperParameters(hyperparameters))
        assert wrapper.loss == LOSS_NAME, wrapper.loss

        print(f"[{wrapper_key}] treino {args.epochs} epocas (loss={LOSS_NAME})...")
        started = time.perf_counter()
        history = wrapper.fit(epochs=args.epochs, batch_size=batch_size,
                              verbose=0, callbacks=None, use_validation=False)
        elapsed = time.perf_counter() - started
        losses = history.history.get("loss", [])
        print(f"[{wrapper_key}] {len(losses)} epocas em {elapsed:.0f}s | "
              f"loss final={losses[-1]:.5f}")

        model_dir = OUTPUT_DIR / wrapper_key
        model_dir.mkdir(parents=True, exist_ok=True)
        wrapper.model.save(model_dir / "model.keras")
        common.write_json(model_dir / "model.json", {
            "wrapper_key": wrapper_key,
            "name": wrapper.name,
            "persistence_gate": False,
            "gate_mode": "static",
            "decoder_mode": getattr(wrapper, "decoder_mode", "direct"),
            "target_mode": getattr(wrapper, "target_mode", "absolute"),
            "normalize": bool(getattr(wrapper, "normalize", True)),
            "hyperparameters": hyperparameters,
            "params_source": "canonica 80/20 (loss start-weighted)",
            "input_steps": 72,
            "output_steps": 36,
            "target_col": "ws100_wavelet",
            "denoise": ["ws100"],
            "denoise_level": 2,
            "loss": LOSS_NAME,
            "training": {"epochs_run": len(losses),
                         "epochs_requested": args.epochs,
                         "seed": args.seed, "patience": 0,
                         "final_loss": float(losses[-1]) if losses else None,
                         "best_val_loss": None, "final_val_loss": None,
                         "training_time_sec": round(elapsed, 1)},
            "dataset": {"train_ratio": 0.80, "val_ratio": 0.0,
                        "split": "80/20 sem validacao (loss start-weighted)"},
        })

        all_df = common.predict_all_horizons(wrapper, test_df)
        all_df["actual_raw"] = dataset["ws100"].reindex(
            all_df["timestamp"]).to_numpy()
        metrics = common.compute_metrics(all_df["actual_raw"],
                                         all_df["predicted"])
        mae = metrics["mae"]
        print(f"[{wrapper_key}] MSE teste={metrics['mse']:.4f} MAE={mae:.4f}")

    print(f"\nModelos: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
