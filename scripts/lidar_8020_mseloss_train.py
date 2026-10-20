"""80/20 com loss = MSE simples (report separado, nada sobrescrito).

Mesmos hp canonicos, mesmo protocolo (60 epocas fixas, seed 42, split 80/20);
a unica diferenca e a funcao de perda: MSE puro em vez da horizon-weighted.
O patch e em runtime (simbolos dos modulos dos wrappers), codigo canonical
intocado.

    python scripts/lidar_8020_mseloss_train.py          # treina os 4
    python tests/relatorio_resultados.py \
        --models-dir pipeline/tmp/trained_8020_mseloss \
        --out-dir data/results/report_8020_mseloss
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common

SEED = 42
WRAPPERS = ("lstm", "lstm_bi", "tcn", "tcn_bi")
OUTPUT_DIR = common.resolve("pipeline/tmp/trained_8020_mseloss")
CANONICAL_DIR = common.resolve("pipeline/tmp/trained_8020")

WRAPPER_MODULES = {
    "lstm": "src.models.s2s_lstm_wrapper",
    "lstm_bi": "src.models.s2s_lstm_bi_wrapper",
    "tcn": "src.models.s2s_tcn_wrapper",
    "tcn_bi": "src.models.s2s_tcn_bi_wrapper",
}


def apply_mse_loss(wrapper_key):
    """Substitui horizon_weighted_mse por 'mse' nos simbolos do modulo."""
    import importlib

    module = importlib.import_module(WRAPPER_MODULES[wrapper_key])
    module.horizon_weighted_mse = lambda output_steps: "mse"
    module.LOSS_NAME = "mse"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--wrappers", nargs="+", default=list(WRAPPERS),
                        choices=WRAPPERS)
    args = parser.parse_args()

    tf = common.setup_tensorflow(True)
    print(f"TensorFlow {tf.__version__} | loss = mse simples")

    dataset = common.load_dataset(common.resolve("data/dataset.csv"))
    train_df, val_df, test_df = common.split_dataset(dataset, 0.80, 0.0)
    val_end = len(train_df)
    print(f"Dataset {len(dataset)} | treino {len(train_df)} | teste {len(test_df)}")

    for wrapper_key in args.wrappers:
        meta = json.load(open(CANONICAL_DIR / wrapper_key / "model.json"))
        hyperparameters = meta["hyperparameters"]
        apply_mse_loss(wrapper_key)

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
        assert wrapper.loss == "mse", wrapper.loss

        print(f"[{wrapper_key}] treino {args.epochs} epocas (loss=mse)...")
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
            "params_source": "canonica 80/20 (loss mse simples)",
            "input_steps": 72,
            "output_steps": 36,
            "target_col": "ws100_wavelet",
            "denoise": ["ws100"],
            "denoise_level": 2,
            "loss": "mse",
            "training": {"epochs_run": len(losses),
                         "epochs_requested": args.epochs,
                         "seed": args.seed, "patience": 0,
                         "final_loss": float(losses[-1]) if losses else None,
                         "best_val_loss": None, "final_val_loss": None,
                         "training_time_sec": round(elapsed, 1)},
            "dataset": {"train_ratio": 0.80, "val_ratio": 0.0,
                        "split": "80/20 sem validacao (loss mse)"},
        })

        all_df = common.predict_all_horizons(wrapper, test_df)
        all_df["actual_raw"] = dataset["ws100"].reindex(
            all_df["timestamp"]).to_numpy()
        metrics = common.compute_metrics(all_df["actual_raw"],
                                         all_df["predicted"])
        print(f"[{wrapper_key}] MSE teste={metrics['mse']:.4f}")

    print(f"\nModelos: {OUTPUT_DIR}")
    print("Relatorio: python tests/relatorio_resultados.py --models-dir "
          "pipeline/tmp/trained_8020_mseloss --out-dir "
          "data/results/report_8020_mseloss")


if __name__ == "__main__":
    main()
