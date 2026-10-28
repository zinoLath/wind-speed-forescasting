"""TCN com HPs ANTIGOS + dilatações estáticas C′ (isola o efeito da dilatação).

Mantém TUDO do treino canônico do lidar/8020 para os TCN — hiperparâmetros do
round 1 (lidos de pipeline/tmp/trained_8020/<tcn, tcn_bi>/model.json), pipeline
de denoise (ws100, nível 2), loss horizon-weighted, 60 épocas, seed 42, split
80/20 — e muda SÓ as dilatações: sequência estática C′ (encoder
[1,2,4,8,16,32,36], com o lag de 36 passos direto; decoder [1,2,4,8,16]).

Grava também as profundidades efetivas (``*_dil_depth``) no model.json para
que a remontagem nos relatórios restaure as dilatações (apply_static_dilations).

    python scripts/experiments/tcn_cprime_train.py
    python studies/dm_vs_hpo.py ...   # já compara tcn_cprime? (ver dm_dilations)
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common
from src.models.tcn_hp import STATIC_DILATIONS

WRAPPERS = ("tcn", "tcn_bi")
CANONICAL_DIR = common.resolve("pipeline/tmp/trained_8020")
OUTPUT_DIR = common.resolve("pipeline/tmp/trained_8020_tcn_cprime")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf = common.setup_tensorflow(True)
    print(f"TensorFlow {tf.__version__} | TCN + C′ (HP antigos)")

    dataset = common.load_dataset(common.resolve("data/dataset.csv"))
    train_df, val_df, test_df = common.split_dataset(dataset, 0.80, 0.0)
    print(f"Dataset {len(dataset)} | treino {len(train_df)} | teste {len(test_df)}")

    for wrapper_key in WRAPPERS:
        meta = json.load(open(CANONICAL_DIR / wrapper_key / "model.json"))
        hyperparameters = dict(meta["hyperparameters"])

        from keras import backend as K

        K.clear_session()
        tf.keras.utils.set_random_seed(args.seed)
        wrapper = common.wrapper_factory(wrapper_key)()
        wrapper.prepare(
            train_df, train_df.tail(400),
            input_steps=72, output_steps=36, target_col="ws100_wavelet",
            denoise=("ws100",), denoise_level=2,
        )
        batch_size = int(hyperparameters.get("batch_size", 32))
        if hasattr(wrapper, "schedule_total_steps"):
            steps = int((len(train_df) + batch_size - 1) // batch_size)
            wrapper.schedule_total_steps = steps * args.epochs
        # ÚNICA mudança: dilatações estáticas C′ (encoder até 36)
        wrapper.static_dilations = STATIC_DILATIONS
        wrapper.build(common.FixedHyperParameters(hyperparameters))
        print(f"[{wrapper_key}] static dilations enc="
              f"{STATIC_DILATIONS['encoder']} dec={STATIC_DILATIONS['decoder']}")

        print(f"[{wrapper_key}] treino {args.epochs} epocas (hwmse)...")
        started = time.perf_counter()
        history = wrapper.fit(epochs=args.epochs, batch_size=batch_size,
                              verbose=0, callbacks=None, use_validation=False)
        elapsed = time.perf_counter() - started
        losses = history.history.get("loss", [])
        print(f"[{wrapper_key}] {len(losses)} epocas em {elapsed:.0f}s | "
              f"loss final={losses[-1]:.5f}")

        # profundidades efetivas gravadas (p/ remontagem em relatórios)
        hyperparameters["encoder_dil_depth"] = len(STATIC_DILATIONS["encoder"])
        hyperparameters["decoder_dil_depth"] = len(STATIC_DILATIONS["decoder"])
        model_dir = OUTPUT_DIR / wrapper_key
        model_dir.mkdir(parents=True, exist_ok=True)
        wrapper.model.save(model_dir / "model.keras")
        common.write_json(model_dir / "model.json", {
            "wrapper_key": wrapper_key, "name": wrapper.name,
            "persistence_gate": False, "gate_mode": "static",
            "decoder_mode": getattr(wrapper, "decoder_mode", "direct"),
            "target_mode": getattr(wrapper, "target_mode", "absolute"),
            "normalize": bool(getattr(wrapper, "normalize", True)),
            "hyperparameters": hyperparameters,
            "params_source": "canonica 80/20 (HP antigos) + dilatações C′",
            "static_dilations": STATIC_DILATIONS,
            "input_steps": 72, "output_steps": 36,
            "target_col": "ws100_wavelet", "denoise": ["ws100"],
            "denoise_level": 2, "loss": "horizon_weighted_mse",
            "training": {"epochs_run": len(losses),
                         "epochs_requested": args.epochs, "seed": args.seed,
                         "patience": 0,
                         "final_loss": float(losses[-1]) if losses else None,
                         "best_val_loss": None, "final_val_loss": None,
                         "training_time_sec": round(elapsed, 1)},
            "dataset": {"train_ratio": 0.80, "val_ratio": 0.0,
                        "split": "80/20 sem validacao (HP antigos + C′)"},
        })
        all_df = common.predict_all_horizons(wrapper, test_df)
        all_df["actual_raw"] = dataset["ws100"].reindex(
            all_df["timestamp"]).to_numpy()
        m = common.compute_metrics(all_df["actual_raw"], all_df["predicted"])
        print(f"[{wrapper_key}] MSE teste={m['mse']:.4f}")

    print(f"\nModelos: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
