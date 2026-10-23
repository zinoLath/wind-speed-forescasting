"""Rodada FINAL do artigo SBSR (80/20) — lineup canônico unificado.

Decisão do autor (22/10/2026, finding `rodada-final-sbsr`):

- **TCN/TCN-Bi**: 46 canônicas + ``humid``/``temp`` (48 colunas) — a
  inclusão de meteorologia faz sentido para a narrativa do paper;
- **LSTM/LSTM-Bi**: featurex (46 + ``cis1..19`` + ``humid``/``temp``,
  67 colunas) — o cisalhamento é o diferencial das recorrentes;
- **MSE pura em todos os wrappers** (ITILDE-Q descartada);
- HPs canônicos (TCN com dilatações C′ estáticas; LSTMs do HPO 80/20),
  seed 42, 60 épocas, split 80/20 sem validação, alvo ``ws100_wavelet``.

    python scripts/lidar_8020_sbsr_train.py
    python tests/relatorio_resultados.py \
        --models-dir pipeline/tmp/trained_8020_sbsr \
        --out-dir data/results/report_8020_sbsr
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
EPOCHS = 60
OUTPUT_DIR = common.resolve("pipeline/tmp/trained_8020_sbsr")
CANON_DIR = common.resolve("pipeline/tmp/trained_8020_itildeq")
KEEP_METEO = ("humid", "temp")
KEEP_FEATUREX = KEEP_METEO + tuple(f"cis{i}" for i in range(1, 20))
KEEP_BY_WRAPPER = {
    "tcn": KEEP_METEO,
    "tcn_bi": KEEP_METEO,
    "lstm": KEEP_FEATUREX,
    "lstm_bi": KEEP_FEATUREX,
}
WRAPPER_ORDER = ("lstm", "lstm_bi", "tcn", "tcn_bi")
FEATURE_SET_LABEL = {
    KEEP_METEO: ("perfil 46 + humid/temp (48 col; sem press/disp/vdisp)"),
    KEEP_FEATUREX: ("perfil 46 + cis1..19 + humid/temp (67 col; "
                    "sem press/disp/vdisp)"),
}


def canonical_meta(wrapper_key):
    path = CANON_DIR / wrapper_key / "model.json"
    return json.load(open(path, encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gate", choices=("none", "static", "dynamic"),
                        default="none",
                        help="persistence gate no decoder (default none = "
                             "rodada canonica SBSR)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = common.resolve(args.output_dir)
    tf = common.setup_tensorflow(True)
    print(f"TensorFlow {tf.__version__} | rodada FINAL SBSR 8020 "
          f"(HPs da canônica {CANON_DIR.name}, MSE pura, seed {SEED}, "
          f"gate={args.gate})")

    datasets = {}
    for keep in dict.fromkeys(KEEP_BY_WRAPPER.values()):
        datasets[keep] = common.load_dataset(
            common.resolve("data/dataset.csv"), keep_raw=keep)
        print(f"dataset keep={len(keep)} extras: "
              f"{datasets[keep].shape[0]} linhas x "
              f"{datasets[keep].shape[1]} colunas")

    for wrapper_key in WRAPPER_ORDER:
        keep = KEEP_BY_WRAPPER[wrapper_key]
        dataset = datasets[keep]
        canon = canonical_meta(wrapper_key)
        hyperparameters = dict(canon["hyperparameters"])
        loss = "mse"
        static = canon.get("static_dilations")

        from keras import backend as K

        K.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        wrapper = common.wrapper_factory(wrapper_key)()
        gate = args.gate != "none"
        train_df, val_df, test_df = common.split_dataset(dataset, 0.80, 0.0)
        wrapper.prepare(
            train_df,
            train_df.tail(400),
            input_steps=72,
            output_steps=36,
            target_col="ws100_wavelet",
            denoise=("ws100",),
            denoise_level=2,
            persistence_gate=gate,
        )
        wrapper.gate_mode = args.gate if gate else "static"

        batch_size = int(hyperparameters.get("batch_size", 32))
        if hasattr(wrapper, "schedule_total_steps"):
            steps = int((len(train_df) + batch_size - 1) // batch_size)
            wrapper.schedule_total_steps = steps * EPOCHS

        wrapper.loss_mode_override = loss
        if static:
            wrapper.static_dilations = static

        wrapper.build(common.FixedHyperParameters(hyperparameters))
        assert wrapper.loss == loss, (wrapper_key, wrapper.loss)
        print(f"[{wrapper_key}] keep={len(keep)} extras loss={loss} "
              f"static={bool(static)} entradas={wrapper.num_encoder_features}")

        print(f"[{wrapper_key}] treino {EPOCHS} epocas (loss={loss})...")
        started = time.perf_counter()
        history = wrapper.fit(epochs=EPOCHS, batch_size=batch_size,
                              verbose=0, callbacks=None, use_validation=False)
        elapsed = time.perf_counter() - started
        losses_hist = history.history.get("loss", [])
        print(f"[{wrapper_key}] {len(losses_hist)} epocas em {elapsed:.0f}s | "
              f"loss final={losses_hist[-1]:.5f}")

        model_dir = output_dir / wrapper_key
        model_dir.mkdir(parents=True, exist_ok=True)
        wrapper.model.save(model_dir / "model.keras")
        meta = {
            "wrapper_key": wrapper_key,
            "name": wrapper.name,
            "persistence_gate": gate,
            "gate_mode": getattr(wrapper, "gate_mode", "static"),
            "decoder_mode": getattr(wrapper, "decoder_mode", "direct"),
            "target_mode": getattr(wrapper, "target_mode", "absolute"),
            "normalize": bool(getattr(wrapper, "normalize", True)),
            "hyperparameters": hyperparameters,
            "params_source": ("canônica 16/10 (trained_8020_itildeq) com "
                              "perda MSE pura — rodada final SBSR"),
            "feature_keep": list(keep),
            "feature_set": FEATURE_SET_LABEL[keep],
            "input_steps": 72,
            "output_steps": 36,
            "target_col": "ws100_wavelet",
            "denoise": ["ws100"],
            "denoise_level": 2,
            "loss": loss,
            "training": {"epochs_run": len(losses_hist),
                         "epochs_requested": EPOCHS, "seed": SEED,
                         "patience": 0,
                         "final_loss": (float(losses_hist[-1])
                                        if losses_hist else None),
                         "best_val_loss": None, "final_val_loss": None,
                         "training_time_sec": round(elapsed, 1)},
            "dataset": {"train_ratio": 0.80, "val_ratio": 0.0,
                        "split": "80/20 sem validacao (rodada final SBSR)"},
        }
        if static:
            meta["static_dilations"] = static
        common.write_json(model_dir / "model.json", meta)

        all_df = common.predict_all_horizons(wrapper, test_df)
        all_df["actual_raw"] = dataset["ws100"].reindex(
            all_df["timestamp"]).to_numpy()
        metrics = common.compute_metrics(all_df["actual_raw"],
                                         all_df["predicted"])
        print(f"[{wrapper_key}] MSE teste={metrics['mse']:.4f} "
              f"R2={metrics['r2']:.4f}")

    print(f"\nModelos: {output_dir}")
    print(f"Relatorio: python tests/relatorio_resultados.py --models-dir "
          f"{output_dir.relative_to(common.resolve('.'))} --out-dir "
          "data/results/report_8020_sbsr")


if __name__ == "__main__":
    main()
