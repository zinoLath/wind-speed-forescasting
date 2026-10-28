"""Revalidação do lineup SBSR: 72,5/12,5/15 + EarlyStopping + RLRP.

Mesmos modelos do artigo (finding `rodada-final-sbsr`: TCN/TCN-Bi 48 col
com meteo, LSTM/LSTM-Bi featurex 67 col, MSE pura, HPs canônicos com
dilatações C′, seed 42, 60 épocas) — mas com split 72,5/12,5/15 COM
validação real e EarlyStopping(val_loss, patience 12, restore_best) +
ReduceLROnPlateau, o regime que o HPO usa e que o protocolo final seco
não tem. Diagnóstico: quanto do gap visto no sweep é regime de treino?

    python scripts/lidar_reval_sbsr_train.py
    python reports/src/relatorio_resultados.py \
        --models-dir pipeline/tmp/trained_reval_sbsr \
        --out-dir data/results/report_reval_sbsr \
        --trim-boundary-origins 72 \
        --run-title "LiDAR 72,5/12,5/15 — revalidação SBSR"
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
PATIENCE = 12
TRAIN_RATIO = 0.725
VAL_RATIO = 0.125
OUTPUT_DIR = common.resolve("pipeline/tmp/trained_reval_sbsr")
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
    KEEP_METEO: "perfil 46 + humid/temp (48 col)",
    KEEP_FEATUREX: "perfil 46 + cis1..19 + humid/temp (67 col)",
}
SPLIT_LABEL = ("72,5/12,5/15 com EarlyStopping(patience 12, restore_best) "
               "+ ReduceLROnPlateau")


def canonical_meta(wrapper_key):
    # snapshot versionado (pipeline/configs/) com fallback ao artefato treinado
    return common.canonical_round_meta(wrapper_key)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = common.resolve(args.output_dir)
    tf = common.setup_tensorflow(True)
    print(f"TensorFlow {tf.__version__} | revalidação SBSR "
          f"({SPLIT_LABEL}, HPs da canônica {CANON_DIR.name}, MSE pura, "
          f"seed {SEED})")

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
        train_df, val_df, test_df = common.split_dataset(
            dataset, TRAIN_RATIO, VAL_RATIO)
        wrapper.prepare(
            train_df,
            val_df,
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

        wrapper.loss_mode_override = loss
        if static:
            wrapper.static_dilations = static

        wrapper.build(common.FixedHyperParameters(hyperparameters))
        assert wrapper.loss == loss, (wrapper_key, wrapper.loss)
        callbacks = common.default_callbacks(wrapper, args.patience)
        print(f"[{wrapper_key}] keep={len(keep)} extras loss={loss} "
              f"static={bool(static)} "
              f"entradas={wrapper.num_encoder_features} | callbacks: "
              f"{[type(c).__name__ for c in callbacks]} | batch={batch_size}")

        print(f"[{wrapper_key}] treino {args.epochs} épocas (ES patience "
              f"{args.patience})...")
        started = time.perf_counter()
        history = wrapper.fit(epochs=args.epochs, batch_size=batch_size,
                              verbose=0, callbacks=callbacks,
                              use_validation=True)
        elapsed = time.perf_counter() - started
        losses_hist = history.history.get("loss", [])
        val_losses = history.history.get("val_loss", [])
        print(f"[{wrapper_key}] {len(losses_hist)}/{args.epochs} épocas em "
              f"{elapsed:.0f}s | loss final={losses_hist[-1]:.5f} | "
              f"melhor val_loss={min(val_losses):.6f}"
              if val_losses else "")

        model_dir = output_dir / wrapper_key
        model_dir.mkdir(parents=True, exist_ok=True)
        wrapper.model.save(model_dir / "model.keras")
        meta = {
            "wrapper_key": wrapper_key,
            "name": wrapper.name,
            "persistence_gate": False,
            "gate_mode": getattr(wrapper, "gate_mode", "static"),
            "decoder_mode": getattr(wrapper, "decoder_mode", "direct"),
            "target_mode": getattr(wrapper, "target_mode", "absolute"),
            "normalize": bool(getattr(wrapper, "normalize", True)),
            "hyperparameters": hyperparameters,
            "params_source": ("canônica 16/10 (trained_8020_itildeq) com "
                              "perda MSE pura — revalidação 72,5/12,5/15 "
                              "+ ES/RLRP"),
            "feature_keep": list(keep),
            "feature_set": FEATURE_SET_LABEL[keep],
            "input_steps": 72,
            "output_steps": 36,
            "target_col": "ws100_wavelet",
            "denoise": ["ws100"],
            "denoise_level": 2,
            "loss": loss,
            "training": {"epochs_run": len(losses_hist),
                         "epochs_requested": args.epochs, "seed": SEED,
                         "patience": args.patience,
                         "early_stopping": True,
                         "reduce_lr_on_plateau": True,
                         "final_loss": (float(losses_hist[-1])
                                        if losses_hist else None),
                         "final_val_loss": (float(val_losses[-1])
                                            if val_losses else None),
                         "best_val_loss": (float(min(val_losses))
                                           if val_losses else None),
                         "training_time_sec": round(elapsed, 1)},
            "dataset": {"train_ratio": TRAIN_RATIO,
                        "val_ratio": VAL_RATIO,
                        "split": SPLIT_LABEL},
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
    print("Relatorio: python reports/src/relatorio_resultados.py --models-dir "
          f"{output_dir.relative_to(common.resolve('.'))} --out-dir "
          "data/results/report_reval_sbsr --trim-boundary-origins 72")


if __name__ == "__main__":
    main()
