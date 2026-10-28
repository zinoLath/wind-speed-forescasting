"""A/B controlado 80/20 — LSTM / LSTM-Bi: horizon-weighted MSE × ITILDE-Q.

Isola o efeito da função de perda: **tudo** é idêntico entre os dois braços
(HP canônicos de ``pipeline/tmp/trained_8020``, 60 épocas fixas, seed 42,
split 80/20 sem validação, janela 72→36, alvo ``ws100_wavelet`` sym18 nível
2); a única diferença é a loss.

Braços (4 modelos):
- ``lstm_hw`` / ``lstm_bi_hw``   → ``horizon_weighted_mse`` (canônica);
- ``lstm_itq`` / ``lstm_bi_itq`` → ``itilde_q`` (α=0.3, β=0.5, γ=0.1).

Além dos ``model.keras``/``model.json``, grava as predições long-format
(denso e rolling) para a análise/report rodarem sem GPU:

    python scripts/lidar_8020_lstm_lossab_train.py
    python tests/lstm_lossab_8020.py \
        --models-dir pipeline/tmp/trained_8020_lstm_lossab \
        --out-dir data/results/lstm_lossab_8020
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src import common

SEED = 42
CANONICAL_DIR = common.resolve("pipeline/tmp/trained_8020")
OUTPUT_DIR = common.resolve("pipeline/tmp/trained_8020_lstm_lossab")

WRAPPER_MODULES = {
    "lstm": "src.models.s2s_lstm_wrapper",
    "lstm_bi": "src.models.s2s_lstm_bi_wrapper",
}
VARIANTS = {
    "lstm_hw": ("lstm", "hweighted"),
    "lstm_bi_hw": ("lstm_bi", "hweighted"),
    "lstm_itq": ("lstm", "itilde_q"),
    "lstm_bi_itq": ("lstm_bi", "itilde_q"),
}


def apply_itildeq_coeffs(wrapper_key, alpha, beta, gamma):
    import importlib

    from src.models import losses

    module = importlib.import_module(WRAPPER_MODULES[wrapper_key])
    module.itilde_q = (
        lambda output_steps: losses.itilde_q(output_steps, alpha=alpha,
                                             beta=beta, gamma=gamma))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS),
                        choices=list(VARIANTS))
    args = parser.parse_args()

    tf = common.setup_tensorflow(True)
    print(f"TensorFlow {tf.__version__} | A/B loss LSTM 80/20")

    dataset = common.load_dataset(common.resolve("data/dataset.csv"))
    train_df, val_df, test_df = common.split_dataset(dataset, 0.80, 0.0)
    val_end = len(train_df)
    observed = common.observed_timestamps(common.resolve("data/dataset.csv"))
    print(f"Dataset {len(dataset)} | treino {len(train_df)} | teste {len(test_df)}")

    summary = {}
    for variant in args.variants:
        wrapper_key, loss_mode = VARIANTS[variant]
        meta = json.load(open(CANONICAL_DIR / wrapper_key / "model.json",
                              encoding="utf-8"))
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

        if loss_mode == "itilde_q":
            wrapper.loss_mode_override = "itilde_q"
            apply_itildeq_coeffs(wrapper_key, args.alpha, args.beta, args.gamma)
        else:
            wrapper.loss_mode_override = None
            wrapper.loss = "horizon_weighted_mse"

        wrapper.build(common.FixedHyperParameters(hyperparameters))
        expected = ("horizon_weighted_mse" if loss_mode == "hweighted"
                    else "itilde_q")
        assert wrapper.loss == expected, (variant, wrapper.loss)

        print(f"[{variant}] {wrapper_key} loss={wrapper.loss} "
              f"treino {args.epochs} epocas...")
        started = time.perf_counter()
        history = wrapper.fit(epochs=args.epochs, batch_size=batch_size,
                              verbose=0, callbacks=None, use_validation=False)
        elapsed = time.perf_counter() - started
        losses = history.history.get("loss", [])

        model_dir = OUTPUT_DIR / variant
        model_dir.mkdir(parents=True, exist_ok=True)
        wrapper.model.save(model_dir / "model.keras")

        # predições densas (mesma grade do relatório de resultados)
        eval_df = dataset.iloc[max(0, val_end - wrapper.input_steps):]
        all_df = common.predict_all_horizons(wrapper, eval_df)
        all_df["actual_raw"] = dataset["ws100"].reindex(
            all_df["timestamp"]).to_numpy()
        if observed is not None:
            all_df = all_df[all_df["timestamp"].isin(observed)]
        all_df = all_df.assign(actual=all_df["actual_raw"])
        all_df.to_csv(model_dir / "predictions_all.csv", index=False)

        # rolling autônomo (h=36)
        roll_pred, roll_actual = wrapper.rolling_forecast(
            dataset, test_start=val_end)
        offset = val_end + wrapper.output_steps - 1
        roll_df = pd.DataFrame({
            "origin": dataset.index[val_end - 1: val_end - 1 + len(roll_actual)],
            "timestamp": dataset.index[offset: offset + len(roll_actual)],
            "actual_denoised": roll_actual.ravel(),
            "predicted": roll_pred.ravel(),
        })
        roll_df["actual_raw"] = dataset["ws100"].reindex(
            roll_df["timestamp"]).to_numpy()
        if observed is not None:
            roll_df = roll_df[roll_df["timestamp"].isin(observed)]
        roll_df.to_csv(model_dir / "predictions_rolling.csv", index=False)

        m_all = common.compute_metrics(all_df["actual_raw"], all_df["predicted"])
        m_roll = common.compute_metrics(roll_df["actual_raw"],
                                        roll_df["predicted"])
        summary[variant] = {
            "wrapper_key": wrapper_key, "loss": wrapper.loss,
            "hp": hyperparameters, "params": int(wrapper.model.count_params()),
            "epochs": len(losses), "training_time_sec": round(elapsed, 1),
            "final_loss": float(losses[-1]) if losses else None,
            "all": m_all, "rolling": m_roll,
        }
        common.write_json(model_dir / "model.json", {
            "wrapper_key": wrapper_key,
            "name": wrapper.name,
            "variant": variant,
            "label": f"{wrapper_key.upper().replace('_', '-')} — {wrapper.loss}",
            "persistence_gate": False, "gate_mode": "static",
            "decoder_mode": getattr(wrapper, "decoder_mode", "direct"),
            "target_mode": getattr(wrapper, "target_mode", "absolute"),
            "normalize": bool(getattr(wrapper, "normalize", True)),
            "hyperparameters": hyperparameters,
            "params_source": "canonica 80/20 (mesma p/ os dois braços)",
            "input_steps": 72, "output_steps": 36,
            "target_col": "ws100_wavelet", "denoise": ["ws100"],
            "denoise_level": 2, "loss": wrapper.loss,
            "loss_coeffs": ({"alpha": args.alpha, "beta": args.beta,
                             "gamma": args.gamma}
                            if loss_mode == "itilde_q" else None),
            "training": {"epochs_run": len(losses),
                         "epochs_requested": args.epochs, "seed": args.seed,
                         "patience": 0,
                         "final_loss": float(losses[-1]) if losses else None,
                         "best_val_loss": None, "final_val_loss": None,
                         "training_time_sec": round(elapsed, 1)},
            "dataset": {"train_ratio": 0.80, "val_ratio": 0.0,
                        "split": "80/20 sem validacao (A/B de loss)"},
        })
        print(f"[{variant}] {elapsed:.0f}s | MSE denso={m_all['mse']:.4f} "
              f"rolling={m_roll['mse']:.4f}")

    common.write_json(OUTPUT_DIR / "summary.json", summary)
    print(f"\nModelos: {OUTPUT_DIR}")
    print("Analise: python tests/lstm_lossab_8020.py --models-dir "
          "pipeline/tmp/trained_8020_lstm_lossab "
          "--out-dir data/results/lstm_lossab_8020")


if __name__ == "__main__":
    main()
