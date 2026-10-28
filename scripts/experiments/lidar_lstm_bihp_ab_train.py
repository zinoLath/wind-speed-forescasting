"""A/B de loss na config bihp (80/20) — horizon-weighted MSE × ITILDE-Q.

bihp = LSTM unidirecional com os hp exatos da lstm_bi canônica 80/20
(``pipeline/tmp/trained_8020/lstm_bi/model.json``). Este script roda os dois
braços NA MESMA SESSÃO (mesma seed, mesma init de dados; única diferença =
loss), eliminando a variância entre sessões que confunde comparações com
runs antigos:

- ``bihp_hw``  → ``horizon_weighted_mse`` (canônica);
- ``bihp_itq`` → ``itilde_q`` (α=0.3, β=0.5, γ=0.1).

Saídas:
    pipeline/tmp/trained_8020_lstm_bihp_ab/<braço>/  (model.keras + preds)
    data/results/lstm_bihp_ab_8020/summary.json

Análise (sem GPU, DM pareado + figuras):
    python studies/lstm_bihp_ab_8020.py

    source scripts/tf_gpu_env.sh
    python scripts/experiments/lidar_lstm_bihp_ab_train.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common  # noqa: E402

SEED = 42
BI_HP = common.resolve("pipeline/tmp/trained_8020/lstm_bi/model.json")
OUTPUT_DIR = common.resolve("pipeline/tmp/trained_8020_lstm_bihp_ab")
RESULTS_DIR = common.resolve("data/results/lstm_bihp_ab_8020")

WRAPPER_MODULE = "src.models.s2s_lstm_wrapper"
VARIANTS = {
    "bihp_hw": "hweighted",
    "bihp_itq": "itilde_q",
}


def apply_itildeq_coeffs(alpha, beta, gamma):
    import importlib

    from src.models import losses

    module = importlib.import_module(WRAPPER_MODULE)
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
    print(f"TensorFlow {tf.version} | A/B loss na config bihp (80/20)")

    dataset = common.load_dataset(common.resolve("data/dataset.csv"))
    train_df, val_df, test_df = common.split_dataset(dataset, 0.80, 0.0)
    val_end = len(train_df)
    observed = common.observed_timestamps(common.resolve("data/dataset.csv"))
    print(f"Dataset {len(dataset)} | treino {len(train_df)} | "
          f"teste {len(test_df)}")

    hyperparameters = dict(json.load(open(BI_HP, encoding="utf-8"))
                           ["hyperparameters"])
    print(f"hp da lstm_bi canônica: {hyperparameters}")

    summary = {}
    for variant in args.variants:
        loss_mode = VARIANTS[variant]

        from keras import backend as K

        K.clear_session()
        tf.keras.utils.set_random_seed(args.seed)
        wrapper = common.wrapper_factory("lstm")()
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
            apply_itildeq_coeffs(args.alpha, args.beta, args.gamma)
        else:
            wrapper.loss_mode_override = None
            wrapper.loss = "horizon_weighted_mse"

        wrapper.build(common.FixedHyperParameters(hyperparameters))
        expected = ("horizon_weighted_mse" if loss_mode == "hweighted"
                    else "itilde_q")
        assert wrapper.loss == expected, (variant, wrapper.loss)

        print(f"[{variant}] loss={wrapper.loss} "
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
            "wrapper_key": "lstm", "loss": wrapper.loss,
            "hp": hyperparameters, "params": int(wrapper.model.count_params()),
            "epochs": len(losses), "training_time_sec": round(elapsed, 1),
            "final_loss": float(losses[-1]) if losses else None,
            "all": m_all, "rolling": m_roll,
        }
        common.write_json(model_dir / "model.json", {
            "wrapper_key": "lstm",
            "name": wrapper.name,
            "variant": variant,
            "label": f"bihp — {wrapper.loss}",
            "persistence_gate": False, "gate_mode": "static",
            "decoder_mode": getattr(wrapper, "decoder_mode", "direct"),
            "target_mode": getattr(wrapper, "target_mode", "absolute"),
            "normalize": bool(getattr(wrapper, "normalize", True)),
            "hyperparameters": hyperparameters,
            "params_source": "lstm_bi canonica 80/20 (mesma p/ os dois braços)",
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
                        "split": "80/20 sem validacao (A/B de loss bihp)"},
        })
        print(f"[{variant}] {elapsed:.0f}s | MSE denso={m_all['mse']:.4f} "
              f"rolling={m_roll['mse']:.4f}")

    common.write_json(RESULTS_DIR / "summary.json", summary)
    print(f"\nModelos: {OUTPUT_DIR}")
    print(f"Resumo: {RESULTS_DIR / 'summary.json'}")
    print("Analise: python studies/lstm_bihp_ab_8020.py")


if __name__ == "__main__":
    main()
