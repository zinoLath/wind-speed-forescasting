"""Busca curta dos coeficientes da perda ITILDE-Q (α, β, γ) — roda NO
KERNEL do Kaggle via pacote colab_round2 (usado por
``scripts/experiments/itildeq_terms_kaggle.py``).

Para cada combo ``alpha:beta:gamma`` treina uma LSTM com os HPs canônicos
da rodada ``8020-itildeq`` (embutidos abaixo; 80/20 sem validação, 60
épocas, seed 42) e avalia all + h=36 + rolling contra o ``ws100`` bruto.
Salva um JSON por combo em ``--output-dir``.

    python scripts/experiments/itildeq_terms_search_train.py \
        --combos "0.3:0.5:0.1;0.1:0.5:0.1" \
        --output-dir pipeline/tmp/itildeq_terms_search
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common  # noqa: E402

SEED = 42
EPOCHS = 60
OUTPUT_STEPS = 36

# HPs canônicos por wrapper (trained_8020_itildeq, restaurado 21/10/2026)
CANONICAL_HP = {
    "lstm": {
        "batch_size": 64,
        "decoder_dropout_rate": 0.246951,
        "encoder_dropout_rate": 0.274739,
        "learning_rate": 0.011097,
        "lstm_units": 192,
        "weight_decay": 0.000189,
    },
    "lstm_bi": {
        "batch_size": 32,
        "decoder_dropout_rate": 0.314776,
        "encoder_dropout_rate": 0.082704,
        "learning_rate": 0.003396,
        "lstm_units": 256,
        "weight_decay": 0.000102,
    },
}


def wrapper_hp(wrapper_key):
    """HPs canônicos do wrapper (do model.json local, com fallback embutido)."""
    path = (PROJECT_ROOT / "pipeline" / "tmp" / "trained_8020_itildeq"
            / wrapper_key / "model.json")
    if path.is_file():
        return dict(json.load(open(path, encoding="utf-8"))["hyperparameters"])
    return dict(CANONICAL_HP[wrapper_key])


def parse_combos(raw):
    combos = []
    for chunk in raw.split(";"):
        a, b, g = (float(x) for x in chunk.strip().split(":"))
        combos.append((a, b, g))
    return combos


WRAPPER_MODULES = {"lstm": "src.models.s2s_lstm_wrapper",
                   "lstm_bi": "src.models.s2s_lstm_bi_wrapper"}


def patch_itildeq(wrapper_key, alpha, beta, gamma):
    """Substitui o binding ``itilde_q`` NO MÓDULO DO WRAPPER (o patch em
    ``losses`` não afeta o ``from .losses import itilde_q`` já importado)."""
    import importlib
    from src.models import losses

    lw = importlib.import_module(WRAPPER_MODULES[wrapper_key])
    lw.itilde_q = (lambda output_steps:
                   losses.itilde_q(output_steps, alpha=alpha,
                                   beta=beta, gamma=gamma))


def evaluate(wrapper, dataset, val_end):
    eval_df = dataset.iloc[max(0, val_end - wrapper.input_steps):]
    all_df = common.predict_all_horizons(wrapper, eval_df)
    all_df["actual"] = dataset["ws100"].reindex(all_df["timestamp"]).to_numpy()
    per_h = common.horizon_metrics(all_df)
    h36 = per_h[per_h["horizon"] == OUTPUT_STEPS].iloc[0].to_dict()
    a = all_df["actual"].to_numpy()
    p = all_df["predicted"].to_numpy()
    mse = float(np.mean((a - p) ** 2))
    metrics = {"mae": float(np.mean(np.abs(a - p))), "mse": mse,
               "rmse": float(np.sqrt(mse)),
               "r2": float(1.0 - mse / np.var(a))}
    return {"all": metrics,
            "h36": {k: float(h36[k]) for k in ("r2", "mse", "mae", "rmse")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--combos", required=True,
                    help="lista 'a:b:g;a:b:g;...'")
    ap.add_argument("--output-dir", default="pipeline/tmp/itildeq_terms_search")
    ap.add_argument("--wrapper", choices=("lstm", "lstm_bi"), default="lstm")
    ap.add_argument("--save-models", action="store_true",
                    help="salva model.keras + model.json junto do JSON")
    args = ap.parse_args()

    tf = common.setup_tensorflow(True)
    print(f"TensorFlow {tf.__version__} | busca de termos ITILDE-Q "
          f"({len(parse_combos(args.combos))} combos)")

    dataset = common.load_dataset(common.resolve("data/dataset.csv"))
    train_df, val_df, test_df = common.split_dataset(dataset, 0.80, 0.0)
    val_end = len(train_df) + len(val_df)
    out_dir = common.resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from keras import backend as K

    for alpha, beta, gamma in parse_combos(args.combos):
        tag = f"{args.wrapper}_{alpha}_{beta}_{gamma}".replace(".", "")
        json_path = out_dir / f"combo_{tag}.json"
        if json_path.is_file():
            print(f"[{tag}] já existe — pulando")
            continue
        K.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        patch_itildeq(args.wrapper, alpha, beta, gamma)

        wrapper = common.wrapper_factory(args.wrapper)()
        wrapper.prepare(train_df, train_df.tail(400), input_steps=72,
                        output_steps=OUTPUT_STEPS,
                        target_col="ws100_wavelet",
                        denoise=("ws100",), denoise_level=2)
        hp = wrapper_hp(args.wrapper)
        batch_size = int(hp.get("batch_size", 32))
        if hasattr(wrapper, "schedule_total_steps"):
            steps = int((len(train_df) + batch_size - 1) // batch_size)
            wrapper.schedule_total_steps = steps * EPOCHS
        wrapper.loss_mode_override = "itilde_q"
        wrapper.build(common.FixedHyperParameters(hp))
        assert wrapper.loss != "mse"

        started = time.perf_counter()
        wrapper.fit(epochs=EPOCHS, batch_size=batch_size, verbose=0,
                    callbacks=None, use_validation=False)
        elapsed = time.perf_counter() - started

        if args.save_models:
            mdir = out_dir / f"model_{args.wrapper}_{tag}"
            mdir.mkdir(parents=True, exist_ok=True)
            wrapper.model.save(mdir / "model.keras")
            meta = {
                "wrapper_key": args.wrapper,
                "name": wrapper.name,
                "persistence_gate": False,
                "gate_mode": "static",
                "decoder_mode": getattr(wrapper, "decoder_mode", "direct"),
                "target_mode": getattr(wrapper, "target_mode", "absolute"),
                "normalize": bool(getattr(wrapper, "normalize", True)),
                "hyperparameters": dict(hp),
                "params_source": "busca de coeficientes itilde-q",
                "input_steps": 72,
                "output_steps": OUTPUT_STEPS,
                "target_col": "ws100_wavelet",
                "denoise": ["ws100"],
                "denoise_level": 2,
                "loss": "itilde_q",
                "loss_coeffs": {"alpha": alpha, "beta": beta,
                                "gamma": gamma},
                "training": {"epochs_run": EPOCHS,
                             "epochs_requested": EPOCHS, "seed": SEED,
                             "patience": 0, "final_loss": None,
                             "best_val_loss": None,
                             "final_val_loss": None,
                             "training_time_sec": None},
                "dataset": {"train_ratio": 0.80, "val_ratio": 0.0,
                            "split": "80/20 sem validacao (busca termos)"},
            }
            common.write_json(mdir / "model.json", meta)
        result = {"wrapper": args.wrapper, "alpha": alpha,
                  "beta": beta, "gamma": gamma,
                  "epochs": EPOCHS, "seed": SEED, "elapsed_sec": elapsed,
                  "hp": dict(hp), **evaluate(wrapper, dataset, val_end)}
        common.write_json(json_path, result)
        print(f"[a={alpha} b={beta} g={gamma}] {elapsed:.0f}s | "
              f"all: r2={result['all']['r2']:.4f} "
              f"mse={result['all']['mse']:.4f} | "
              f"h36 r2={result['h36']['r2']:.4f}")


if __name__ == "__main__":
    main()
