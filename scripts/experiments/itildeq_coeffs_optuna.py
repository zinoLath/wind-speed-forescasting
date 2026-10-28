"""Busca Optuna dos coeficientes da ITILDE-Q (α, β, γ) sob o protocolo da
Optuna Round 5.

Protocolo: split 72/8/20 (teste de 20% intocado), HPs promovidos da round
5 (pipeline/tmp/optuna/<Seq2Seq_*>/best_trial.json), 60 épocas, seed 42.
Objetivo por trial = **MSE em 4 blocos de horizonte na validação** (mesma
família do objetivo da round 5). Ao final: melhor trial é reavaliado no
teste + painel de diagnóstico (DTW etc.).

    python scripts/experiments/itildeq_coeffs_optuna.py --wrapper lstm --n-trials 16
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common  # noqa: E402
from src.models import losses  # noqa: E402
import src.models.s2s_lstm_wrapper as lw_lstm  # noqa: E402
import src.models.s2s_lstm_bi_wrapper as lw_lstm_bi  # noqa: E402
from scripts.prediction_diagnostics import panel  # noqa: E402

SEED = 42
EPOCHS = 60
OUT_STEPS = 36
K_BLOCKS = 4
WRAPPERS = {"lstm": ("Seq2Seq_LSTM", lw_lstm),
            "lstm_bi": ("Seq2Seq_LSTM_Bidirectional", lw_lstm_bi)}
STORE = PROJECT_ROOT / "pipeline" / "tmp" / "optuna_itildeq_coeffs"
RESULTS = PROJECT_ROOT / "data" / "results" / "itildeq_coeffs_optuna"


def patch(wrapper_key, alpha, beta, gamma):
    """Patch do binding itilde_q no módulo do wrapper."""
    lw = WRAPPERS[wrapper_key][1]
    lw.itilde_q = (lambda steps: losses.itilde_q(
        steps, alpha=alpha, beta=beta, gamma=gamma))


def r5_hp(wrapper_key):
    name = WRAPPERS[wrapper_key][0]
    path = PROJECT_ROOT / "pipeline" / "tmp" / "optuna" / name \
        / "best_trial.json"
    return dict(json.loads(path.read_text("utf-8"))["best_params"])


def block_val_mse(wrapper, train_df, val_df):
    """MSE da validação em K_BLOCKS blocos de horizonte (como na round 5)."""
    eval_df = val_df.copy()
    ctx = train_df.tail(wrapper.input_steps)
    import pandas as pd
    eval_df = pd.concat([ctx, eval_df])
    df = common.predict_all_horizons(wrapper, eval_df)
    df = df[df["origin"].isin(val_df.index)]
    raw = val_df["ws100"]  # actual bruto da validação
    df["actual"] = raw.reindex(df["timestamp"]).to_numpy()
    df = df[df["actual"].notna()]
    per_h = common.horizon_metrics(df)
    blocks = np.array_split(per_h.sort_values("horizon")["mse"].to_numpy(),
                            K_BLOCKS)
    return float(np.mean([float(b.mean()) for b in blocks]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wrapper", choices=("lstm", "lstm_bi"), default="lstm")
    ap.add_argument("--n-trials", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=None,
                    help="timeout em segundos (opcional)")
    args = ap.parse_args()

    tf = common.setup_tensorflow(True)
    dataset = common.load_dataset(common.resolve("data/dataset.csv"))
    train_df, val_df, test_df = common.split_dataset(dataset, 0.72, 0.08)
    hp = r5_hp(args.wrapper)
    OUT = PROJECT_ROOT / "pipeline" / "tmp" / "optuna_itildeq_coeffs"
    OUT.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{OUT / (args.wrapper + '.db')}"
    study = optuna.create_study(
        study_name=f"itildeq_coeffs_{args.wrapper}", storage=storage,
        direction="minimize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED))
    print(f"TensorFlow {tf.__version__} | estudo {study.study_name} | "
          f"{len(study.trials)} trials existentes | split 72/8/20 | "
          f"objetivo val_block_mse(k={K_BLOCKS})")

    from keras import backend as K

    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-3, 0.5, log=True)
        beta = trial.suggest_float("beta", 0.2, 0.8)
        gamma = trial.suggest_float("gamma", 1e-3, 0.5, log=True)
        K.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        patch(args.wrapper, alpha, beta, gamma)
        wrapper = common.wrapper_factory(args.wrapper)()
        wrapper.prepare(train_df, val_df, input_steps=72,
                        output_steps=OUT_STEPS,
                        target_col="ws100_wavelet",
                        denoise=("ws100",), denoise_level=2)
        batch_size = int(hp.get("batch_size", 32))
        if hasattr(wrapper, "schedule_total_steps"):
            steps = int((len(train_df) + batch_size - 1) // batch_size)
            wrapper.schedule_total_steps = steps * EPOCHS
        wrapper.loss_mode_override = "itilde_q"
        wrapper.build(common.FixedHyperParameters(dict(hp)))
        t0 = time.perf_counter()
        wrapper.fit(epochs=EPOCHS, batch_size=batch_size, verbose=0,
                    callbacks=None, use_validation=False)
        value = block_val_mse(wrapper, train_df, val_df)
        trial.set_user_attr("elapsed_sec", time.perf_counter() - t0)
        print(f"  trial {trial.number}: a={alpha:.4f} b={beta:.3f} "
              f"g={gamma:.4f} -> val_block_mse={value:.4f} "
              f"({time.perf_counter() - t0:.0f}s)")
        return value

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    trials = sorted([t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE],
                    key=lambda t: t.value)
    best = trials[0]
    alpha, beta, gamma = (best.params["alpha"], best.params["beta"],
                          best.params["gamma"])

    # reavaliação do melhor no TESTE + painel de comportamento
    K.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    patch(args.wrapper, alpha, beta, gamma)
    wrapper = common.wrapper_factory(args.wrapper)()
    wrapper.prepare(train_df, val_df, input_steps=72, output_steps=OUT_STEPS,
                    target_col="ws100_wavelet", denoise=("ws100",),
                    denoise_level=2)
    batch_size = int(hp.get("batch_size", 32))
    wrapper.loss_mode_override = "itilde_q"
    wrapper.build(common.FixedHyperParameters(dict(hp)))
    wrapper.fit(epochs=EPOCHS, batch_size=batch_size, verbose=0,
                callbacks=None, use_validation=False)
    ctx = dataset.iloc[max(0, len(train_df) + len(val_df)
                          - wrapper.input_steps):]
    df = common.predict_all_horizons(wrapper, ctx)
    cutoff = dataset.index[len(train_df) + len(val_df)]
    df = df[df["origin"] >= cutoff]
    df["actual"] = dataset["ws100"].reindex(df["timestamp"]).to_numpy()
    df = df[df["actual"].notna()]
    per_h = common.horizon_metrics(df)
    h36 = per_h[per_h["horizon"] == 36].iloc[0]
    test_panel = dict(panel(df[df["actual"].notna()], dataset))
    test_panel["h36"] = {k: float(h36[k]) for k in ("r2", "mse")}

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {
        "wrapper": args.wrapper,
        "protocol": "72/8/20, 60 epocas, seed 42, HPs round5",
        "objective": f"val_block_mse(k={K_BLOCKS})",
        "n_trials": len(trials),
        "best": {"alpha": alpha, "beta": beta, "gamma": gamma,
                 "val_block_mse": best.value},
        "test": test_panel,
        "trials": [{"alpha": t.params["alpha"], "beta": t.params["beta"],
                    "gamma": t.params["gamma"], "value": t.value,
                    "elapsed_sec": t.user_attrs.get("elapsed_sec")}
                   for t in trials],
    }
    (RESULTS / f"{args.wrapper}.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")

    print(f"\nmelhor: a={alpha:.4f} b={beta:.3f} g={gamma:.4f} | "
          f"val_block_mse={best.value:.4f}")
    print(f"teste: R²={test_panel['r2']:.4f} MSE={test_panel['mse']:.4f} | "
          f"DTW={test_panel['dtw_mean']:.3f} corr={test_panel['corr_mean']:.3f}")
    print(f"salvo em {RESULTS / (args.wrapper + '.json')}")


if __name__ == "__main__":
    main()
