"""Estudo Optuna (TPU) — HPs da LSTM-Bi com perda MSE.

Protocolo escolhido (exploratório, 1B): split 80/20 canônico **sem
validação**; objetivo = MSE no TESTE (vazamento assumido — NÃO reportável
no artigo; revalidar em validação/seeds antes). Espaço = round 5
(lr/weight_decay/units/dropouts), 60 épocas, seed 42.
Pruning: MedianPruner com MSE de teste em subamostra de origens a cada
10 épocas.

Roda no kernel TPU (scripts/dist/kaggle_tpu_push.py); fallback CPU/GPU se a
TPU não estiver disponível.
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

SEED = 42
EPOCHS = 60
OUT_STEPS = 36
BATCH = 64
SUBSAMPLE = 200  # origens p/ avaliação intermediária (pruning)
CHECKPOINTS = (10, 20, 30, 40, 50)


def test_eval(wrapper, dataset, train_len, subsample=None):
    eval_df = dataset.iloc[max(0, train_len - wrapper.input_steps):]
    df = common.predict_all_horizons(wrapper, eval_df)
    df["actual"] = dataset["ws100"].reindex(df["timestamp"]).to_numpy()
    df = df[df["actual"].notna()]
    if subsample:
        keep = sorted(df["origin"].unique())[::max(1, len(
            df["origin"].unique()) // subsample)]
        df = df[df["origin"].isin(keep)]
    return float(np.mean((df["predicted"] - df["actual"]) ** 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--timeout", type=int, default=None)
    args = ap.parse_args()

    import tensorflow as tf

    strategy = tf.distribute.get_strategy()
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu="local")
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        print(f"TPU OK | réplicas={strategy.num_replicas_in_sync}")
    except Exception as error:  # noqa: BLE001
        print(f"TPU indisponível ({type(error).__name__}: {error}) — "
              f"fallback {strategy.num_replicas_in_sync}x")

    dataset = common.load_dataset(common.resolve("data/dataset.csv"))
    train_df, _, _ = common.split_dataset(dataset, 0.80, 0.0)
    out_dir = PROJECT_ROOT / "pipeline" / "tmp" / "itildeq_tpu"
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{out_dir / 'round6_lstm_bi.db'}"
    study = optuna.create_study(
        study_name="round6_lstm_bi", storage=storage, direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8,
                                           n_warmup_steps=5))
    print(f"TF {tf.__version__} | estudo {study.study_name} | "
          f"{len(study.trials)} trials existentes | objetivo=test MSE "
          f"(80/20, exploratório)")

    from keras import backend as K

    class _Stop(Exception):
        pass

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-4, 2e-2, log=True)
        wd = trial.suggest_categorical("weight_decay",
                                       [0.0, 1e-5, 1e-4, 3e-4, 1e-3])
        units = trial.suggest_int("lstm_units", 48, 252, step=4)
        enc_do = trial.suggest_float("encoder_dropout_rate", 0.0, 0.4)
        dec_do = trial.suggest_float("decoder_dropout_rate", 0.0, 0.4)
        hp = {"learning_rate": lr, "weight_decay": wd, "lstm_units": units,
              "encoder_dropout_rate": enc_do,
              "decoder_dropout_rate": dec_do, "batch_size": BATCH}
        K.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        with strategy.scope():
            wrapper = common.wrapper_factory("lstm_bi")()
            wrapper.prepare(train_df, train_df.tail(400), input_steps=72,
                            output_steps=OUT_STEPS,
                            target_col="ws100_wavelet",
                            denoise=("ws100",), denoise_level=2)
            steps = int((len(train_df) + BATCH - 1) // BATCH)
            if hasattr(wrapper, "schedule_total_steps"):
                wrapper.schedule_total_steps = steps * EPOCHS
            wrapper.loss_mode_override = "mse"
            wrapper.build(common.FixedHyperParameters(hp))

        class _Cb(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                ep = epoch + 1
                if ep in CHECKPOINTS:
                    value = test_eval(wrapper, dataset, len(train_df),
                                      subsample=SUBSAMPLE)
                    trial.report(value, ep)
                    print(f"  [{trial.number}] ep {ep} val(test-sub)="
                          f"{value:.4f}")
                    if trial.should_prune():
                        raise optuna.TrialPruned()

        t0 = time.perf_counter()
        wrapper.fit(epochs=EPOCHS, batch_size=BATCH, verbose=0,
                    callbacks=[_Cb()], use_validation=False)
        value = test_eval(wrapper, dataset, len(train_df))
        print(f"  trial {trial.number}: lr={lr:.4g} wd={wd:g} u={units} "
              f"do={enc_do:.2f}/{dec_do:.2f} -> test_mse={value:.4f} "
              f"({time.perf_counter() - t0:.0f}s)")
        return value

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    done = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE]
    rows = sorted([{"trial": t.number, **t.params, "test_mse": t.value,
                    "elapsed_sec": t.user_attrs.get("elapsed_sec")}
                   for t in done], key=lambda r: r["test_mse"])
    best = rows[0] if rows else None
    (out_dir / "trials.json").write_text(json.dumps(rows, indent=1),
                                         encoding="utf-8")
    if best:
        # best_trial.json no formato do coletor (pipeline/tmp/optuna)
        (out_dir / "best_trial.json").write_text(json.dumps({
            "best_params": {k: best[k] for k in
                            ("learning_rate", "weight_decay", "lstm_units",
                             "encoder_dropout_rate",
                             "decoder_dropout_rate")},
            "best_value": -best["test_mse"],
            "n_trials": study.trials and len(study.trials),
            "objective": "test_mse (80/20, exploratorio)",
            "source": "round6_lstm_bi (TPU)",
        }, indent=1), encoding="utf-8")
        print(f"BEST: {best}")
    print(f"salvo em {out_dir}")


if __name__ == "__main__":
    main()
