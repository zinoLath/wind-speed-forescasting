"""Optuna: LSTM na metodologia 80/20 atual, partida dos hp da lstm_bi.

Design:
- split interno 72/8/20 — o teste de 20% permanece IDENTICO ao canonico 80/20
  e nunca entra na otimizacao (objetivo = val_loss, convencao de deployment);
- primeira tentativa enfileirada = hp exatos da lstm_bi canonica (ponto bom
  conhecido); espaco log/categoria centrado neles; TPE + MedianPruner;
- estudo persistido em sqlite (resume com load_if_exists);
- ao fim: re-treino com os melhores hp no treino completo (80%) + avaliacao
  no teste intocado vs lstm canonica, lstm_bi, ARIMA e persistencia.

    source .venv/bin/activate && source scripts/tf_gpu_env.sh
    python scripts/experiments/lidar_lstm_hpo.py --trials 40
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

SEED = 42
OUTPUT_DIR = common.resolve("pipeline/tmp/trained_8020_lstm_hpo")
RESULTS_DIR = common.resolve("data/results/lstm_hpo_8020")
STUDY_DB = common.resolve("pipeline/tmp/optuna_8020_lstm/optuna.db")
BI_HP = common.resolve("pipeline/tmp/trained_8020/lstm_bi/model.json")
CANONICAL = common.resolve("data/results/lidar_8020_features/all.json")
BASELINES = common.resolve("data/results/baselines/arima_rf/metrics.json")
NTFY = "https://ntfy.sh/zino-imputeoptuna"


def notify(message):
    try:
        import urllib.request

        urllib.request.urlopen(
            urllib.request.Request(NTFY, data=message.encode()), timeout=10
        )
    except Exception:
        pass


def build_space(trial, bi_params):
    """Espaco centrado no ponto bom conhecido (lstm_bi)."""
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", 5e-4, 2e-2, log=True),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-6, 1e-3, log=True),
        "lstm_units": trial.suggest_categorical(
            "lstm_units", (96, 128, 160, 192, 256)),
        "encoder_dropout_rate": trial.suggest_float(
            "encoder_dropout_rate", 0.05, 0.45),
        "decoder_dropout_rate": trial.suggest_float(
            "decoder_dropout_rate", 0.05, 0.40),
        "batch_size": 32,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=None,
                        help="segundos totais (opcional)")
    args = parser.parse_args()

    import optuna
    import tensorflow as tf
    from keras import callbacks as kcb

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bi_params_full = json.load(open(BI_HP))["hyperparameters"]

    tf_gpu = common.setup_tensorflow(True)
    print(f"TensorFlow {tf_gpu.__version__} | "
          f"GPUs: {tf_gpu.config.list_physical_devices('GPU')}")

    dataset = common.load_dataset(common.resolve("data/dataset.csv"))
    train_full, _, test_df = common.split_dataset(dataset, 0.80, 0.0)
    train_df, val_df, _ = common.split_dataset(train_full, 0.90, 0.10)
    val_end_full = len(train_full)
    print(f"Dataset {len(dataset)} | hpo-train {len(train_df)} | "
          f"hpo-val {len(val_df)} | teste intocado {len(test_df)}")

    STUDY_DB.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name="lstm_8020_metodologia_atual",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6,
                                           n_warmup_steps=15, interval_steps=5),
        storage=f"sqlite:///{STUDY_DB}",
        load_if_exists=True,
    )
    # ponto bom conhecido primeiro: hp exatos da lstm_bi canonica (apenas se o
    # estudo ainda nao tem trial completo — em retomadas o trial entra de novo)
    if not any(t.state == optuna.trial.TrialState.COMPLETE
               for t in study.trials):
        study.enqueue_trial({
            "learning_rate": bi_params_full["learning_rate"],
            "weight_decay": bi_params_full.get("weight_decay", 0.0) or 1e-6,
            "lstm_units": bi_params_full["lstm_units"],
            "encoder_dropout_rate": bi_params_full["encoder_dropout_rate"],
            "decoder_dropout_rate": bi_params_full["decoder_dropout_rate"],
            "batch_size": int(bi_params_full.get("batch_size", 32)),
        })

    class OptunaReport(kcb.Callback):
        def __init__(self, trial):
            super().__init__()
            self.trial = trial

        def on_epoch_end(self, epoch, logs=None):
            loss = logs.get("val_loss")
            if loss is None:
                return
            self.trial.report(float(loss), epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned()

    def objective(trial):
        params = build_space(trial, bi_params_full)
        from keras import backend as K

        K.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        wrapper = common.wrapper_factory("lstm")()
        wrapper.prepare(
            train_df,
            val_df,
            input_steps=72,
            output_steps=36,
            target_col="ws100_wavelet",
            denoise=("ws100",),
            denoise_level=2,
        )
        batch_size = int(params["batch_size"])
        if hasattr(wrapper, "schedule_total_steps"):
            steps = int((len(train_df) + batch_size - 1) // batch_size)
            wrapper.schedule_total_steps = steps * args.epochs
        wrapper.build(common.FixedHyperParameters(params))
        started = time.perf_counter()
        history = wrapper.fit(
            epochs=args.epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[OptunaReport(trial)],
            use_validation=True,
        )
        val_loss = float(history.history["val_loss"][-1])
        print(f"[trial {trial.number}] {params} -> val_loss={val_loss:.5f} "
              f"({time.perf_counter() - started:.0f}s)")
        return val_loss

    study.optimize(objective, n_trials=args.trials, timeout=args.timeout,
                   gc_after_trial=True)
    pruned = len([t for t in study.trials
                  if t.state == optuna.trial.TrialState.PRUNED])
    best = study.best_trial
    best_params = dict(best.params)
    print(f"\nMelhor val_loss={best.value:.5f} (podados: {pruned})")
    print(f"Melhores hp: {best_params}")
    notify(f"Optuna LSTM 80/20: melhor val_loss={best.value:.5f} "
           f"({len(study.trials)} trials, {pruned} podados)")

    # re-treino final no treino COMPLETO (80%) com os melhores hp
    from keras import backend as K

    K.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    wrapper = common.wrapper_factory("lstm")()
    wrapper.prepare(
        train_full,
        train_full.tail(400),
        input_steps=72,
        output_steps=36,
        target_col="ws100_wavelet",
        denoise=("ws100",),
        denoise_level=2,
    )
    batch_size = int(best_params["batch_size"])
    if hasattr(wrapper, "schedule_total_steps"):
        steps = int((len(train_full) + batch_size - 1) // batch_size)
        wrapper.schedule_total_steps = steps * args.epochs
    wrapper.build(common.FixedHyperParameters(best_params))
    started = time.perf_counter()
    history = wrapper.fit(epochs=args.epochs, batch_size=batch_size, verbose=0,
                          callbacks=None, use_validation=False)
    elapsed = time.perf_counter() - started
    losses = history.history.get("loss", [])
    print(f"[final] 60 epocas no treino completo em {elapsed:.0f}s")

    model_dir = OUTPUT_DIR / "lstm"
    model_dir.mkdir(parents=True, exist_ok=True)
    wrapper.model.save(model_dir / "model.keras")
    common.write_json(model_dir / "model.json", {
        "wrapper_key": "lstm",
        "name": wrapper.name,
        "persistence_gate": False,
        "gate_mode": "static",
        "decoder_mode": getattr(wrapper, "decoder_mode", "direct"),
        "target_mode": getattr(wrapper, "target_mode", "absolute"),
        "normalize": bool(getattr(wrapper, "normalize", True)),
        "hyperparameters": best_params,
        "params_source": "optuna (80/20 metodologia atual)",
        "input_steps": 72,
        "output_steps": 36,
        "target_col": "ws100_wavelet",
        "denoise": ["ws100"],
        "denoise_level": 2,
        "training": {"epochs_run": len(losses), "seed": SEED,
                     "final_loss": float(losses[-1]) if losses else None,
                     "best_val_loss": best.value},
        "dataset": {"train_ratio": 0.80, "val_ratio": 0.0,
                    "split": "80/20 sem validacao (hp via 72/8 interno)"},
    })

    all_df = common.predict_all_horizons(wrapper, test_df)
    all_df["actual_raw"] = dataset["ws100"].reindex(
        all_df["timestamp"]).to_numpy()
    metrics = common.compute_metrics(all_df["actual_raw"], all_df["predicted"])
    persistence = common.compute_metrics(
        all_df["actual_raw"], all_df["persistence"])
    predictions, actuals = wrapper.rolling_forecast(dataset,
                                                    test_start=val_end_full)
    roll = common.compute_metrics(actuals.ravel(), predictions.ravel())

    canonical = json.load(open(CANONICAL))
    baselines = json.load(open(BASELINES))
    summary = {
        "lstm_hpo": {**metrics, "rolling": roll,
                     "skill": 1 - metrics["mse"] / persistence["mse"],
                     "val_loss": best.value, "hp": best_params,
                     "trials": len(study.trials), "pruned": pruned},
        "lstm_canonico": {k: canonical["lstm"][k]
                          for k in ("mse", "mae", "r2", "skill")},
        "lstm_bi_canonico": {k: canonical["lstm_bi"][k]
                             for k in ("mse", "mae", "r2", "skill")},
        "arima": {"all_horizons": baselines["arima"]["all_horizons"]},
        "persistencia": persistence,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    common.write_json(RESULTS_DIR / "summary.json", summary)

    print("\n=== comparacao (todos os horizontes vs ws100 bruto) ===")
    for label, m, roll_mse in (
            ("lstm otimizada", metrics, roll["mse"]),
            ("lstm canonica", canonical["lstm"], float("nan")),
            ("lstm_bi canonica", canonical["lstm_bi"], float("nan")),
            ("ARIMA", baselines["arima"]["all_horizons"],
             baselines["arima"]["rolling"]["mse"])):
        skill = 1 - m["mse"] / persistence["mse"]
        print(f"{label:<18} MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  "
              f"R2={m['r2']:+.3f}  skill={skill:+.1%}  "
              f"roll_MSE={roll_mse:.4f}")
    print(f"persistencia        MSE={persistence['mse']:.4f}")
    notify(f"Optuna LSTM concluido: teste MSE={metrics['mse']:.4f} "
           f"(canonica {canonical['lstm']['mse']:.4f})")
    print(f"\nEstudo: {STUDY_DB}\nResumo: {RESULTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
