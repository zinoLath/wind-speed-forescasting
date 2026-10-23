"""80/20 — LSTM-Bi da rodada final SBSR em várias seeds (variância).

Mantém **tudo fixo** da rodada `trained_8020_sbsr` (featurex 67 colunas,
MSE pura, HPs canônicos do HPO 80/20, 60 épocas, split 80/20 sem
validação, janela 72→36, alvo ``ws100_wavelet``) e varia apenas a **seed**.
Para cada seed grava modelo, métricas e predições (denso com filtro de
observadas + rolling) no mesmo formato do estudo ``itildeq_seeds_8020``.

    python scripts/lidar_8020_sbsr_seeds_train.py --wrappers lstm_bi \
        --seeds 43 44 45 46

Saída: ``pipeline/tmp/trained_8020_sbsr_seeds/<wrapper>/seed_<seed>/``
(``model.keras``, ``model.json``, ``predictions_all.csv``,
``predictions_rolling.csv``) e ``<wrapper>/summary.json``.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common

DEFAULT_SEEDS = (43, 44, 45, 46)
DEFAULT_OUTPUT = "pipeline/tmp/trained_8020_sbsr_seeds"
CANON_DIR = common.resolve("pipeline/tmp/trained_8020_itildeq")
KEEP_FEATUREX = ("humid", "temp") + tuple(f"cis{i}" for i in range(1, 20))
WRAPPER_MODULES = {
    "lstm": "src.models.s2s_lstm_wrapper",
    "lstm_bi": "src.models.s2s_lstm_bi_wrapper",
}


def load_hp(wrapper_key, hp_json):
    if hp_json:
        path = Path(hp_json)
        if path.is_dir():
            path = path / wrapper_key / "model.json"
        meta = json.load(open(path, encoding="utf-8"))
        return dict(meta.get("hyperparameters", meta))
    path = CANON_DIR / wrapper_key / "model.json"
    meta = json.load(open(path, encoding="utf-8"))
    return dict(meta["hyperparameters"])


def environment_info(tf):
    info = {
        "tensorflow": tf.__version__,
        "hostname": os.uname().nodename,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import subprocess
        info["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=False).stdout.strip()
    except Exception:  # noqa: BLE001 - metadado opcional
        info["gpu"] = None
    try:
        import subprocess
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip()
    except Exception:  # noqa: BLE001
        info["git_commit"] = None
    return info


def train_one(tf, wrapper_key, seed, dataset, train_df, test_df, val_end,
              observed, args, env):
    from keras import backend as K

    hp = load_hp(wrapper_key, args.hp_json)
    batch_size = int(hp.get("batch_size", 32))

    K.clear_session()
    tf.keras.utils.set_random_seed(seed)
    wrapper = common.wrapper_factory(wrapper_key)()
    wrapper.prepare(
        train_df, train_df.tail(400),
        input_steps=72, output_steps=36, target_col="ws100_wavelet",
        denoise=("ws100",), denoise_level=2,
    )
    if hasattr(wrapper, "schedule_total_steps"):
        steps = int((len(train_df) + batch_size - 1) // batch_size)
        wrapper.schedule_total_steps = steps * args.epochs

    wrapper.loss_mode_override = "mse"
    wrapper.build(common.FixedHyperParameters(hp))
    assert wrapper.loss == "mse", (wrapper_key, wrapper.loss)

    print(f"[{wrapper_key} seed={seed}] treino {args.epochs} épocas "
          f"(loss=mse, bs={batch_size})...", flush=True)
    started = time.perf_counter()
    history = wrapper.fit(epochs=args.epochs, batch_size=batch_size,
                          verbose=0, callbacks=None, use_validation=False)
    elapsed = time.perf_counter() - started
    losses = history.history.get("loss", [])

    model_dir = OUTPUT_DIR / wrapper_key / f"seed_{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    wrapper.model.save(model_dir / "model.keras")

    eval_df = dataset.iloc[max(0, val_end - wrapper.input_steps):]
    all_df = common.predict_all_horizons(wrapper, eval_df)
    all_df = all_df.rename(columns={"actual": "actual_wavelet"})
    all_df["actual_raw"] = dataset["ws100"].reindex(
        all_df["timestamp"]).to_numpy()
    if observed is not None:
        all_df = all_df[all_df["timestamp"].isin(observed)]
    all_df = all_df.assign(actual=all_df["actual_raw"])
    all_df.to_csv(model_dir / "predictions_all.csv", index=False)

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

    m_all = common.compute_metrics(all_df["actual"], all_df["predicted"])
    m_pers = common.compute_metrics(all_df["actual"], all_df["persistence"])
    m_roll = common.compute_metrics(roll_df["actual_raw"],
                                    roll_df["predicted"])
    pers36 = (all_df[all_df["horizon"] == all_df["horizon"].max()]
              .drop_duplicates("origin").set_index("origin")["persistence"])
    roll_df["persistence"] = roll_df["origin"].map(pers36)
    valid = roll_df["persistence"].notna()
    m_pers_roll = common.compute_metrics(
        roll_df.loc[valid, "actual_raw"],
        roll_df.loc[valid, "persistence"])

    meta = {
        "wrapper_key": wrapper_key,
        "name": wrapper.name,
        "seed": seed,
        "loss": "mse",
        "hyperparameters": hp,
        "params_source": "canônica 16/10 (trained_8020_itildeq) com perda "
                         "MSE pura — rodada final SBSR (variância de seed)",
        "feature_keep": list(KEEP_FEATUREX),
        "feature_set": ("perfil 46 + cis1..19 + humid/temp (67 col; "
                        "sem press/disp/vdisp)"),
        "params": int(wrapper.model.count_params()),
        "persistence_gate": False,
        "gate_mode": "static",
        "decoder_mode": getattr(wrapper, "decoder_mode", "direct"),
        "target_mode": getattr(wrapper, "target_mode", "absolute"),
        "normalize": bool(getattr(wrapper, "normalize", True)),
        "input_steps": 72,
        "output_steps": 36,
        "target_col": "ws100_wavelet",
        "denoise": ["ws100"],
        "denoise_level": 2,
        "training": {
            "epochs_run": len(losses),
            "epochs_requested": args.epochs,
            "batch_size": batch_size,
            "seed": seed,
            "patience": 0,
            "final_loss": float(losses[-1]) if losses else None,
            "best_val_loss": None,
            "final_val_loss": None,
            "training_time_sec": round(elapsed, 1),
        },
        "evaluation": {
            "all": m_all,
            "rolling": m_roll,
            "persistence_all": m_pers,
            "persistence_rolling": m_pers_roll,
            "skill": float(1.0 - m_all["mse"] / m_pers["mse"]),
            "skill_rolling": float(1.0 - m_roll["mse"] / m_pers_roll["mse"]),
        },
        "dataset": {
            "train_ratio": 0.80,
            "val_ratio": 0.0,
            "split": "80/20 sem validação (estudo de seeds SBSR)",
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        },
        "environment": env,
    }
    common.write_json(model_dir / "model.json", meta)
    print(f"[{wrapper_key} seed={seed}] {elapsed:.0f}s | "
          f"MSE={m_all['mse']:.4f} MAE={m_all['mae']:.4f} "
          f"R²={m_all['r2']:+.3f} | rolling={m_roll['mse']:.4f}",
          flush=True)
    return meta


def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--wrappers", nargs="+", default=["lstm_bi"],
                        choices=list(WRAPPER_MODULES))
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--hp-json", default=None,
                        help="model.json (ou diretório com <wrapper>/"
                             "model.json) para sobrescrever os HPs")
    args = parser.parse_args()

    OUTPUT_DIR = common.resolve(args.output_dir)

    tf = common.setup_tensorflow(True)
    env = environment_info(tf)
    print(f"TensorFlow {tf.__version__} | GPU: {env['gpu']} | "
          f"host: {env['hostname']}", flush=True)

    dataset = common.load_dataset(common.resolve("data/dataset.csv"),
                                  keep_raw=KEEP_FEATUREX)
    train_df, _val_df, test_df = common.split_dataset(dataset, 0.80, 0.0)
    val_end = len(train_df)
    observed = common.observed_timestamps(common.resolve("data/dataset.csv"))
    print(f"Dataset {len(dataset)} linhas x {dataset.shape[1]} colunas | "
          f"treino {len(train_df)} | teste {len(test_df)} | "
          f"seeds {args.seeds}", flush=True)

    for wrapper_key in args.wrappers:
        summary = {}
        summary_path = OUTPUT_DIR / wrapper_key / "summary.json"
        for seed in args.seeds:
            meta = train_one(tf, wrapper_key, seed, dataset, train_df,
                             test_df, val_end, observed, args, env)
            summary[str(seed)] = meta
            common.write_json(summary_path, {
                "study": "sbsr_seeds_8020",
                "wrapper_key": wrapper_key,
                "seeds": args.seeds,
                "protocol": {
                    "train_ratio": 0.80, "val_ratio": 0.0,
                    "epochs": args.epochs, "input_steps": 72,
                    "output_steps": 36, "target_col": "ws100_wavelet",
                    "denoise": ["ws100"], "denoise_level": 2,
                    "loss": "mse",
                    "feature_keep": list(KEEP_FEATUREX),
                },
                "environment": env,
                "runs": summary,
            })
        print(f"[{wrapper_key}] {len(summary)} seeds concluídas -> "
              f"{summary_path}", flush=True)

    print(f"\nModelos e predições em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
