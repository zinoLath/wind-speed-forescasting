"""Sweep 80/20 sobre o conjunto 1-SE da Optuna round 7.

Treina TODOS os trials candidatos da regra 1-SE (não só o escolhido) com
o protocolo padrão (80/20 sem validação, 60 épocas, seed 42, MSE) e
avalia cada um no teste de 20% — diagnóstico de quão estável é a região
selecionada. NÃO promove nada e NÃO sobrescreve os modelos do relatório
(trained_8020_optuna7); grava em trained_8020_optuna7_sweep + resumo em
data/results/lidar_8020_optuna7_sweep/sweep.json (resume por trial).

    python scripts/experiments/lidar_8020_optuna7_sweep.py            # todos os wrappers
    python scripts/experiments/lidar_8020_optuna7_sweep.py --wrappers lstm
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src import common
from optuna_select_1se import WRAPPER_DIRS, select_study

STUDY_NAME = "round7"
STORAGE_DIR = common.resolve("pipeline/tmp/optuna_round7")
OUTPUT_DIR = common.resolve("pipeline/tmp/trained_8020_optuna7_sweep")
SUMMARY_DIR = common.resolve("data/results/lidar_8020_optuna7_sweep")
TRAIN_RATIO = 0.80
EPOCHS = 60
BATCH_SIZE = 32
SEED = 42


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wrappers", nargs="*", default=None,
                        help="subset (default: os com optuna.db na rodada)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    keys = args.wrappers or [k for k, d in WRAPPER_DIRS.items()
                             if (STORAGE_DIR / d / "optuna.db").is_file()]
    if not keys:
        raise SystemExit(f"Nenhum optuna.db em {STORAGE_DIR}")

    tf = common.setup_tensorflow(True)
    print(f"TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")

    dataset = common.load_dataset(common.resolve("data/dataset.csv"),
                                  keep_raw=("ws100",))
    train_df, val_df, test_df = common.split_dataset(dataset, TRAIN_RATIO, 0.0)
    assert len(val_df) == 0 and len(test_df) > 0
    print(f"Dataset: {len(dataset)} linhas | treino {len(train_df)} | "
          f"teste {len(test_df)} | validacao: REMOVIDA")

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SUMMARY_DIR / "sweep.json"
    rows = []
    if summary_path.is_file():
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
    done = {(r["wrapper"], r["trial"]) for r in rows}

    for key in keys:
        db = STORAGE_DIR / WRAPPER_DIRS[key] / "optuna.db"
        sel = select_study(db, STUDY_NAME, key)
        if sel is None:
            print(f"[{key}] sem trials COMPLETE; pulando")
            continue
        chosen, best = sel["chosen"], sel["best"]
        ordered = ([chosen] if chosen.number != best.number else []) \
            + [best] + sorted(sel["candidates"], key=lambda t: t.value)
        seen, plan = set(), []
        for t in ordered:
            if t.number not in seen:
                seen.add(t.number)
                plan.append(t)
        n_total = len(plan)
        n_skip = sum(1 for t in plan if (key, t.number) in done)
        SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
        meta_path = SUMMARY_DIR / "sweep_meta.json"
        meta = (json.loads(meta_path.read_text(encoding="utf-8"))
                if meta_path.is_file() else {})
        meta[key] = {"n_total": n_total, "se": round(sel["se"], 4),
                     "chosen": chosen.number, "argmin": best.number}
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"\n[{key}] conjunto 1-SE: {n_total} trials "
              f"(SE={sel['se']:.4f}; escolhido #{chosen.number}, "
              f"argmin #{best.number}) | ja feitos: {n_skip}")
        if n_skip >= n_total:
            continue

        for i, trial in enumerate(plan, start=1):
            if (key, trial.number) in done:
                continue
            hp = dict(trial.params)
            from keras import backend as K

            K.clear_session()
            tf.keras.utils.set_random_seed(SEED)
            wrapper = common.wrapper_factory(key)()
            val_for_prepare = train_df.tail(400)
            wrapper.prepare(
                train_df,
                val_for_prepare,
                input_steps=72,
                output_steps=36,
                target_col="ws100_wavelet",
                denoise=("ws100",),
                denoise_level=2,
            )
            batch_size = int(hp.get("batch_size", BATCH_SIZE))
            if hasattr(wrapper, "schedule_total_steps"):
                steps = int((len(train_df) + batch_size - 1) // batch_size)
                wrapper.schedule_total_steps = steps * args.epochs
            wrapper.loss = "mse"
            wrapper.loss_mode_override = "mse"
            wrapper.build(common.FixedHyperParameters(hp))

            print(f"[{key}] ({i}/{n_total}) trial #{trial.number} "
                  f"val={trial.value:+.4f} hp={hp} ...", flush=True)
            started = time.perf_counter()
            history = wrapper.fit(
                epochs=args.epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=None,
                use_validation=False,
            )
            losses = history.history.get("loss", [])
            train_sec = time.perf_counter() - started

            all_df = common.predict_all_horizons(wrapper, test_df)
            all_df["actual_raw"] = dataset["ws100"].reindex(
                all_df["timestamp"]
            ).to_numpy()
            metrics = common.compute_metrics(
                all_df["actual_raw"], all_df["predicted"])
            persistence = common.compute_metrics(
                all_df["actual_raw"], all_df["persistence"])
            skill = (1 - metrics["mse"] / persistence["mse"]
                     if persistence["mse"] else None)

            model_dir = OUTPUT_DIR / key / f"trial_{trial.number:04d}"
            model_dir.mkdir(parents=True, exist_ok=True)
            wrapper.model.save(model_dir / "model.keras")

            rows.append({
                "wrapper": key,
                "trial": trial.number,
                "is_1se_chosen": trial.number == chosen.number,
                "is_argmin": trial.number == best.number,
                "val_objective": trial.value,
                "val_skill": trial.user_attrs.get("skill"),
                "val_skill_blocks": trial.user_attrs.get("skill_blocks"),
                "params": hp,
                "test": metrics,
                "test_skill_persistence": skill,
                "persistence": persistence,
                "persistence_mse": persistence["mse"],
                "final_loss": float(losses[-1]) if losses else None,
                "train_sec": round(train_sec, 1),
            })
            summary_path.write_text(
                json.dumps(rows, indent=1), encoding="utf-8")
            print(f"[{key}] trial #{trial.number}: "
                  f"MSE={metrics['mse']:.4f} RMSE={metrics['rmse']:.4f} "
                  f"skill={skill:.3f} ({train_sec:.0f}s) "
                  f"-> resumo com {len(rows)} entradas", flush=True)

    # CSV plano para inspeção rápida
    csv_path = SUMMARY_DIR / "sweep.csv"
    cols = ["wrapper", "trial", "is_1se_chosen", "is_argmin",
            "val_objective", "val_skill", "test_mse", "test_rmse",
            "test_mae", "test_skill_persistence", "train_sec"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        for r in rows:
            t = r["test"]
            writer.writerow([
                r["wrapper"], r["trial"], r["is_1se_chosen"], r["is_argmin"],
                r["val_objective"], r["val_skill"], t.get("mse"),
                t.get("rmse"), t.get("mae"), r["test_skill_persistence"],
                r["train_sec"]])
    print(f"\nResumo: {summary_path}")
    print(f"CSV: {csv_path}")
    print(f"Modelos: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
