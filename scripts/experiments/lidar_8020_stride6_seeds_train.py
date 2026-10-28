"""Estudo de seeds do decoder esparso stride-6 (h=6..36) — pareado.

Repete o experimento ``lidar_8020_stride6_train.py`` em várias seeds E treina
a variante canônica (36 passos densos) nas MESMAS seeds — só assim a
comparação stride × canônica se separa do ruído de seed (piso ≈ ±0,04 MSE;
R² ±0,07 nos estudos de seeds). Protocolo idêntico ao canônico: 80/20 sem
validação, 60 épocas, HPs Round 5, TCN=MSE+dilatações C′ / LSTM=ITILDE-Q.

Para cada (wrapper, seed) grava modelo + métricas em
``pipeline/tmp/trained_8020_stride6_seeds/<wrapper>/seed_<s>/`` (stride) e
``pipeline/tmp/trained_8020_canon36_seeds/<wrapper>/seed_<s>/`` (canônica).
Resumo com média±std e Wilcoxon pareado por seed (R² e MAE agregados nos
horizontes 6..36) em ``data/results/lidar_8020_stride6/seeds_summary.json``
— a seed 42 aproveita os artefatos já existentes (stride 1ª rodada +
report_8020_itildeq) como par adicional.

    python scripts/experiments/lidar_8020_stride6_seeds_train.py                    # seeds 43..47
    python scripts/experiments/lidar_8020_stride6_seeds_train.py --seeds 43 44     # subconjunto
    python scripts/experiments/lidar_8020_stride6_seeds_train.py --skip-canonical  # só stride
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
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src import common  # noqa: E402
import lidar_8020_stride6_train as s6  # noqa: E402

SEEDS = tuple(range(43, 48))  # 42 já existe (stride + canônica canônica)
WRAPPERS = ("lstm", "lstm_bi", "tcn", "tcn_bi")
OUT_STRIDE = common.resolve("pipeline/tmp/trained_8020_stride6_seeds")
OUT_CANON = common.resolve("pipeline/tmp/trained_8020_canon36_seeds")
SUMMARY = common.resolve("data/results/lidar_8020_stride6/seeds_summary.json")
CANONICAL_EVAL = s6.CANONICAL_EVAL
STRIDE_EVAL = common.resolve(
    "data/results/lidar_8020_stride6/evaluation.json")


def train_one(tf, wrapper_key, seed, variant, train_df, test_df, dataset,
              epochs):
    """Treina e avalia uma variante ('stride6' | 'canon36'); retorna o registro."""
    from keras import backend as K

    cfg = s6.CONFIGS[wrapper_key]
    hp = s6.round5_hp(wrapper_key)
    out_dir = (OUT_STRIDE if variant == "stride6" else OUT_CANON)
    model_dir = out_dir / wrapper_key / f"seed_{seed}"
    model_path = model_dir / "model.keras"

    record = None
    if model_path.is_file():
        meta = json.load(open(model_dir / "model.json", encoding="utf-8"))
        record = meta.get("stride6_summary")
        if record is not None:
            print(f"[{wrapper_key} seed={seed} {variant}] já existe — pulando "
                  f"(apague {model_dir} para retreinar)", flush=True)
            return record
    if record is None:
        K.clear_session()
        tf.keras.utils.set_random_seed(seed)
        if variant == "stride6":
            wrapper = s6.stride_wrapper(wrapper_key)
            output_steps = 6
        else:
            wrapper = common.wrapper_factory(wrapper_key)()
            output_steps = 36
        wrapper.prepare(
            train_df, train_df.tail(400),
            input_steps=72, output_steps=output_steps,
            target_col="ws100_wavelet", denoise=("ws100",), denoise_level=2,
        )
        batch_size = int(hp.get("batch_size", 32))
        if hasattr(wrapper, "schedule_total_steps"):
            steps = int((len(train_df) + batch_size - 1) // batch_size)
            wrapper.schedule_total_steps = steps * epochs

        wrapper.loss_mode_override = cfg["loss"]
        if cfg["loss"] == "itilde_q":
            s6.apply_itildeq_coeffs(wrapper_key, 0.3, 0.5, 0.1)
        if cfg["static_dilations"]:
            from src.models.tcn_hp import STATIC_DILATIONS

            wrapper.static_dilations = STATIC_DILATIONS
        wrapper.build(common.FixedHyperParameters(hp))
        assert wrapper.loss == cfg["loss"], (wrapper_key, wrapper.loss)

        print(f"[{wrapper_key} seed={seed} {variant}] treino {epochs} "
          f"epocas (loss={wrapper.loss})...", flush=True)
        started = time.perf_counter()
        history = wrapper.fit(epochs=epochs, batch_size=batch_size,
                              verbose=0, callbacks=None, use_validation=False)
        elapsed = time.perf_counter() - started
        losses = history.history.get("loss", [])
        print(f"[{wrapper_key} seed={seed} {variant}] {len(losses)} epocas "
              f"em {elapsed:.0f}s | loss final={losses[-1]:.5f}", flush=True)

        model_dir.mkdir(parents=True, exist_ok=True)
        wrapper.model.save(model_path)
        common.write_json(model_dir / "model.json", {
            "wrapper_key": wrapper_key,
            "name": wrapper.name,
            "experiment": ("stride6-seeds (decoder esparso h=6..36)"
                           if variant == "stride6"
                           else "canon36-seeds (controle pareado do stride6)"),
            "seed": seed,
            "output_stride": s6.STRIDE if variant == "stride6" else 1,
            "horizons": list(s6.HORIZONS),
            "persistence_gate": False,
            "gate_mode": "static",
            "decoder_mode": getattr(wrapper, "decoder_mode", "direct"),
            "target_mode": getattr(wrapper, "target_mode", "absolute"),
            "normalize": bool(getattr(wrapper, "normalize", True)),
            "hyperparameters": hp,
            "params_source": "HPs promovidos da Optuna Round 5",
            "input_steps": 72,
            "output_steps": output_steps,
            "target_col": "ws100_wavelet",
            "denoise": ["ws100"],
            "denoise_level": 2,
            "loss": cfg["loss"],
            "static_dilations": (wrapper.static_dilations
                                 if cfg["static_dilations"] else None),
            "training": {"epochs_run": len(losses),
                         "epochs_requested": epochs, "seed": seed,
                         "patience": 0,
                         "final_loss": float(losses[-1]) if losses else None,
                         "training_time_sec": round(elapsed, 1)},
            "dataset": {"train_ratio": 0.80, "val_ratio": 0.0,
                        "split": f"80/20 sem validacao ({variant} seeds)"},
        })

    # ---- avaliação: métricas por horizonte vs ws100 bruto -----------------
    if variant == "stride6":
        all_df = s6.predict_all_horizons_stride(wrapper, test_df)
    else:
        all_df = common.predict_all_horizons(wrapper, test_df)
    all_df["actual_raw"] = dataset["ws100"].reindex(
        all_df["timestamp"]).to_numpy()
    long_df = all_df.assign(actual=all_df["actual_raw"])
    per_h = common.horizon_metrics(long_df)
    sel = per_h[per_h["horizon"].isin(s6.HORIZONS)]
    record = {
        "seed": seed,
        "sparse": {k: round(float(v), 4)
                   for k, v in s6.aggregate(sel.to_dict(orient="records"))
                   .items()},
        "persistence": {k: round(float(v), 4) for k, v in s6.aggregate([
            common.compute_metrics(g["actual"], g["persistence"])
            for _, g in long_df.groupby("horizon") if
            int(g["horizon"].iloc[0]) in s6.HORIZONS]).items()},
        "per_horizon": [
            {"horizon": int(r["horizon"]),
             **{k: round(float(r[k]), 4) for k in ("mae", "mse", "rmse", "r2")}}
            for r in sel.to_dict(orient="records")],
    }
    meta = json.load(open(model_dir / "model.json", encoding="utf-8"))
    meta["stride6_summary"] = record
    common.write_json(model_dir / "model.json", meta)
    return record


def seed42_records(wrapper_key):
    """Par seed-42 reaproveitado: stride da 1ª rodada + canônica do report."""
    stride = json.load(open(STRIDE_EVAL, encoding="utf-8"))
    canon = json.load(open(CANONICAL_EVAL, encoding="utf-8"))
    s = stride["wrappers"][wrapper_key]
    c_rows = [r for r in canon[s6.SEQ2SEQ_NAME[wrapper_key]]["per_horizon"]
              if int(r["horizon"]) in s6.HORIZONS]
    return {
        "stride6": {"seed": 42, "source": str(STRIDE_EVAL),
                    "sparse": s["all_sparse"],
                    "persistence": s["persistence_sparse"]},
        "canon36": {"seed": 42, "source": str(CANONICAL_EVAL),
                    "sparse": s6.aggregate(c_rows)},
    }


def paired_stats(pairs_s, pairs_c, key):
    """Wilcoxon pareado por seed na métrica *key* (stride vs canônica)."""
    from scipy.stats import wilcoxon

    x = np.array([p["sparse"][key] for p in pairs_s], dtype=float)
    y = np.array([p["sparse"][key] for p in pairs_c], dtype=float)
    diff = x - y
    out = {
        "metric": key,
        "stride6_mean": round(float(x.mean()), 4),
        "stride6_std": round(float(x.std(ddof=1)), 4),
        "canon36_mean": round(float(y.mean()), 4),
        "canon36_std": round(float(y.std(ddof=1)), 4),
        "delta_mean": round(float(diff.mean()), 4),
        "n_seeds": int(len(x)),
    }
    if len(x) >= 6 and np.any(diff != 0):
        stat, p = wilcoxon(x, y)
        out["wilcoxon_p"] = round(float(p), 4)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--wrappers", nargs="+", default=list(WRAPPERS),
                        choices=WRAPPERS)
    parser.add_argument("--skip-canonical", action="store_true",
                        help="treina só o stride-6 (sem o par canônico)")
    global OUT_STRIDE, OUT_CANON, SUMMARY
    parser.add_argument("--stride-dir", default=str(OUT_STRIDE))
    parser.add_argument("--canon-dir", default=str(OUT_CANON))
    parser.add_argument("--summary", default=str(SUMMARY))
    args = parser.parse_args()
    OUT_STRIDE = common.resolve(args.stride_dir)
    OUT_CANON = common.resolve(args.canon_dir)
    SUMMARY = common.resolve(args.summary)

    tf = common.setup_tensorflow(True)
    print(f"TensorFlow {tf.__version__} | stride6-seeds pareado | "
          f"seeds {args.seeds} (+42 reaproveitada) | "
          f"{'stride only' if args.skip_canonical else 'stride + canon36'}",
          flush=True)

    dataset = common.load_dataset(common.resolve("data/dataset.csv"))
    train_df, _val, test_df = common.split_dataset(dataset, 0.80, 0.0)
    print(f"Dataset {len(dataset)} | treino {len(train_df)} | "
          f"teste {len(test_df)}", flush=True)

    summary_path = SUMMARY
    summary = {}
    if summary_path.is_file():
        summary = json.load(open(summary_path, encoding="utf-8"))

    for wrapper_key in args.wrappers:
        entry = summary.get(wrapper_key, {"stride6": {}, "canon36": {}})
        if "42" not in entry["stride6"]:
            seed42 = seed42_records(wrapper_key)
            entry["stride6"]["42"] = seed42["stride6"]
            entry["canon36"]["42"] = seed42["canon36"]
        for seed in args.seeds:
            s_rec = train_one(tf, wrapper_key, seed, "stride6", train_df,
                              test_df, dataset, args.epochs)
            entry["stride6"][str(seed)] = s_rec
            if not args.skip_canonical:
                c_rec = train_one(tf, wrapper_key, seed, "canon36", train_df,
                                  test_df, dataset, args.epochs)
                entry["canon36"][str(seed)] = c_rec
            # resumo incremental após cada seed (restart-safe)
            seeds_sorted = sorted(entry["stride6"], key=int)
            pairs_s = [entry["stride6"][s] for s in seeds_sorted
                       if s in entry["canon36"]]
            pairs_c = [entry["canon36"][s] for s in seeds_sorted
                       if s in entry["canon36"]]
            entry["seeds"] = seeds_sorted
            entry["paired"] = [paired_stats(pairs_s, pairs_c, k)
                               for k in ("r2", "mae", "rmse")]
            summary[wrapper_key] = entry
            common.write_json(summary_path, {
                "study": "stride6_seeds_pareado",
                "protocol": {"train_ratio": 0.80, "val_ratio": 0.0,
                             "epochs": args.epochs, "input_steps": 72,
                             "horizons": list(s6.HORIZONS),
                             "loss_by_wrapper": {k: v["loss"] for k, v in
                                                 s6.CONFIGS.items()}},
                "wrappers": summary,
            })
            p = entry["paired"]
            print(f"[{wrapper_key}] n={p[0]['n_seeds']} | "
                  f"R² stride {p[0]['stride6_mean']}±{p[0]['stride6_std']} vs "
                  f"canônica {p[0]['canon36_mean']}±{p[0]['canon36_std']} "
                  f"(Δ={p[0]['delta_mean']}, p={p[0].get('wilcoxon_p')}) | "
                  f"MAE {p[1]['stride6_mean']} vs {p[1]['canon36_mean']} "
                  f"(p={p[1].get('wilcoxon_p')})", flush=True)

    print(f"\nResumo: {summary_path}")
    print(f"Modelos: {OUT_STRIDE}\n         {OUT_CANON}")


if __name__ == "__main__":
    main()
