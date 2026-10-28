"""Painel de diagnóstico das predições — por que uma perda converge melhor?

Para cada modelo (wrapper já treinado, carregado dos pesos): gera as
predições multi-horizonte e mede não só o erro, mas o COMPORTAMENTO da
série prevista:

- erro: MSE/MAE/R² (all) e viés médio (pred − actual);
- amplitude: razão std(pred)/std(actual) (encolhimento da variância?);
- suavidade: energia de alta frequência (rfft, bins ≥ 4) / total;
- dinâmica: autocorrelação lag-1 (e a do real, como referência);
- forma: DTW médio (distância de forma, alinhamento temporal) e
  correlação de Pearson média por origem;
- fase: defasagem média |lag| pelo pico da correlação cruzada.

Uso:
    python scripts/prediction_diagnostics.py \
        --variants itilde_q=pipeline/tmp/trained_8020_itildeq/lstm \
                   hweight=pipeline/tmp/trained_8020/lstm \
                   mse=pipeline/tmp/trained_8020_mseloss/lstm
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "results" / "prediction_diagnostics"


def load_rr():
    path = PROJECT_ROOT / "reports" / "src" / "relatorio_resultados.py"
    spec = importlib.util.spec_from_file_location("relatorio_resultados", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["relatorio_resultados"] = mod
    spec.loader.exec_module(mod)
    return mod


def dtw_distance(a, b):
    """DTW clássico (|x−y|), normalizado pelo comprimento do caminho."""
    n, m = len(a), len(b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = abs(a[i - 1] - b[j - 1]) + min(
                cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return cost[n, m] / (n + m)


def panel(entry, dataset):
    """Métricas de comportamento por origem (36 passos cada)."""
    groups = [g.sort_values("horizon") for _, g in entry.groupby("origin")]
    groups = [g for g in groups if len(g) == 36]  # origens completas
    preds = np.stack([g["predicted"].to_numpy() for g in groups])
    acts = np.stack([g["actual"].to_numpy() for g in groups])
    n = len(preds)

    def ac1(mat):
        m = mat - mat.mean(axis=1, keepdims=True)
        num = (m[:, 1:] * m[:, :-1]).sum(axis=1)
        den = (m * m).sum(axis=1) + 1e-12
        return float(np.mean(num / den))

    spec_p = np.abs(np.fft.rfft(preds, axis=1)) ** 2
    spec_a = np.abs(np.fft.rfft(acts, axis=1)) ** 2
    hf_p = float(np.mean(spec_p[:, 4:].sum(1) / (spec_p.sum(1) + 1e-12)))
    hf_a = float(np.mean(spec_a[:, 4:].sum(1) / (spec_a.sum(1) + 1e-12)))

    corr = float(np.mean([np.corrcoef(p, a)[0, 1] for p, a in
                          zip(preds, acts)]))
    dtw = float(np.mean([dtw_distance(p, a) for p, a in zip(preds, acts)]))

    lags = []
    for p, a in zip(preds, acts):
        p = p - p.mean()
        a = a - a.mean()
        xc = np.correlate(p, a, mode="full")
        lag = int(np.argmax(xc) - (len(a) - 1))
        lags.append(abs(lag))
    lag_mean = float(np.mean(lags))

    err = preds - acts
    mse = float(np.mean(err ** 2))
    return {
        "n_origins": n,
        "mse": mse,
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - mse / np.var(acts)),
        "bias": float(np.mean(err)),
        "std_ratio": float(np.std(preds) / np.std(acts)),
        "ac1_pred": ac1(preds),
        "ac1_actual": ac1(acts),
        "hf_ratio_pred": hf_p,
        "hf_ratio_actual": hf_a,
        "corr_mean": corr,
        "dtw_mean": dtw,
        "lag_mean_steps": lag_mean,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", required=True,
                    help="rótulo=caminho/do/modelo (usa model.json + pesos)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    tf = common.setup_tensorflow(True)
    rr = load_rr()
    dataset = common.load_dataset(rr.DATA_FILE, keep_raw=("ws40", "ws100", "v40"))
    observed = common.observed_timestamps(rr.DATA_FILE)
    print(f"TensorFlow {tf.__version__} | {len(args.variants)} variante(s)")

    results = {}
    for spec in args.variants:
        label, path = spec.split("=", 1)
        model_dir = Path(path) if Path(path).is_absolute() else \
            PROJECT_ROOT / path
        meta = json.loads((model_dir / "model.json").read_text("utf-8"))
        ratios = meta.get("dataset", {})
        train_df, val_df, _ = common.split_dataset(
            dataset, ratios.get("train_ratio", 0.8),
            ratios.get("val_ratio", 0.0))
        val_end = len(train_df) + len(val_df)
        wrapper = rr.build_wrapper(meta, train_df,
                                   val_df if len(val_df) else train_df.tail(400),
                                   model_dir / "model.keras")
        eval_df = dataset.iloc[max(0, val_end - wrapper.input_steps):]
        df = common.predict_all_horizons(wrapper, eval_df)
        df["actual"] = dataset["ws100"].reindex(df["timestamp"]).to_numpy()
        if observed is not None:
            df = df[df["timestamp"].isin(observed)]
        results[label] = {"loss": meta.get("loss"), **panel(df, dataset)}
        print(f"[{label}] loss={meta.get('loss')} | "
              f"R²={results[label]['r2']:.4f} "
              f"MSE={results[label]['mse']:.4f}")

    cols = [("mse", 4), ("r2", 4), ("bias", 4), ("std_ratio", 3),
            ("hf_ratio_pred", 3), ("ac1_pred", 3), ("corr_mean", 3),
            ("dtw_mean", 3), ("lag_mean_steps", 1)]
    print(f"\n{'variante':<12} " + " ".join(f"{c:>{w}}" for c, w in cols))
    for label, r in results.items():
        print(f"{label:<12} " + " ".join(
            f"{r[c]:>{w}.{w}}" if isinstance(r[c], float) else
            f"{r[c]:>{w}}" for c, w in cols))
    print("\nreferência do real (mesmas origens): "
          f"hf={results[next(iter(results))]['hf_ratio_actual']:.3f} "
          f"ac1={results[next(iter(results))]['ac1_actual']:.3f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnostics.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8")
    print(f"salvo em {out / 'diagnostics.json'}")


if __name__ == "__main__":
    main()
