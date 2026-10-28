"""Erro normalizado + Diebold–Mariano ajustado — rodada mista 8020-itildeq.

Análise pós-modelo sobre a MESMA grade de origens (DL + baselines ARIMA/RF/
persistência) que responde a duas perguntas:

1. **Erro normalizado pelo intrínseco do dado.** Os modelos erram sobretudo
   *onde o dado é difícil* (rampas, extremos) — ali a persistência também
   erra muito. Para separar "o dado é difícil" de "o modelo é ruim", cada
   erro é escalado pelo erro da persistência **na mesma célula/estrato**:

   - ``skill_h = 1 − MSE_modelo,h / MSE_persist,h`` (por horizonte, com IC);
   - ``MASE = MAE_modelo / MAE_persist`` (Hyndman & Koehler 2006);
   - ``ratio_local = |e_modelo| / |e_persist|`` (média/mediana por célula);
   - por regime diurno/noturno e por faixa de intensidade;
   - ``erro/piso`` = MAE_modelo / piso de suavização wavelet (erro
     irredutível do alvo: bruto vs wavelet).

2. **Diebold–Mariano com ajustes.** A mesma bateria de pares é avaliada por
   quatro variantes, para o relatório explicar a diferença entre elas:

   - ``DM(HLN+HAC)`` — variância HAC com lag h−1 + correção de pequena
     amostra de Harvey–Leybourne–Newbold (1997), t(n−1);
   - ``DM(NW auto)`` — HAC com banda automática de Newey–West (1994) e
     normal assintótica;
   - ``Holm`` — p-valores do DM(HLN+HAC) ajustados por multiplicidade;
   - ``Clark–West`` — MSPE-ajustado (2007) para o benchmark aninhado
     (persistência).

Somente CPU leves + GPU para reconstruir as predições (mesma avaliação do
relatório canônico).

    source .venv/bin/activate && source scripts/tf_gpu_env.sh
    python tests/metodos_itildeq.py \
        --models-dir pipeline/tmp/trained_8020_itildeq \
        --out-dir data/results/metodos_itildeq
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import metodos_complementares as mc
import relatorio_resultados as rr
from src import common
from src.utils import (
    adjust_pvalues,
    clark_west_stat,
    diebold_mariano_loss_diff,
    newey_west_lag,
    wavelet_denoising,
)

ALPHA = 0.05
SEED = 42
BANDS = mc.BANDS
BAND_LABELS = mc.BAND_LABELS
EPS = 1e-6


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}".replace(".", ",")


def savefig(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=140)
    plt.close(fig)


def build_matrices(results, extras, base_pred):
    """Matrizes origin×horizon de erros (actual_raw − pred) por método."""
    matrices, shorts = {}, {}
    for name, m in results.items():
        matrices[name] = mc.error_matrix_long(extras[name]["all_df"], "predicted")
        shorts[name] = m["short"]
    first = extras[results[list(results)[0]]["name"]]["all_df"]
    matrices["__persistence__"] = mc.error_matrix_long(first, "persistence")
    shorts["__persistence__"] = "Persistência"
    if base_pred is not None:
        sub = base_pred.rename(columns={"actual": "actual_raw"}).copy()
        sub["origin"] = sub["origin"].dt.round("min")
        sub["timestamp"] = sub["timestamp"].dt.round("min")
        for col, label in (("arima", "ARIMA"), ("rf", "Random Forest")):
            key = "__" + col + "__"
            matrices[key] = mc.error_matrix_long(sub, col)
            shorts[key] = label
    matrices = {n: m.dropna(axis=0) for n, m in matrices.items()}
    idx = None
    for mat in matrices.values():
        idx = mat.index if idx is None else idx.intersection(mat.index)
    idx = idx.sort_values()
    return {n: m.loc[idx] for n, m in matrices.items()}, shorts, idx


def dm_variants(matrices, shorts):
    """DM (HLN+HAC), DM (NW auto), p de Holm e Clark–West por par."""
    names = [n for n in matrices if n != "__persistence__"]
    rows = []
    per_h = matrices[names[0]].shape[1]
    raw_p = []
    for a, b in combinations(names, 2):
        A = matrices[a].to_numpy(float)
        B = matrices[b].to_numpy(float)
        d = (A ** 2).mean(axis=1) - (B ** 2).mean(axis=1)
        hln = diebold_mariano_loss_diff(d, h=per_h)
        nw = diebold_mariano_loss_diff(
            d, h=per_h, hac_lag=newey_west_lag(d.size), hln=False)
        rows.append({
            "a": shorts[a], "b": shorts[b],
            "dMSE": hln["mean_loss_diff"],
            "dm_hln": hln["dm"], "p_hln": hln["p_value"],
            "dm_nw": nw["dm"], "p_nw": nw["p_value"],
            "melhor": shorts[a] if hln["mean_loss_diff"] < 0 else shorts[b],
        })
        raw_p.append(hln["p_value"])
    adj = adjust_pvalues(raw_p, "holm")
    for row, padj in zip(rows, adj):
        row["p_holm"] = float(padj)
        row["sig_hln"] = row["p_hln"] < ALPHA
        row["sig_holm"] = padj < ALPHA
        row["sig_nw"] = row["p_nw"] < ALPHA
    # Clark–West: cada modelo (irrestrito) contra a persistência (restrito).
    # f_t = e_b² − e_m² + (e_m − e_b)², com e_b = erro da persistência e
    # e_m = erro do modelo; agrega por origem (média dos 36 horizontes) para
    # absorver a dependência das janelas sobrepostas.
    cw_rows = []
    pers = matrices["__persistence__"].to_numpy(float)
    for n in names:
        A = matrices[n].to_numpy(float)
        f = pers ** 2 - A ** 2 + (A - pers) ** 2
        cw = clark_west_stat(f.mean(axis=1))
        cw_rows.append({"modelo": shorts[n],
                        "dMSE": float((pers ** 2 - A ** 2).mean()),
                        "cw": cw["cw"], "p": cw["p_value"],
                        "favorece": shorts[n] if cw["cw"] > 0 else "Persistência"})
    return pd.DataFrame(rows).sort_values("p_hln"), pd.DataFrame(cw_rows)


def normalized_metrics(matrices, shorts):
    """MASE, skill global e razão local |e|/|e_persist| por método."""
    pers = matrices["__persistence__"].to_numpy(float)
    mse_p = float((pers ** 2).mean())
    mae_p = float(np.abs(pers).mean())
    rows = []
    for n, mat in matrices.items():
        A = mat.to_numpy(float)
        ratio = np.abs(A) / (np.abs(pers) + EPS)
        rows.append({
            "modelo": shorts[n],
            "MAE": float(np.abs(A).mean()),
            "MASE": float(np.abs(A).mean() / mae_p),
            "skill": 1.0 - float((A ** 2).mean()) / mse_p,
            "ratio_medio": float(ratio.mean()),
            "ratio_mediano": float(np.median(ratio)),
        })
    df = pd.DataFrame(rows).sort_values("MASE")
    return df


def skill_by_horizon(matrices, shorts, rng, n_boot=500, mean_block=24):
    """Skill vs persistência por horizonte + IC 95% (stationary bootstrap)."""
    pers = matrices["__persistence__"].to_numpy(float)
    rows = []
    for n, mat in matrices.items():
        if n == "__persistence__":
            continue
        A = mat.to_numpy(float)
        for j in range(A.shape[1]):
            mse_m = float((A[:, j] ** 2).mean())
            mse_p = float((pers[:, j] ** 2).mean())
            d = A[:, j] ** 2 - pers[:, j] ** 2
            lo, hi = mc.boot_ci_mean(d, rng, n_boot=n_boot,
                                      mean_block=mean_block)
            rows.append({"modelo": shorts[n], "key": n, "horizon": j + 1,
                         "skill": 1.0 - mse_m / mse_p,
                         "ic_inf": 1.0 - hi / mse_p, "ic_sup": 1.0 - lo / mse_p})
    return pd.DataFrame(rows)


def strata_normalized(results, extras, shorts):
    """Razão MAE_modelo/MAE_persistência e skill por regime e por faixa."""
    frames = []
    for name, m in results.items():
        df = extras[name]["all_df"][
            ["origin", "horizon", "timestamp", "actual_raw",
             "predicted", "persistence"]].copy()
        df["modelo"] = m["short"]
        frames.append(df)
    stacked = pd.concat(frames, ignore_index=True)
    stacked["hour"] = pd.to_datetime(stacked["timestamp"]).dt.hour
    stacked["regime"] = np.where(
        (stacked["hour"] >= 6) & (stacked["hour"] < 18),
        "diurno (06-17h)", "noturno (18-05h)")
    stacked["banda"] = pd.cut(stacked["actual_raw"],
                              bins=[b[0] for b in BANDS] + [np.inf],
                              labels=BAND_LABELS, right=False)
    stacked["ae"] = (stacked["actual_raw"] - stacked["predicted"]).abs()
    stacked["ae_p"] = (stacked["actual_raw"] - stacked["persistence"]).abs()
    stacked["se"] = (stacked["actual_raw"] - stacked["predicted"]) ** 2
    stacked["se_p"] = (stacked["actual_raw"] - stacked["persistence"]) ** 2
    out = {}
    for col in ("regime", "banda"):
        g = (stacked.groupby(["modelo", col], observed=True)
             .agg(n=("ae", "size"),
                  MAE=("ae", "mean"), MAE_persist=("ae_p", "mean"),
                  MSE=("se", "mean"), MSE_persist=("se_p", "mean"))
             .reset_index())
        g["ratio_MAE"] = g["MAE"] / g["MAE_persist"]
        g["skill"] = 1.0 - g["MSE"] / g["MSE_persist"]
        out[col] = g
    return out


def wavelet_floor(dataset, val_end, level=2):
    ws = dataset["ws100"].to_numpy()
    w = wavelet_denoising(ws, wavelet="sym18", level=level)[:len(ws)]
    raw, den = ws[val_end:], w[val_end:]
    return (float(np.mean(np.abs(raw - den))),
            float(np.sqrt(np.mean((raw - den) ** 2))))


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def fig_norm_bars(norm_df, img_dir):
    d = norm_df[norm_df["modelo"] != "Persistência"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    panels = (("MASE", "MASE (modelo / persistência)", 1.0),
              ("skill", "Skill vs. persistência", 0.0),
              ("ratio_mediano", "|erro| / |erro persist.|\n(mediana por célula)", 1.0))
    for ax, (col, title, ref) in zip(axes, panels):
        y = np.arange(len(d))
        ax.barh(y, d[col], color="#1b6ca8")
        ax.axvline(ref, color="black", ls="--", lw=1.1)
        for yi, v in zip(y, d[col]):
            ax.text(v, yi, f" {v:.2f}", va="center", fontsize=8)
        ax.set_yticks(y, d["modelo"])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
    axes[0].set_xlabel("MASE")
    axes[1].set_xlabel("1 − MSE_modelo/MSE_persist.")
    axes[2].set_xlabel("razão mediana")
    savefig(fig, img_dir / "fig_err_norm.png")


def fig_skill_horizon(skill_df, img_dir):
    fig, ax = plt.subplots(figsize=(13, 4.8))
    for key, g in skill_df.groupby("key", sort=False):
        g = g.sort_values("horizon")
        ax.plot(g["horizon"], g["skill"], lw=1.6,
                label=g["modelo"].iloc[0])
        ax.fill_between(g["horizon"], g["ic_inf"], g["ic_sup"], alpha=0.12)
    ax.axhline(0, color="black", lw=1.0, ls="--")
    ax.set_xlabel("Horizonte (passos de 10 min)")
    ax.set_ylabel("Skill vs. persistência")
    ax.set_title("Erro normalizado pelo intrínseco (persistência) por horizonte "
                 "— IC 95% bootstrap")
    ax.set_xticks(range(1, 37, 2))
    ax.legend(ncol=2, fontsize=8)
    sec = ax.secondary_xaxis("top", functions=(lambda x: x / 6, lambda x: x * 6))
    sec.set_xlabel("Horizonte (horas)")
    savefig(fig, img_dir / "fig_skill_horizon.png")


def fig_dm_variants(dm_df, img_dir):
    d = dm_df.sort_values("p_hln").head(12).copy()
    labels = [f"{r.a} vs {r.b}" for r in d.itertuples()]
    y = np.arange(len(d))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    w = 0.4
    axes[0].barh(y - w / 2, d["dm_hln"], height=w, color="#1b6ca8",
                 label="DM(HLN+HAC)")
    axes[0].barh(y + w / 2, d["dm_nw"], height=w, color="#e08214",
                 label="DM(NW auto)")
    axes[0].axvline(0, color="black", lw=0.9)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Estatística DM (positivo = A pior)")
    axes[0].set_title("DM: HLN+HAC × NW automático")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].barh(y - w / 2, -np.log10(np.clip(d["p_hln"], 1e-16, 1)), height=w,
                 color="#1b6ca8", label="p (HLN)")
    axes[1].barh(y + w / 2, -np.log10(np.clip(d["p_holm"], 1e-16, 1)), height=w,
                 color="#7b3294", label="p (Holm)")
    axes[1].axvline(-np.log10(ALPHA), color="red", ls="--", lw=1.1,
                    label=f"α={ALPHA}")
    axes[1].set_yticks(y, ["" for _ in labels])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("−log₁₀(p)")
    axes[1].set_title("Significância: bruta × ajustada por multiplicidade")
    axes[1].legend(loc="lower right", fontsize=8)
    savefig(fig, img_dir / "fig_dm_variants.png")


def fig_strata(strata, img_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, col, title in ((axes[0], "regime", "Regime diurno × noturno"),
                           (axes[1], "banda", "Faixa de intensidade")):
        g = strata[col]
        regimes = list(g[col].unique())
        modelos = list(g["modelo"].unique())
        w = 0.8 / max(len(modelos), 1)
        x = np.arange(len(regimes))
        for i, modelo in enumerate(modelos):
            sub = g[g["modelo"] == modelo].set_index(col).reindex(regimes)
            ax.bar(x + (i - (len(modelos) - 1) / 2) * w, sub["ratio_MAE"],
                   width=w, label=modelo)
        ax.axhline(1.0, color="black", ls="--", lw=1.1)
        ax.set_xticks(x, [str(r) for r in regimes], fontsize=8)
        ax.set_ylabel("MAE_modelo / MAE_persistência")
        ax.set_title(f"Erro normalizado por {title}")
    axes[1].legend(fontsize=7.5, ncol=2)
    savefig(fig, img_dir / "fig_strata_normalized.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", type=Path,
                        default=PROJECT_ROOT / "pipeline" / "tmp" / "trained_8020_itildeq")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "results" / "metodos_itildeq")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = args.out_dir / "images"
    img_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    rr.MODELS_DIR = args.models_dir
    dataset = common.load_dataset(rr.DATA_FILE, keep_raw=("ws40", "ws100", "v40"))
    print(f"Dataset: {len(dataset)} linhas")

    results, extras = rr.evaluate_models(dataset)
    base_pred = rr.load_baseline_predictions()
    matrices, shorts, idx = build_matrices(results, extras, base_pred)
    print(f"Origens comuns: {len(idx)}")

    n_rows = len(dataset)
    train_end = int(n_rows * rr.SPLIT_RATIOS["train_ratio"])
    val_end = train_end + int(n_rows * rr.SPLIT_RATIOS["val_ratio"])

    dm_df, cw_df = dm_variants(matrices, shorts)
    norm_df = normalized_metrics(matrices, shorts)
    skill_df = skill_by_horizon(matrices, shorts, rng)
    strata = strata_normalized(results, extras, shorts)
    floor_mae, floor_rmse = wavelet_floor(dataset, val_end)

    fig_norm_bars(norm_df, img_dir)
    fig_skill_horizon(skill_df, img_dir)
    fig_dm_variants(dm_df, img_dir)
    fig_strata(strata, img_dir)

    norm_df.to_csv(args.out_dir / "erro_normalizado.csv", index=False)
    dm_df.to_csv(args.out_dir / "dm_ajustado.csv", index=False)
    cw_df.to_csv(args.out_dir / "clark_west.csv", index=False)
    skill_df.to_csv(args.out_dir / "skill_por_horizonte.csv", index=False)

    payload = {
        "n_origins": int(len(idx)),
        "alpha": ALPHA, "seed": SEED,
        "floor_mae": floor_mae, "floor_rmse": floor_rmse,
        "dm": dm_df.to_dict("records"),
        "clark_west": cw_df.to_dict("records"),
        "norm": norm_df.to_dict("records"),
        "skill": skill_df.to_dict("records"),
        "strata": {k: v.to_dict("records") for k, v in strata.items()},
    }
    common.write_json(args.out_dir / "metodos_itildeq.json", payload)
    print(f"OK -> {args.out_dir}/metodos_itildeq.json")


if __name__ == "__main__":
    main()
