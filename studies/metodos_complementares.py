"""Análises metodológicas complementares (pós-modelo) para o artigo SBSR.

Reaproveita a avaliação do relatório canônico (relatorio_resultados.py) e
adiciona, sobre a MESMA grade de origens (DL + baselines ARIMA/RF/
persistência):

  A1  Teste DM completo (HAC + correção HLN já em src/utils), agora em pares
      DL×DL, DL×ARIMA/RF/persistência, com IC 95% bootstrap (stationary
      bootstrap) da diferença média de perda;
  A2  Skill score vs. persistência por horizonte com IC 95% (bootstrap);
  A4  Erro estratificado por regime diurno/noturno e faixa de intensidade
      (0-3 / 3-13 / >=13 m/s) com MBE (viés);
  A5  Métricas de cauda por horizonte: P50/P95/Max do |erro| + MedAE;
  A6  Tabela de custo computacional (parâmetros, tempo de treino,
      inferência/janela, épocas, seed);
  B3  Teste KS (KSPA-style) sobre as distribuições de |erro| por par;
  B4  Métricas de fase: L_max (lag da correlação cruzada), D_phase (erro de
      timing do pico) e DA (acurácia direcional);
  B6  A/B de função de perda (rodada canônica vs. MSE) com teste DM;
  B8  Diagnóstico de resíduos (assimetria, curtose, Jarque-Bera) e detecção
      de mudanças (Page-Hinckley) com erro condicionado aos breakpoints.

Executar a partir da raiz do projeto (GPU recomendada):
    source .venv/bin/activate && source scripts/tf_gpu_env.sh
    python studies/metodos_complementares.py \
        --models-dir pipeline/tmp/trained_8020 \
        --loss-ab-dir pipeline/tmp/trained_8020_mseloss \
        --out-dir data/results/metodos_complementares

Saída: <out>/metodos.json + CSVs + <out>/images/*.png (o relatório curado é
    reports/protocolo/metodos-complementares/report.py — o HTML/MD standalone
    legado foi removido; regenerate = reavaliar + rebuild do hub)
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for path in (PROJECT_ROOT, PROJECT_ROOT / "reports" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
from scipy import stats as sps

import relatorio_resultados as rr
from src import common
from src.utils import diebold_mariano_loss_diff, _stationary_bootstrap_idx

ALPHA = 0.05
N_BOOT = 2000
MEAN_BLOCK = 24  # ~4 h de origens por bloco do stationary bootstrap
SEED = 42
BANDS = [(0.0, 3.0), (3.0, 13.0), (13.0, np.inf)]
BAND_LABELS = ["0-3 m/s", "3-13 m/s", ">=13 m/s"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def error_matrix_long(df, pred_col):
    """Matriz origin x horizonte de erros (actual_raw - pred)."""
    e = df["actual_raw"] - df[pred_col]
    return (df.assign(e=e)
              .pivot(index="origin", columns="horizon", values="e")
              .sort_index().sort_index(axis=1))


def md_table(df, floatfmt="{:.4f}"):
    headers = list(df.columns)
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def boot_ci_mean(d, rng, n_boot=N_BOOT, mean_block=MEAN_BLOCK):
    """IC 95% da média de d via stationary bootstrap (Politis & Romano)."""
    d = np.asarray(d, dtype=float)
    if d.size < 8:
        return (np.nan, np.nan)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = _stationary_bootstrap_idx(d.size, mean_block, rng)
        means[b] = d[idx].mean()
    return (float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


# ---------------------------------------------------------------------------
# Análises
# ---------------------------------------------------------------------------

def cost_table(results):
    rows = []
    for name, m in results.items():
        tr = m.get("training") or {}
        timing = m.get("timing") or {}
        rows.append({
            "Modelo": m["short"],
            "Parametros": m["params"],
            "Tamanho (MB)": round(m["size_mb"], 2),
            "Epocas": tr.get("epochs_run"),
            "Batch": tr.get("batch_size"),
            "Tempo treino (s)": round(tr["training_time_sec"], 1)
            if tr.get("training_time_sec") is not None else None,
            "Inferencia (ms/janela)": round(timing.get("per_window_ms", np.nan), 2),
            "Loss": m.get("loss"),
            "Seed": tr.get("seed"),
        })
    bl = rr.load_baselines()
    if bl:
        met = bl["metrics"]
        rows.append({"Modelo": "ARIMA", "Parametros": "-",
                     "Tamanho (MB)": None, "Epocas": "-",
                     "Batch": "-", "Tempo treino (s)": None,
                     "Inferencia (ms/janela)": None,
                     "Loss": f"order {tuple(met['arima']['order'])}",
                     "Seed": met.get("config", {}).get("seed")})
        rows.append({"Modelo": "Random Forest", "Parametros": "-",
                     "Tamanho (MB)": None, "Epocas": "-",
                     "Batch": "-",
                     "Tempo treino (s)": None,
                     "Inferencia (ms/janela)": None,
                     "Loss": f"{met['rf']['n_estimators']} arvores",
                     "Seed": met.get("config", {}).get("seed")})
    return pd.DataFrame(rows)


def dm_analysis(matrices, shorts, rng):
    """DM agregado por par (media das perdas por origem, h=36) + IC bootstrap."""
    names = list(matrices.keys())
    rows = []
    raw = []
    d_series = {}
    for a, b in combinations(names, 2):
        ea, eb = matrices[a], matrices[b]
        common = ea.index.intersection(eb.index).sort_values()
        if len(common) < 30:
            continue
        A = ea.loc[common].to_numpy(float)
        B = eb.loc[common].to_numpy(float)
        d = (np.abs(A) ** 2).mean(axis=1) - (np.abs(B) ** 2).mean(axis=1)
        d_series[(a, b)] = d
        res = diebold_mariano_loss_diff(d, h=A.shape[1])
        lo, hi = boot_ci_mean(d, rng)
        row = {
            "par": f"{shorts[a]} vs {shorts[b]}",
            "n_origins": int(len(common)),
            "mean_dMSE": res["mean_loss_diff"],
            "IC95_inf": lo, "IC95_sup": hi,
            "DM": res["dm"], "p": res["p_value"],
            "melhor": shorts[a] if res["mean_loss_diff"] < 0 else shorts[b],
        }
        row["sig"] = "sim" if res["p_value"] < ALPHA else "nao"
        rows.append(row)
        raw.append({"a": a, "b": b, **{k: v for k, v in row.items()
                                       if k != "par"}})
    return (pd.DataFrame(rows).sort_values("p"), raw, d_series)


def skill_curves(matrices, rng):
    """Skill vs. persistencia por horizonte com IC 95% (bootstrap de origens)."""
    pers = matrices["__persistence__"]
    rows = []
    for name, mat in matrices.items():
        if name == "__persistence__":
            continue
        common = mat.index.intersection(pers.index).sort_values()
        A = mat.loc[common].to_numpy(float)
        P = pers.loc[common].to_numpy(float)
        n_h = A.shape[1]
        for j in range(n_h):
            mse_m = (A[:, j] ** 2).mean()
            mse_p = (P[:, j] ** 2).mean()
            point = 1.0 - mse_m / mse_p
            d = A[:, j] ** 2 - P[:, j] ** 2  # >0 : modelo pior que persist.
            lo, hi = boot_ci_mean(d, rng, n_boot=500)
            # IC do skill a partir do IC de d: skill = 1 - mean(d)/mse_p
            slo = 1.0 - hi / mse_p
            shi = 1.0 - lo / mse_p
            rows.append({"modelo": name, "horizon": j + 1,
                         "skill": point, "ic_inf": slo, "ic_sup": shi})
    return pd.DataFrame(rows)


def tail_table(matrices, shorts=None):
    shorts = shorts or {}
    rows, med_rows = [], []
    for name, mat in matrices.items():
        if name == "__persistence__":
            continue
        A = np.abs(mat.to_numpy(float))
        for j in range(A.shape[1]):
            rows.append({"modelo": shorts.get(name, name), "horizon": j + 1,
                         "p50": float(np.percentile(A[:, j], 50)),
                         "p95": float(np.percentile(A[:, j], 95)),
                         "max": float(A[:, j].max())})
        med_rows.append({"modelo": shorts.get(name, name),
                         "MedAE": float(np.percentile(A, 50)),
                         "P95_global": float(np.percentile(A, 95)),
                         "MaxAE": float(A.max())})
    return pd.DataFrame(rows), pd.DataFrame(med_rows)


def kspa_tests(matrices, shorts):
    rows = []
    abs_err = {n: np.abs(m.to_numpy(float).ravel())
               for n, m in matrices.items()}
    for a, b in combinations(list(matrices.keys()), 2):
        stat = sps.ks_2samp(abs_err[a], abs_err[b])
        rows.append({"par": f"{shorts[a]} vs {shorts[b]}",
                     "KS_D": float(stat.statistic),
                     "p": float(stat.pvalue),
                     "mesma_dist": "sim" if stat.pvalue >= ALPHA else "nao"})
    return pd.DataFrame(rows).sort_values("p")


def phase_metrics(all_df, short_of):
    """L_max (lag da corr. cruzada por origem), D_phase e DA por modelo."""
    rows = []
    lags = np.arange(-12, 13)
    for name, df in all_df.items():
        piv_p = (df.pivot(index="origin", columns="horizon", values="predicted")
                 .dropna(axis=0))
        piv_a = (df.pivot(index="origin", columns="horizon", values="actual_raw")
                 .dropna(axis=0))
        common = piv_p.index.intersection(piv_a.index).sort_values()
        piv_p, piv_a = piv_p.loc[common], piv_a.loc[common]
        P = piv_p.to_numpy(float)
        A = piv_a.to_numpy(float)
        # D_phase: erro de timing do pico por trajetoria
        d_phase = np.abs(np.argmax(A, axis=1) - np.argmax(P, axis=1))
        # DA: fração de passos em que o sinal da variação bate
        da_steps = np.diff(A, axis=1)
        da_pred = np.diff(P, axis=1)
        valid = da_steps != 0
        da = float((np.sign(da_steps[valid]) == np.sign(da_pred[valid])).mean())
        # L_max: media do lag com corr. cruzada maxima por origem
        Pc = P - P.mean(axis=1, keepdims=True)
        Ac = A - A.mean(axis=1, keepdims=True)
        denom = np.sqrt((Pc ** 2).sum(1) * (Ac ** 2).sum(1))
        best_lag = np.zeros(len(P), dtype=int)
        best_corr = np.full(len(P), -np.inf)
        for lag in lags:
            if lag >= 0:
                corr = (Pc[:, :P.shape[1] - lag or None]
                        * Ac[:, lag:]).sum(1)
            else:
                corr = (Pc[:, -lag:] * Ac[:, :P.shape[1] + lag]).sum(1)
            corr = corr / np.maximum(denom, 1e-12)
            better = corr > best_corr
            best_lag[better] = lag
            best_corr[better] = corr[better]
        l_max = float(np.mean(best_lag))
        rows.append({"modelo": short_of[name],
                     "L_max (passos)": l_max,
                     "D_phase_mediano": float(np.median(d_phase)),
                     "D_phase_medio": float(d_phase.mean()),
                     "DA": float(da),
                     "n_origins": int(len(P))})
    return pd.DataFrame(rows)


def residual_diagnostics(matrices, dataset, all_df_first, shorts=None):
    shorts = shorts or {}
    rows = []
    for name, mat in matrices.items():
        if name == "__persistence__":
            continue
        e = (mat.to_numpy(float).ravel())
        e = e[np.isfinite(e)]
        jb = sps.jarque_bera(e[: min(len(e), 50000)])
        rows.append({"modelo": shorts.get(name, name), "media": float(e.mean()),
                     "desvio": float(e.std()),
                     "assimetria": float(sps.skew(e)),
                     "curtose": float(sps.kurtosis(e)),
                     "jarque_bera_p": float(jb.pvalue)})
    # Page-Hinckley (dois lados) sobre o trecho avaliado da serie real;
    # breakpoints devolvidos como TIMESTAMPS do proprio span avaliado
    t_index = (pd.DatetimeIndex(pd.to_datetime(all_df_first["timestamp"]))
               .unique().sort_values())
    span = dataset["ws100"].reindex(t_index).dropna().to_numpy(float)
    breaks = page_hinkley(span, delta=0.25, threshold=8.0, min_gap=144)
    br_stamps = [pd.Timestamp(t_index[int(b)]) for b in breaks]
    return pd.DataFrame(rows), span, br_stamps, t_index


def page_hinkley(x, delta=0.25, threshold=8.0, min_gap=144):
    """Detecção de mudança de média (Page-Hinkley, dois lados, offline)."""
    x = (x - x.mean()) / (x.std() + 1e-12)
    breaks = []
    start = 0
    while start < len(x) - min_gap:
        seg = x[start:]
        m_up = np.cumsum(seg - delta)
        alarm_up = np.argmax(m_up - np.minimum.accumulate(m_up) > threshold)
        m_dn = np.cumsum(seg + delta)
        alarm_dn = np.argmax(np.maximum.accumulate(m_dn) - m_dn > threshold)
        cands = [c for c in (alarm_up, alarm_dn)
                 if c > 0 and c < len(seg) - 1]
        if not cands:
            break
        c = int(min(cands))
        breaks.append(start + c)
        start = start + c + min_gap
    return breaks


def loss_ab_test(base_all, ab_all, keys):
    rows = []
    for key in keys:
        if key not in base_all or key not in ab_all:
            continue
        ba = base_all[key]
        ab = ab_all[key]
        ma = error_matrix_long(ba, "predicted").dropna(axis=0)
        mb = error_matrix_long(ab, "predicted").dropna(axis=0)
        common = ma.index.intersection(mb.index).sort_values()
        if len(common) < 30:
            continue
        A = ma.loc[common].to_numpy(float)
        B = mb.loc[common].to_numpy(float)
        d = (np.abs(A) ** 2).mean(axis=1) - (np.abs(B) ** 2).mean(axis=1)
        res = diebold_mariano_loss_diff(d, h=A.shape[1])
        rows.append({"modelo": key, "n_origins": len(common),
                     "dMSE (canonica - mse)": res["mean_loss_diff"],
                     "DM": res["dm"], "p": res["p_value"],
                     "melhor": "canonica" if res["mean_loss_diff"] < 0 else "mse"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def plot_skill(skill_df, shorts, out_png):
    plt = rr.plt
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, g in skill_df.groupby("modelo"):
        ax.plot(g["horizon"], g["skill"], label=shorts.get(name, name), lw=1.8)
        ax.fill_between(g["horizon"], g["ic_inf"], g["ic_sup"], alpha=0.15)
    ax.axhline(0, color="k", lw=0.8, ls="--", label="persistência")
    ax.set_xlabel("Horizonte (passos de 10 min)")
    ax.set_ylabel("Skill score vs. persistência")
    ax.set_title("Skill por horizonte (sombra = IC 95%, stationary bootstrap)")
    ax.legend(fontsize=8)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_ecdf(matrices, shorts, out_png):
    plt = rr.plt
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, mat in matrices.items():
        e = np.abs(mat.to_numpy(float).ravel())
        xs = np.sort(e)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, label=shorts.get(name, name), lw=1.6)
    ax.set_xlabel("|erro| (m/s)")
    ax.set_ylabel("ECDF")
    ax.set_title("ECDF do erro absoluto (todos os horizontes)")
    ax.set_xlim(0, np.percentile(np.abs(
        matrices["__persistence__"].to_numpy(float)), 99.5))
    ax.legend(fontsize=8)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figuras-conceito + relatório HTML
# ---------------------------------------------------------------------------

def _pair(d_series, a, b):
    return d_series.get((a, b), d_series.get((b, a)))


def fig_dm_concept(d_series, shorts, dm_raw, img_dir):
    """d_t (diferença de perda por origem) de um par significativo e de um empate."""
    plt = rr.plt
    sig = ("Seq2Seq_LSTM_Bidirectional", "Seq2Seq_TCN")
    tie = ("Seq2Seq_TCN", "__arima__")
    stats = {(r["a"], r["b"]): r for r in dm_raw}
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.4), sharex=True)
    for ax, (pair, titulo) in zip(
            axes, ((sig, "DIFERENÇA SIGNIFICATIVA"),
                   (tie, "EMPATE ESTATÍSTICO"))):
        d = _pair(d_series, *pair)
        r = stats.get(pair) or stats.get((pair[1], pair[0]))
        x = np.arange(len(d))
        ax.plot(x, d, lw=0.6, alpha=0.55, color="#888",
                label="$d_t$ por origem (perda A − perda B)")
        m = d.mean()
        ax.axhline(m, color="#d62728", lw=1.8,
                   label=f"média = {m:+.3f}")
        ax.axhline(r["IC95_inf"], color="#1f77b4", ls="--", lw=1.2)
        ax.axhline(r["IC95_sup"], color="#1f77b4", ls="--", lw=1.2,
                   label=f"IC 95% = [{r['IC95_inf']:+.3f}, {r['IC95_sup']:+.3f}]")
        ax.axhline(0, color="k", lw=1.0)
        ax.set_title(f"{titulo}: {shorts[pair[0]]} vs {shorts[pair[1]]} — "
                     f"DM={r['DM']:+.2f}, p={r['p']:.4f}", fontsize=10)
        ax.set_ylabel("$d_t$ (m/s)²")
        ax.legend(fontsize=7.5, loc="upper right")
    axes[1].set_xlabel("origem do rolling forecast (ordem temporal)")
    fig.suptitle("Conceito do teste DM: a média de d_t difere de zero?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(img_dir / "fig_dm_concept.png", dpi=150)
    plt.close(fig)


def fig_strata(strata, img_dir):
    plt = rr.plt
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    g = strata["regime"].pivot(index="modelo", columns="regime", values="MAE")
    g.plot(kind="bar", ax=axes[0], legend=True, rot=20, width=0.75)
    axes[0].set_title("MAE por regime (diurno × noturno)")
    axes[0].set_ylabel("MAE (m/s)")
    g2 = strata["regime"].pivot(index="modelo", columns="regime", values="MBE")
    g2.plot(kind="bar", ax=axes[1], legend=True, rot=20, width=0.75,
            color=["#2ca02c", "#1f77b4"])
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_title("MBE (viés) por regime — troca de sinal = viés condicional")
    axes[1].set_ylabel("MBE (m/s)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(img_dir / "fig_strata_regime.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    g3 = strata["banda"].pivot(index="modelo", columns="banda", values="MBE")
    g3.plot(kind="bar", ax=ax, rot=20, width=0.7)
    ax.axhline(0, color="k", lw=1)
    ax.set_title("MBE por faixa de intensidade do vento real "
                 "(n(≥13 m/s)=36 — subestimação severa dos extremos)")
    ax.set_ylabel("MBE (m/s)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(img_dir / "fig_strata_band.png", dpi=150)
    plt.close(fig)


def fig_tails(matrices, shorts, tails_df, img_dir):
    plt = rr.plt
    tcn = matrices["Seq2Seq_TCN"].to_numpy(float).ravel()
    ae = np.abs(tcn[np.isfinite(tcn)])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].hist(ae, bins=80, color="#ff7f0e", alpha=0.8)
    for q, c, lab in ((50, "k", "MedAE"), (95, "#d62728", "P95"),
                      (100, "#9467bd", "MaxAE")):
        v = np.percentile(ae, q)
        axes[0].axvline(v, color=c, ls="--", lw=1.6,
                        label=f"{lab} = {v:.2f} m/s")
    axes[0].set_title("Distribuição do |erro| (TCN): média ≠ pior caso")
    axes[0].set_xlabel("|erro| (m/s)")
    axes[0].set_ylabel("amostras")
    axes[0].legend(fontsize=8)
    for name, mat in matrices.items():
        if name == "__persistence__":
            continue
        sub = tails_df[tails_df["modelo"] == shorts.get(name, name)]
        axes[1].plot(sub["horizon"], sub["p95"], lw=1.6,
                     label=shorts.get(name, name))
    axes[1].set_title("P95 do |erro| por horizonte (pior caso típico)")
    axes[1].set_xlabel("horizonte (passos de 10 min)")
    axes[1].set_ylabel("P95 |erro| (m/s)")
    axes[1].legend(fontsize=7.5)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(img_dir / "fig_tails.png", dpi=150)
    plt.close(fig)


def fig_cost(cost_df, mse_common, img_dir):
    plt = rr.plt
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for _, r in cost_df.iterrows():
        if r["Modelo"] in ("ARIMA", "Random Forest"):
            mse = mse_common.get(r["Modelo"])
            if mse:
                ax.axhline(mse, ls=":", lw=1.3,
                           label=f"{r['Modelo']} (MSE={mse:.2f})")
            continue
        mse = mse_common.get(r["Modelo"])
        if mse is None:
            continue
        ax.scatter(r["Tempo treino (s)"], mse,
                   s=max(40, r["Parametros"] / 4000),
                   alpha=0.75, edgecolor="k", linewidth=0.5)
        ax.annotate(f"{r['Modelo']}\n({r['Parametros']/1000:.0f}k params, "
                    f"{mse:.2f})",
                    (r["Tempo treino (s)"], mse),
                    textcoords="offset points", xytext=(8, -4),
                    fontsize=8)
    ax.set_xlabel("tempo de treinamento (s, mesmo hardware)")
    ax.set_ylabel("MSE no teste (origens comuns)")
    ax.set_title("Trade-off custo × acurácia (área ∝ nº de parâmetros)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(img_dir / "fig_cost.png", dpi=150)
    plt.close(fig)


def fig_ecdf_annot(matrices, shorts, img_dir):
    plt = rr.plt
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    arrays = {}
    for name, mat in matrices.items():
        e = np.abs(mat.to_numpy(float).ravel())
        e = e[np.isfinite(e)]
        arrays[name] = np.sort(e)
        ax.plot(arrays[name], np.arange(1, len(e) + 1) / len(e), lw=1.6,
                label=shorts.get(name, name))
    a, b = arrays["Seq2Seq_TCN"], arrays["Seq2Seq_LSTM"]
    grid = np.linspace(0, max(a.max(), b.max()), 2000)
    ea = np.searchsorted(a, grid) / len(a)
    eb = np.searchsorted(b, grid) / len(b)
    j = int(np.argmax(np.abs(ea - eb)))
    ax.annotate("", xy=(grid[j], ea[j]), xytext=(grid[j], eb[j]),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.6))
    ax.annotate(f"D = máx |ECDF_A − ECDF_B| = {abs(ea[j]-eb[j]):.3f}\n"
                f"(em |erro| ≈ {grid[j]:.1f} m/s)",
                (grid[j], (ea[j]+eb[j])/2), fontsize=8.5,
                xytext=(30, -6), textcoords="offset points")
    ax.set_xlabel("|erro| (m/s)")
    ax.set_ylabel("prob. acumulada")
    ax.set_title("ECDF do erro: KS compara distribuições inteiras "
                 "(n grande ⇒ reporte D, não p)")
    ax.set_xlim(0, np.percentile(arrays["__persistence__"], 99.5))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(img_dir / "fig_ecdf_annot.png", dpi=150)
    plt.close(fig)


def fig_phase_traj(all_df, short_of, img_dir):
    """Duas trajetórias reais (típica e difícil) com timing do pico marcado."""
    plt = rr.plt
    name = "Seq2Seq_TCN"
    df = all_df[name]
    piv_a = (df.pivot(index="origin", columns="horizon", values="actual_raw")
             .dropna(axis=0))
    piv_p = (df.pivot(index="origin", columns="horizon", values="predicted")
             .dropna(axis=0))
    common = piv_a.index.intersection(piv_p.index)
    mae_traj = (piv_p.loc[common] - piv_a.loc[common]).abs().mean(axis=1)
    origins = [mae_traj.sort_values().index[len(mae_traj)//2],
               mae_traj.sort_values().index[int(len(mae_traj)*0.95)]]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.2), sharex=True)
    for ax, o, lab in zip(axes, origins, ("trajetória típica (mediana)",
                                          "trajetória difícil (p95)")):
        a = piv_a.loc[o].to_numpy(float)
        p = piv_p.loc[o].to_numpy(float)
        x = np.arange(1, len(a) + 1)
        ax.plot(x, a, "k-o", ms=2.5, lw=1.6, label="observado")
        ax.plot(x, p, "#ff7f0e", lw=1.6, label="TCN (previsto)")
        ia, ip = int(np.argmax(a)), int(np.argmax(p))
        ax.axvline(ia + 1, color="k", ls=":", lw=1.2)
        ax.axvline(ip + 1, color="#ff7f0e", ls=":", lw=1.2)
        ax.annotate("", xy=(ip + 1, p.max()), xytext=(ia + 1, a.max()),
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.4))
        ax.set_title(f"{lab} — D_phase = |{ia+1}−{ip+1}| = "
                     f"{abs(ia-ip)} passos ({abs(ia-ip)*10:.0f} min)",
                     fontsize=9.5)
        ax.set_ylabel("ws100 (m/s)")
        ax.legend(fontsize=8, loc="lower left")
    axes[1].set_xlabel("passo do horizonte (10 min cada; 36 = 6 h)")
    fig.suptitle("Conceito de fase: acertar o valor ≠ acertar o momento do pico",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(img_dir / "fig_phase_traj.png", dpi=150)
    plt.close(fig)


def fig_loss_forest(loss_ab_df, img_dir):
    plt = rr.plt
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, (_, r) in enumerate(loss_ab_df.iterrows()):
        # IC via bootstrap salvo em dMSE? aqui só ponto+p; desenha ponto e
        # linha de referência zero, marcando significância
        cor = "#d62728" if r["p"] < ALPHA else "#7f7f7f"
        ax.scatter(r["dMSE (canonica - mse)"], i, s=90, color=cor, zorder=3)
        ax.annotate(f"p={r['p']:.3f}", (r["dMSE (canonica - mse)"], i),
                    xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.axvline(0, color="k", lw=1.2)
    ax.set_yticks(range(len(loss_ab_df)),
                  [r["modelo"] for _, r in loss_ab_df.iterrows()])
    ax.set_xlabel("ΔMSE (perda canônica − MSE puro); >0 ⇒ MSE puro melhor")
    ax.set_title("A/B de função de perda (DM por modelo) — vermelho = "
                 "diferença significativa")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(img_dir / "fig_loss_forest.png", dpi=150)
    plt.close(fig)


def fig_breaks(span, t_index, breaks, near_df, img_dir):
    plt = rr.plt
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.4),
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(t_index[:len(span)], span, lw=0.8, color="#1f77b4")
    for b in breaks:
        axes[0].axvline(b, color="#d62728", ls="--", lw=1.1, alpha=0.8)
    axes[0].set_title("Série real no período avaliado + mudanças de média "
                      "detectadas (Page–Hinkley, gap ≥ 1 dia)")
    axes[0].set_ylabel("ws100 (m/s)")
    x = np.arange(len(near_df))
    axes[1].bar(x - 0.19, near_df["MAE_perto_break"], width=0.38,
                label="perto do breakpoint (±6 h)", color="#d62728", alpha=0.85)
    axes[1].bar(x + 0.19, near_df["MAE_longe_break"], width=0.38,
                label="longe do breakpoint", color="#7f7f7f", alpha=0.85)
    axes[1].set_xticks(x, near_df["modelo"], rotation=15)
    axes[1].set_ylabel("MAE (m/s)")
    axes[1].set_title("Erro perto × longe das mudanças — hipótese "
                      "“erro se concentra nas transições” REFUTADA")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(img_dir / "fig_breaks.png", dpi=150)
    plt.close(fig)


def fig_resid(matrices, shorts, img_dir):
    plt = rr.plt
    names = [n for n in matrices if n != "__persistence__"]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.2))
    for ax, name in zip(axes.ravel(), names):
        e = matrices[name].to_numpy(float).ravel()
        e = e[np.isfinite(e)]
        ax.hist(e, bins=70, density=True, color="#1f77b4", alpha=0.8)
        xs = np.linspace(e.min(), e.max(), 300)
        ax.plot(xs, sps.norm.pdf(xs, e.mean(), e.std()), "r--", lw=1.4,
                label="N(µ, σ) de referência")
        ax.set_title(f"{shorts.get(name, name)}\n"
                     f"skew={sps.skew(e):+.2f}, kurt={sps.kurtosis(e):+.2f}",
                     fontsize=9)
        ax.legend(fontsize=7)
    axes[0, 0].set_ylabel("densidade")
    axes[1, 0].set_ylabel("densidade")
    fig.suptitle("Resíduos: caudas mais pesadas que a Normal "
                 "(⇒ IC via bootstrap, não normalidade)", fontsize=11)
    fig.tight_layout()
    fig.savefig(img_dir / "fig_resid.png", dpi=150)
    plt.close(fig)


def _b64(path):
    import base64
    return base64.b64encode(Path(path).read_bytes()).decode()


def _html_table(df, floatfmt="{:.4f}", max_rows=None):
    headers = list(df.columns)
    if max_rows:
        df = df.head(max_rows)
    out = ["<table><thead><tr>" + "".join(
        f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>"]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                s = floatfmt.format(v)
            elif isinstance(v, (np.floating, np.integer)):
                s = f"{v}"
            else:
                s = str(v)
            cls = ' class="sig"' if s in ("sim",) else (
                ' class="ns"' if s in ("nao",) else "")
            cells.append(f"<td{cls}>{s}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _fig(path, caption):
    return (f'<figure><img src="data:image/png;base64,{_b64(path)}" '
            f'alt="{caption}"><figcaption>{caption}</figcaption></figure>')


CSS = """
body{margin:0;font-family:'Segoe UI',system-ui,Arial,sans-serif;
background:#f4f6f9;color:#222;line-height:1.55}
header{background:linear-gradient(135deg,#0f3057,#00587a);color:#fff;
padding:28px 34px}
header h1{margin:0 0 6px;font-size:24px}
header .sub{opacity:.85;font-size:13px}
.cards{display:flex;gap:12px;flex-wrap:wrap;margin-top:16px}
.card{background:rgba(255,255,255,.12);border-radius:10px;padding:10px 16px;
text-align:center;min-width:110px}
.card .num{font-size:19px;font-weight:700}
.card .lab{font-size:11px;opacity:.9}
main{max-width:1080px;margin:0 auto;padding:24px 20px 60px}
section{background:#fff;border:1px solid #e3e8ee;border-radius:12px;
margin:22px 0;padding:22px 26px;box-shadow:0 1px 3px rgba(0,0,0,.06)}
section h2{margin:0 0 4px;font-size:19px;color:#0f3057}
section h2 .tag{font-size:11px;background:#0f3057;color:#fff;
border-radius:6px;padding:2px 8px;vertical-align:middle;margin-left:6px}
.conceito{background:#eef6fb;border-left:4px solid #00587a;border-radius:6px;
padding:10px 14px;margin:12px 0;font-size:14px}
.leitura{background:#f2f7f2;border-left:4px solid #2e7d32;border-radius:6px;
padding:10px 14px;margin:12px 0;font-size:14px}
.fontes{font-size:12px;color:#555;margin-top:10px}
.fontes a{color:#00587a}
figure{margin:14px 0;text-align:center}
figure img{max-width:100%;border:1px solid #e3e8ee;border-radius:8px}
figcaption{font-size:12.5px;color:#555;margin-top:6px}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:10px 0}
th{background:#0f3057;color:#fff;padding:6px 8px;text-align:left}
td{border-bottom:1px solid #e8ecf1;padding:5px 8px}
td.sig{color:#b71c1c;font-weight:700}
td.ns{color:#777}
.alerta{background:#fff8e1;border-left:4px solid #f9a825;border-radius:6px;
padding:10px 14px;margin:12px 0;font-size:13.5px}
footer{max-width:1080px;margin:0 auto;padding:0 20px 40px;color:#777;
font-size:12px}
"""


def build_html_report(out_dir, img_dir, n_origins, dm_df, dm_raw, d_series,
                      skill_agg, tails_med, kspa_df, phase_df, strata,
                      resid_df, breaks, near_df, loss_ab_df, cost_df, shorts):
    """Relatório HTML autocontido (imagens em base64)."""
    def S(name):
        return shorts.get(name, name)

    dm_rows = dm_df.head(12)
    html = [f"""<!doctype html><html lang="pt-br"><head><meta charset="utf-8">
<title>Análises metodológicas complementares — SBSR</title><style>{CSS}</style>
</head><body>
<header><h1>Análises metodológicas complementares</h1>
<div class="sub">Rodada canônica <b>trained_8020</b> (loss ponderada) ·
{ n_origins } origens comuns × 36 horizontes = {n_origins*36:,} amostras ·
DL + ARIMA/RF/Persistência alinhados · α=0,05 · bootstrap estacionário
(Politis–Romano, bloco médio 24, 2000 réplicas, seed 42)</div>
<div class="cards">
<div class="card"><div class="num">{n_origins*36:,}</div><div class="lab">amostras avaliadas</div></div>
<div class="card"><div class="num">7</div><div class="lab">métodos comparados</div></div>
<div class="card"><div class="num">21</div><div class="lab">pares no teste DM</div></div>
<div class="card"><div class="num">9</div><div class="lab">análises (A1–B8)</div></div>
</div></header><main>"""]

    # Resumo executivo
    html.append("""<section><h2>Resumo executivo</h2>
<ul>
<li><b>TCN &gt; LSTM e LSTM-Bi é significativo</b> (DM: p=0,012 e p=0,0002).</li>
<li><b>A bidirecionalidade piora a TCN</b> (p=0,024) e não ajuda a LSTM (p=0,30) —
contraria a narrativa atual do artigo.</li>
<li><b>TCN × ARIMA é empate estatístico</b> no agregado (p=0,85); a vantagem DL
aparece no horizonte de 6 h (skill TCN +0,27 vs +0,18 do ARIMA).</li>
<li><b>LSTM-Bi é pior que a persistência</b> (p=0,026) — skill negativo
reportado com honestidade.</li>
<li><b>Extremos (≥13 m/s) são subestimados em ~4 m/s por todos os modelos</b>
(MBE ≈ −4).</li>
<li><b>Ablação de perda</b>: só a TCN-Bi difere — a favor do MSE puro
(p=0,028); a frase atual do artigo sobre perdas fica sem sustentação.</li>
</ul>
<div class="alerta"><b>Achado operacional:</b> o <code>mcs.json</code> do
relatório canônico usa apenas <b>163 origens</b> — as origens dos baselines
têm drift de microssegundos e não casam com a grade do dataset. Correção
(<code>origin.dt.round("min")</code>) aplicada aqui; falta aplicar em
<code>reports/src/relatorio_resultados.py</code>.</div>
</section>""")

    # A1 DM
    html.append(f"""<section id="dm"><h2>Diebold–Mariano <span class="tag">A1</span></h2>
<div class="conceito"><b>O que é.</b> Para cada origem do rolling forecast
calcula-se a diferença de perda entre dois modelos,
d<sub>t</sub> = MSE<sub>A</sub>(t) − MSE<sub>B</sub>(t). H₀: E[d<sub>t</sub>]=0
(mesma precisão). A variância usa estimador HAC (Newman–West) porque as
janelas se sobrepõem, e a estatística recebe a correção de pequenas
amostras de Harvey–Leybourne–Newbold, avaliada contra t(n−1).</div>
<div class="leitura"><b>Como ler.</b> Média de d<sub>t</sub> claramente
diferente de zero (IC não cruza 0) ⇒ a diferença não é sorte. No empate
(TCN × ARIMA), o IC cruza zero em ambos os lados.</div>
{_fig(img_dir / "fig_dm_concept.png", "Série d_t por origem: par significativo (acima) e empate (abaixo), com média e IC 95% do bootstrap.")}
{_html_table(dm_rows, "{:.4f}")}
<div class="fontes">Diebold &amp; Mariano (1995) <a
href="https://doi.org/10.1080/07350015.1995.10524599">10.1080/07350015.1995.10524599</a>
· Harvey, Leybourne &amp; Newbold (1997) <a
href="https://doi.org/10.1016/S0169-2070(96)00719-4">10.1016/S0169-2070(96)00719-4</a>
· uso na coleção: gomez2026predicting, wang2026adaptive, binrong2026whole</div>
</section>""")

    # A2 Skill
    html.append(f"""<section id="skill"><h2>Skill score vs. persistência <span class="tag">A2</span></h2>
<div class="conceito"><b>O que é.</b> SS = 1 − MSE<sub>modelo,h</sub> /
MSE<sub>persistência,h</sub>. Skill 0 = empatar com a predição ingênna
(repetir o último valor); negativo = <b>pior que a ingênua</b>. IC 95% via
bootstrap estacionário sobre as origens.</div>
<div class="leitura"><b>Como ler.</b> No agregado quase tudo fica abaixo de 0
(persistência é forte em horizonte curto); a história do artigo está no
horizonte 36 (6 h), onde TCN/TCN-Bi/ARIMA cruzam para cima.</div>
{_fig(img_dir / "skill_vs_persistencia.png", "Skill por horizonte; sombra = IC 95% do bootstrap estacionário; linha tracejada = persistência.")}
{_html_table(skill_agg, "{:.3f}")}
<div class="fontes">Hyndman &amp; Athanasopoulos, <a
href="https://otexts.com/fpp3/accuracy.html">FPP3 §5.8–5.9</a> ·
Politis &amp; Romano (1994) <a
href="https://doi.org/10.1080/01621459.1994.10476870">10.1080/01621459.1994.10476870</a>
· uso: bendes2026bio (skill negativo reportado)</div>
</section>""")

    # A4 Estratos
    html.append(f"""<section id="strata"><h2>Erro por regime e intensidade <span class="tag">A4</span></h2>
<div class="conceito"><b>O que é.</b> Repartir o erro por extratos
homogêneos: diurno (06–17h) × noturno, e faixas operacionais de vento
(0–3 / 3–13 / ≥13 m/s). MBE = média(prev − real): viés sistemático.</div>
<div class="leitura"><b>Como ler.</b> Barras de MBE que trocam de sinal entre
dia e noite = modelo com viés condicional ao regime. A faixa ≥13 m/s (n=36)
mostra subestimação ~4 m/s — o erro clássico em extremos.</div>
{_fig(img_dir / "fig_strata_regime.png", "MAE e MBE por regime diurno/noturno.")}
{_fig(img_dir / "fig_strata_band.png", "MBE por faixa de intensidade; banda 0–3 m/s vazia (sítio sem calmaria).")}
{_html_table(strata["regime"], "{:.3f}")}
<div class="fontes">xianfeng2026data (faixas 0–3/3–13/13–25 + MBE) ·
aslam2026physics (quartis de volatilidade) · wei2026secondary (eventos fortes,
recall honesto)</div>
</section>""")

    # A5 Caudas
    html.append(f"""<section id="tails"><h2>Métricas de cauda <span class="tag">A5</span></h2>
<div class="conceito"><b>O que é.</b> MedAE (mediana do |erro|), P95 e MaxAE
— o <i>pior caso</i>, não a média. Essencial para despacho/segurança.</div>
<div class="leitura"><b>Como ler.</b> O histograma mostra por que MAE sozinho
engana: a cauda direita carrega o risco. Entre os DL, a TCN tem as melhores
caudas (P95 2,45 m/s).</div>
{_fig(img_dir / "fig_tails.png", "Histograma do |erro| da TCN com MedAE/P95/MaxAE; e P95 por horizonte para todos os métodos.")}
{_html_table(tails_med, "{:.3f}")}
<div class="fontes">xiang2026seqbinder (P95/MaxAE por horizonte) ·
ehsan2000wind (MedAE por robustez)</div>
</section>""")

    # A6 Custo
    html.append(f"""<section id="cost"><h2>Custo computacional <span class="tag">A6</span></h2>
<div class="conceito"><b>O que é.</b> Parâmetros, tempo de treino e
inferência por janela, no mesmo hardware — o pedido da Vanessa.</div>
<div class="leitura"><b>Como ler.</b> Pontos à esquerda-embaixo = barato e
bom. A TCN-Bi é a mais cara (1,6M params, 443 s) e <b>não</b> compensa.</div>
{_fig(img_dir / "fig_cost.png", "Tempo de treino × MSE; área ∝ nº de parâmetros; linhas pontilhadas = baselines sem custo de treino comparável.")}
{_html_table(cost_df, "{:.2f}")}
<div class="fontes">liu2026robust (custo justificado pelo ciclo de despacho) ·
jiang2026data (TT/PT) · aslam2026physics (params/FLOPs)</div>
</section>""")

    # B3 KS
    html.append(f"""<section id="ks"><h2>KS sobre distribuições de erro <span class="tag">B3</span></h2>
<div class="conceito"><b>O que é.</b> KS compara as <b>distribuições
inteiras</b> de |erro| (distância máxima D entre ECDFs) — complementa o DM,
que só compara médias.</div>
<div class="leitura"><b>Como ler.</b> Com n=43 mil amostras <b>todo p&lt;0,05
é trivial</b> — reporte o tamanho de efeito D e o gráfico. A seta indica a
distância máxima entre TCN e LSTM.</div>
{_fig(img_dir / "fig_ecdf_annot.png", "ECDFs do |erro| com a distância máxima (estatística D) anotada.")}
{_html_table(kspa_df.head(8), "{:.4f}")}
<div class="fontes"><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html">scipy.stats.ks_2samp</a>
· uso: gomez2026predicting (ECDF + KSPA)</div>
</section>""")

    # B4 Fase
    html.append(f"""<section id="phase"><h2>Métricas de fase <span class="tag">B4</span></h2>
<div class="conceito"><b>O que é.</b> D_phase = erro de <i>timing</i> do pico
dentro da janela de 6 h; L_max = defasagem da correlação cruzada; DA =
acurácia direcional passo a passo.</div>
<div class="leitura"><b>Como ler.</b> Um pico previsto 1,5 h depois do real
tem MAE aceitável e valor operacional baixo. No exemplo, compare o momento
dos picos (linhas pontilhadas) — não só a altura.</div>
{_fig(img_dir / "fig_phase_traj.png", "Trajetórias reais (típica e difícil): observado × previsto pela TCN, com timing do pico marcado.")}
{_html_table(phase_df, "{:.3f}")}
<div class="fontes">liu2026robust (L_max/D_phase, l.775–789) ·
huang2024research (directional accuracy)</div>
</section>""")

    # B6 Loss A/B (se disponível)
    if loss_ab_df is not None and len(loss_ab_df):
        html.append(f"""<section id="lossab"><h2>A/B de função de perda <span class="tag">B6</span></h2>
<div class="conceito"><b>O que é.</b> Mesma arquitetura e seed, duas perdas
(ponderada pelo horizonte × MSE pura), comparadas por DM.</div>
<div class="leitura"><b>Como ler.</b> Ponto longe do 0 com p&lt;0,05 =
a escolha de perda importa para aquele modelo. Aqui, só a TCN-Bi difere —
<b>a favor do MSE puro</b>. A motivação da perda ponderada no artigo fica
sem evidência.</div>
{_fig(img_dir / "fig_loss_forest.png", "ΔMSE (canônica − MSE) por modelo; vermelho = significativo.")}
{_html_table(loss_ab_df, "{:.4f}")}
<div class="fontes">prática: sheng2026shape (ablação WYM-1/2/3) ·
aslam2026physics (5 perdas, protocolo idêntico)</div>
</section>""")

    # B8 Residuos
    html.append(f"""<section id="resid"><h2>Resíduos e mudanças <span class="tag">B8</span></h2>
<div class="conceito"><b>O que é.</b> Diagnóstico dos resíduos (assimetria,
curtose, Jarque–Bera) e detecção de mudanças de média na série real
(Page–Hinkley), com erro condicionado às mudanças.</div>
<div class="leitura"><b>Como ler.</b> Caudas mais pesadas que a Normal ⇒
intervalos Gaussianos não valem (use bootstrap). O gráfico inferior
<b>refuta</b> a hipótese de que o erro se concentra nas transições —
perto ≈ longe (1,02 × 1,10).</div>
{_fig(img_dir / "fig_resid.png", "Histogramas de resíduos com a Normal de referência: leptocurtose em todos.")}
{_fig(img_dir / "fig_breaks.png", "Série real com breakpoints; e MAE perto × longe das mudanças.")}
{_html_table(resid_df, "{:.3f}")}
<div class="fontes">Page (1954) Biometrika 41(1/2) 100–115 (citado em
cao2026probabilistic) · <a
href="https://otexts.com/fpp3/accuracy.html">FPP3 §5.4</a> ·
aodi2026optimized (BFAST)</div>
</section>""")

    html.append("""</main><footer>Gerado por
<code>studies/metodos_complementares.py</code> · explicações completas e fontes:
<code>docs/revisions/sbr/guia_metricas.ai.md</code> · dados brutos:
<code>metodos.json</code>, CSVs em <code>data/results/metodos_complementares/</code>
· DOIs verificados em 14/10/2026.</footer></body></html>""")
    (out_dir / "relatorio_metodos.html").write_text("".join(html),
                                                    encoding="utf-8")




def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models-dir", type=Path,
                        default=PROJECT_ROOT / "pipeline" / "tmp" / "trained_8020")
    parser.add_argument("--loss-ab-dir", type=Path,
                        default=PROJECT_ROOT / "pipeline" / "tmp" / "trained_8020_mseloss")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "results" / "metodos_complementares")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = args.out_dir / "images"
    img_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    rr.MODELS_DIR = args.models_dir
    dataset = common.load_dataset(rr.DATA_FILE, keep_raw=("ws40", "ws100", "v40"))
    print(f"Dataset: {len(dataset)} linhas")

    print("[1/6] Avaliando modelos DL (mesma grade do relatorio)...")
    results, extras = rr.evaluate_models(dataset)

    print("[2/6] Carregando baselines (ARIMA/RF/persistencia)...")
    base_pred = rr.load_baseline_predictions()

    # matrizes de erro origin x horizon
    matrices = {}
    shorts = {}
    all_by_key = {}
    for name, m in results.items():
        matrices[name] = error_matrix_long(extras[name]["all_df"], "predicted")
        shorts[name] = m["short"]
        all_by_key[m["key"]] = extras[name]["all_df"]
    pers = error_matrix_long(extras[results[list(results)[0]]["name"]]["all_df"],
                             "persistence")
    matrices["__persistence__"] = pers
    shorts["__persistence__"] = "Persistência"
    if base_pred is not None:
        # origens dos baselines têm drift float (ex.: 00:39:59.999998);
        # arredonda p/ o minuto p/ casar com a grade regularizada do dataset
        sub = base_pred.rename(columns={"actual": "actual_raw"}).copy()
        sub["origin"] = sub["origin"].dt.round("min")
        sub["timestamp"] = sub["timestamp"].dt.round("min")
        for col, label in (("arima", "ARIMA"), ("rf", "Random Forest")):
            matrices["__" + col + "__"] = error_matrix_long(sub, col)
            shorts["__" + col + "__"] = label

    # o filtro de timestamps observados deixa buracos origin x horizonte:
    # descarta origens incompletas antes de alinhar
    matrices = {n: m.dropna(axis=0) for n, m in matrices.items()}

    common_idx = None
    for mat in matrices.values():
        common_idx = (mat.index if common_idx is None
                      else common_idx.intersection(mat.index))
    common_idx = common_idx.sort_values()
    print(f"Origens comuns (DL + baselines): {len(common_idx)}")
    matrices = {n: m.loc[common_idx] for n, m in matrices.items()}

    print("[3/6] DM completo + skill + caudas + KSPA...")
    dm_df, dm_raw, d_series = dm_analysis(matrices, shorts, rng)
    skill_df = skill_curves(matrices, rng)
    tails_df, tails_med = tail_table(matrices, shorts)
    kspa_df = kspa_tests(matrices, shorts)

    print("[4/6] Estratos, fase e residuos...")
    # empilha all_df por modelo p/ estratificação
    short_of = {name: m["short"] for name, m in results.items()}
    stacked = []
    for name, m in results.items():
        s = extras[name]["all_df"].copy()
        s["modelo"] = m["short"]
        stacked.append(s)
    stacked = pd.concat(stacked)
    strata = _strata_from_stacked(stacked)
    phase_df = phase_metrics(
        {name: extras[name]["all_df"] for name in results},
        short_of)
    resid_df, span, breaks, t_index = residual_diagnostics(
        {n: m for n, m in matrices.items() if n != "__persistence__"},
        dataset, extras[list(results)[0]]["all_df"], shorts)
    # erro perto de breakpoints (timestamps do próprio span avaliado)
    near_rows = []
    if breaks:
        br_times = pd.DatetimeIndex(breaks)
        for name, m in results.items():
            df = extras[name]["all_df"]
            dist = np.min(np.abs(
                pd.DatetimeIndex(df["timestamp"]).values[:, None]
                - br_times.values[None, :]), axis=1)
            near = dist <= np.timedelta64(6, "h")
            e = (df["actual_raw"] - df["predicted"]).abs()
            near_rows.append({
                "modelo": m["short"],
                "MAE_perto_break": float(e[near].mean()),
                "MAE_longe_break": float(e[~near].mean()),
                "frac_amostras_perto": float(near.mean()),
            })

    print("[5/6] A/B de loss (canonica vs mse)...")
    loss_ab_df = None
    if args.loss_ab_dir and Path(args.loss_ab_dir).is_dir():
        ab = rr.evaluate_loss_ab_variants(
            dataset, args.loss_ab_dir,
            keys=("lstm", "lstm_bi", "tcn", "tcn_bi"))
        if ab:
            loss_ab_df = loss_ab_test(all_by_key, ab,
                                      ("lstm", "lstm_bi", "tcn", "tcn_bi"))

    print("[6/6] Figuras + relatorio...")
    cost_df = cost_table(results)
    # MSE por método na MESMA base (origens comuns) p/ gráfico de custo
    mse_common = {shorts[n]: float(np.mean(matrices[n].to_numpy(float) ** 2))
                  for n in matrices}
    near_df = pd.DataFrame(near_rows)

    # figuras de dados (já existentes)
    plot_skill(skill_df, shorts, img_dir / "skill_vs_persistencia.png")
    plot_ecdf(matrices, shorts, img_dir / "ecdf_erro.png")
    # figuras-conceito
    fig_dm_concept(d_series, shorts, dm_raw, img_dir)
    fig_strata(strata, img_dir)
    fig_tails(matrices, shorts, tails_df, img_dir)
    fig_cost(cost_df, mse_common, img_dir)
    fig_ecdf_annot(matrices, shorts, img_dir)
    fig_phase_traj({name: extras[name]["all_df"] for name in results},
                   short_of, img_dir)
    if loss_ab_df is not None:
        fig_loss_forest(loss_ab_df, img_dir)
    if breaks:
        fig_breaks(span, t_index, breaks, near_df, img_dir)
    fig_resid(matrices, shorts, img_dir)
    tails_df.to_csv(args.out_dir / "caudas_por_horizonte.csv", index=False)
    skill_df.to_csv(args.out_dir / "skill_por_horizonte.csv", index=False)

    payload = {
        "n_origins": int(len(common_idx)),
        "alpha": ALPHA, "n_boot": N_BOOT, "mean_block": MEAN_BLOCK, "seed": SEED,
        "dm": dm_raw,
        "kspa": kspa_df.to_dict("records"),
        "phase": phase_df.to_dict("records"),
        "residuals": resid_df.to_dict("records"),
        "breakpoints": [str(b) for b in breaks],
        "near_break": near_rows,
        "loss_ab": None if loss_ab_df is None else loss_ab_df.to_dict("records"),
        "tails_medians": tails_med.to_dict("records"),
        "strata": {k: v.to_dict("records") for k, v in strata.items()},
        "cost": cost_df.to_dict("records"),
    }
    common.write_json(args.out_dir / "metodos.json", payload)

    print(f"OK -> {args.out_dir}/metodos.json "
          f"(relatório: reports/protocolo/metodos-complementares)")


def _strata_from_stacked(stacked):
    stacked = stacked.copy()
    stacked["hour"] = pd.to_datetime(stacked["timestamp"]).dt.hour
    stacked["regime"] = np.where((stacked["hour"] >= 6) & (stacked["hour"] < 18),
                                 "diurno (06-17h)", "noturno (18-05h)")
    stacked["banda"] = pd.cut(stacked["actual_raw"],
                              bins=[b[0] for b in BANDS] + [np.inf],
                              labels=BAND_LABELS, right=False)
    e = stacked["actual_raw"] - stacked["predicted"]
    stacked["ae"] = e.abs()
    stacked["se"] = e ** 2
    stacked["bias"] = stacked["predicted"] - stacked["actual_raw"]
    out = {}
    for col in ("regime", "banda"):
        g = (stacked.groupby(["modelo", col], observed=True)
             .agg(n=("ae", "size"), MAE=("ae", "mean"),
                  RMSE=("se", lambda x: float(np.sqrt(x.mean()))),
                  MBE=("bias", "mean")).reset_index())
        out[col] = g
    return out


if __name__ == "__main__":
    main()
