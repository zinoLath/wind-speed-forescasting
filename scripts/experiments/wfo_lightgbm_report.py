"""Relatorio HTML do experimento walk-forward com imputacao LightGBM.

Le data/wfo_lightgbm/ (serie, flags, auditoria, validacao por coluna) e
data/results/wfo_lightgbm/<modelo>/ (agregados do walk-forward) e gera um
unico HTML auto-contido (graficos matplotlib embutidos em base64) no hub:
reports/imputacao/wfo-lightgbm/report.html (variantes --results-dir caem em
reports/imputacao/<slug>/report.html, ex.: wfo-lightgbm-2m).

Usage:
    python scripts/experiments/wfo_lightgbm_report.py [--open]
"""

import argparse
import base64
import io
import json
import sys
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import common
import wfo_lightgbm_common as proto

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

MODEL_COLORS = {
    "lstm": "#1f77b4",
    "lstm_bi": "#ff7f0e",
    "tcn": "#2ca02c",
    "tcn_bi": "#d62728",
    "arima": "#7f7f7f",
    "rf": "#9467bd",
}
FAMILY_COLORS = {"ws": "#4c72b0", "v": "#55a868", "dir": "#c44e52"}
ORIGINAL_EVAL_JSON = common.resolve("data/results/report_homologado/evaluation.json")

# Regra de pular origens do rolling forecast: encoder com gap imputado maior
# que 6 h (36 amostras) OU mais de 40% de celulas imputadas. Combinada com
# alvos observados, da o recorte "deploy observado" usado nos graficos.
SKIP_GAP_LIMIT = 36
SKIP_FRAC_LIMIT = 0.4
ARIMA_COLOR = "#7f7f7f"


def max_input_gap_per_origin(frame, row_imputed, input_steps=72):
    """Maior run imputado (amostras) dentro do encoder de cada origem."""
    grid = pd.date_range(
        frame["origin"].min() - pd.Timedelta(minutes=10 * (input_steps - 1)),
        frame["origin"].max(), freq="10min",
    )
    arr = row_imputed.reindex(grid).fillna(False).to_numpy()
    run = np.zeros(len(arr), dtype=int)
    count = 0
    for i in range(len(arr) - 1, -1, -1):
        count = count + 1 if arr[i] else 0
        run[i] = count
    position = {ts: i for i, ts in enumerate(grid)}
    max_gap = {}
    for origin in frame["origin"].drop_duplicates():
        p = position[origin]
        lo = max(0, p - input_steps + 1)
        max_gap[origin] = int(run[lo:p + 1].max())
    return max_gap


def deployment_frame(frame, max_gap):
    """Predicoes 'deploy observado': so alvos observados e origens sem input
    muito imputado (gap <= 6 h e fracao <= 40%)."""
    out = frame[frame["output_imputed"].astype(str) != "True"].copy()
    out = out[out["origin"].map(max_gap) <= SKIP_GAP_LIMIT]
    out = out[out["input_imputed_frac"].astype(float) <= SKIP_FRAC_LIMIT]
    return out


# --- Helpers HTML -------------------------------------------------------------

def fig_to_html(fig, caption=None, max_width="100%"):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    block = f'<img src="data:image/png;base64,{encoded}" style="max-width:{max_width}">'
    if caption:
        block += f'<div class="caption">{caption}</div>'
    return f'<figure>{block}</figure>'


def table_html(df, float_format="%.4f"):
    if df is None or len(df) == 0:
        return "<p><em>Indisponivel.</em></p>"
    return df.to_html(index=False, border=0, float_format=float_format,
                      classes="data")


def section(title, body, anchor):
    return f'<section id="{anchor}"><h2>{title}</h2>\n{body}\n</section>'


# --- Carga de dados -----------------------------------------------------------

def load_inputs():
    data = {"meta": {}, "validation": {}, "wrappers": {}}
    if proto.META_JSON.is_file():
        with open(proto.META_JSON, encoding="utf-8") as handle:
            data["meta"] = json.load(handle)
    summary_path = proto.VALIDATION_DIR / "summary.csv"
    if summary_path.is_file():
        data["validation"]["summary"] = pd.read_csv(summary_path)
    detailed_path = proto.VALIDATION_DIR / "detailed.csv"
    if detailed_path.is_file():
        data["validation"]["detailed"] = pd.read_csv(detailed_path)
    for wrapper_key in proto.WRAPPERS:
        wrapper_name = common.wrapper_factory(wrapper_key)().name
        wrapper_dir = proto.RESULTS_DIR / wrapper_name
        metrics_path = wrapper_dir / "metrics.json"
        predictions_path = wrapper_dir / "predictions_all.csv"
        entry = {"key": wrapper_key, "name": wrapper_name, "dir": wrapper_dir}
        if metrics_path.is_file():
            with open(metrics_path, encoding="utf-8") as handle:
                entry["metrics"] = json.load(handle)
        if predictions_path.is_file():
            entry["predictions"] = pd.read_csv(
                predictions_path, parse_dates=["origin", "timestamp"],
                dtype={"output_imputed": str, "input_imputed_any": str},
            )
        data["wrappers"][wrapper_key] = entry
    data["wrappers"] = {
        key: entry for key, entry in data["wrappers"].items()
        if "metrics" in entry
    }
    # baseline ARIMA (mesmo protocolo, scripts/experiments/wfo_arima_baseline.py)
    arima_metrics = proto.RESULTS_DIR / "arima_baseline" / "metrics.json"
    if arima_metrics.is_file():
        with open(arima_metrics, encoding="utf-8") as handle:
            arima_entry = {"key": "arima", "name": "ARIMA (baseline)",
                           "dir": arima_metrics.parent,
                           "metrics": json.load(handle)}
        arima_predictions = arima_metrics.parent / "predictions_all.csv"
        if arima_predictions.is_file():
            arima_entry["predictions"] = pd.read_csv(
                arima_predictions, parse_dates=["origin", "timestamp"],
                dtype={"output_imputed": str, "input_imputed_any": str},
            )
        if arima_entry.get("predictions") is not None:
            data["wrappers"]["arima"] = arima_entry
    # baseline Random Forest (mesmo protocolo, scripts/experiments/wfo_rf_baseline.py)
    rf_metrics = proto.RESULTS_DIR / "rf_baseline" / "metrics.json"
    if rf_metrics.is_file():
        with open(rf_metrics, encoding="utf-8") as handle:
            rf_entry = {"key": "rf", "name": "RF (baseline)",
                        "dir": rf_metrics.parent,
                        "metrics": json.load(handle)}
        rf_predictions = rf_metrics.parent / "predictions_all.csv"
        if rf_predictions.is_file():
            rf_entry["predictions"] = pd.read_csv(
                rf_predictions, parse_dates=["origin", "timestamp"],
                dtype={"output_imputed": str, "input_imputed_any": str},
            )
        if rf_entry.get("predictions") is not None:
            data["wrappers"]["rf"] = rf_entry
    return data


# --- Secoes -------------------------------------------------------------------

def sec_intro(data):
    meta = data.get("meta", {})
    rows = data.get("rows_info", {})
    items = [
        ("Serie de imputacao", "LiDAR 40-140 m (dataset.csv) + SODAR 40-140 m (wind_data.csv), grade de 10 minutos"),
        ("Periodo", f"{meta.get('period_start', '?')} -> {meta.get('period_end', '?')} "
                    f"({meta.get('rows', '?')} linhas)"),
        ("Imputacao", f"LightGBM (IterativeImputer) com parametros {meta.get('imputation_params', {})}; "
                      "flag booleana por celula"),
        ("Celulas imputadas", f"{meta.get('missing_cells_frac', 0):.2%} do total "
                              f"({meta.get('missing_cells_total', 0):,} celulas)"),
        ("Protocolo", "treino = 3 meses civis anteriores (split interno 80/20), "
                      "teste = 1 mes civil, avanco mes a mes"),
        ("Rolling forecast", f"origens a cada {proto.STRIDE} amostras (1 h), horizonte {proto.OUTPUT_STEPS} "
                             f"passos (6 h), encoder {proto.INPUT_STEPS} passos (12 h)"),
        ("Modelos", "LSTM, LSTM bidirecional, TCN e TCN bidirecional (seq2seq, decoder direct, "
                    "perda MSE ponderada por horizonte)"),
        ("Hiperparametros", "best_trial.json do Optuna (pipeline/tmp/optuna) por modelo"),
        ("Metrica principal", "MSE contra ws100 BRUTO (RMSE/MAE secundarias); "
                              "skill = 1 - MSE_modelo / MSE_persistencia"),
        ("Baseline ARIMA", "ARIMA(p,d,q) escolhido por AIC por janela, retreinado "
                           "mensalmente, mesmas origens e censura dos modelos"
        if "arima" in data.get("wrappers", {}) else
        "ARIMA (nao calculado ainda — scripts/experiments/wfo_arima_baseline.py)"),
        ("Ablação 2 meses", "variação com janela de treino de 2 meses civis "
                            "(lstm_bi, tcn_bi, ARIMA) em "
                            "<code>reports/imputacao/wfo-lightgbm-2m/"
                            "report.html</code>"),
    ]
    items.extend(rows.items())
    body = "<ul>" + "".join(f"<li><strong>{k}:</strong> {v}</li>" for k, v in items) + "</ul>"
    return section("1. Protocolo do experimento", body, "protocolo")


def sec_dados(data):
    meta = data.get("meta", {})
    if not meta:
        return section("2. Dados e imputacao", "<p>series_meta.json ausente — rode "
                       "scripts/experiments/wfo_lightgbm_prepare.py.</p>", "dados")
    series = pd.read_csv(proto.SERIES_CSV, index_col="timestamp", parse_dates=True)
    flags = pd.read_csv(proto.FLAGS_CSV, index_col="timestamp", parse_dates=True).astype(bool)
    row_flag = flags.any(axis=1)

    parts = ["<p>A serie combina o periodo LiDAR (set-nov/2021) com o SODAR "
             "(nov/2021-out/2022) no mesmo ponto de medicao (P1); a troca de instrumento "
             "acontece no gap de 8-10/nov/2021. As faixas vermelhas marcam linhas com "
             "pelo menos uma celula imputada.</p>"]

    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.fill_between(series.index, 0, 1, where=row_flag.to_numpy(),
                    transform=ax.get_xaxis_transform(), color="red", alpha=0.18, linewidth=0)
    for source, color in (("lidar", "#4c72b0"), ("sodar", "#dd8452")):
        if source == "lidar":
            mask = series.index < pd.Timestamp("2021-11-09")
        else:
            mask = series.index >= pd.Timestamp("2021-11-09")
        ax.plot(series.index[mask], series["ws100"][mask], color=color,
                lw=0.4, label=f"ws100 {source}")
    ax.set_ylim(0, max(12, np.nanpercentile(series["ws100"], 99.5) * 1.1))
    ax.set_ylabel("ws100 (m/s)")
    ax.set_title("Serie completa (ws100) com linhas imputadas sombreadas")
    ax.legend(loc="upper right", ncols=2)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    parts.append(fig_to_html(fig))

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.4),
                             gridspec_kw={"width_ratios": [1.4, 1]})
    fracs = flags.mean().sort_values(ascending=False)
    colors = [FAMILY_COLORS.get(col.rstrip("0123456789"), "grey") for col in fracs.index]
    axes[0].barh(fracs.index[::-1], fracs.values[::-1] * 100,
                 color=colors[::-1])
    axes[0].set_xlabel("% de celulas imputadas")
    axes[0].set_title("Fracao imputada por coluna")
    monthly = meta.get("monthly_coverage", {})
    months = sorted(monthly)
    values = [monthly[m]["imputed_frac"] * 100 for m in months]
    bar_colors = ["#4c72b0" if pd.Period(m).end_time < pd.Timestamp("2021-11-09")
                  else "#dd8452" for m in months]
    axes[1].bar(range(len(months)), values, color=bar_colors)
    axes[1].set_xticks(range(len(months)))
    axes[1].set_xticklabels(months, rotation=60, ha="right")
    axes[1].set_ylabel("% linhas imputadas")
    axes[1].set_title("Fracao imputada por mes (azul = LiDAR, laranja = SODAR)")
    fig.tight_layout()
    parts.append(fig_to_html(fig))

    gaps_df = pd.DataFrame(meta.get("largest_all_nan_gaps", []))
    parts.append("<h3>Maiores vazios totais da serie</h3>" + table_html(gaps_df))

    runs = [
        {"coluna": col, "inicio": run["start"], "linhas": run["rows"], "valor": run["value"]}
        for col, run_list in meta.get("constant_runs_imputed", {}).items()
        for run in run_list
    ]
    runs_df = pd.DataFrame(runs).sort_values("linhas", ascending=False).head(10) if runs else None
    parts.append(
        "<h3>Auditoria: trechos imputados constantes</h3>"
        "<p>Imputadores iterativos podem produzir patamares constantes (artefato "
        "documentado em docs/wfo_imputed_detailed_study.md). Trechos longos com valor "
        "identico merecem cautela ao interpretar metricas sobre alvos imputados.</p>"
        + table_html(runs_df, float_format="%.3f")
    )

    # Leques de imputacao: para cada comprimento pedido, o gap real de ws100
    # mais proximo (ex.: 135 quando nao ha 138) com o trecho imputado em
    # destaque contra os dados observados ao redor.
    flag_ws = flags["ws100"].to_numpy()
    index = flags.index
    gaps, run_start = [], None
    for i, value in enumerate(flag_ws):
        if value and run_start is None:
            run_start = i
        elif not value and run_start is not None:
            gaps.append((run_start, i - run_start))
            run_start = None
    if run_start is not None:
        gaps.append((run_start, len(flag_ws) - run_start))

    if gaps:
        requested = [6, 12, 18, 24, 36, 48, 72, 108, 138]
        used, picks = set(), []
        by_length = sorted(range(len(gaps)), key=lambda g: gaps[g][1])
        for target in requested:
            candidates = [g for g in by_length if g not in used]
            if not candidates:
                break
            best = min(candidates, key=lambda g: abs(gaps[g][1] - target))
            used.add(best)
            picks.append((target, best))
        longest = max(by_length, key=lambda g: gaps[g][1])
        if longest not in used:
            picks.append(("max", longest))

        rf_series = None
        rf_path = proto.DATA_DIR / "series_imputed_rf.csv"
        if rf_path.is_file():
            rf_series = pd.read_csv(rf_path, index_col="timestamp", parse_dates=True)
        columns = [(series, "#c44e52", "LightGBM")]
        if rf_series is not None:
            columns.append((rf_series, "#4c72b0", "Random Forest"))

        parts.append("<h3>Exemplos de imputacao por comprimento de gap</h3>"
                     "<p>Para cada comprimento alvo (6 a 138 amostras e o maximo "
                     "da serie) foi selecionado o <strong>gap real de ws100 mais "
                     "proximo</strong>. Cada linha compara dois imputadores no "
                     "MESMO gap: <span style='color:#c44e52'>LightGBM</span> "
                     "(coluna esquerda) e "
                     "<span style='color:#4c72b0'>Random Forest</span> (direita). "
                     "Preto = dado observado; ambar = outras imputacoes de ws100 "
                     "na janela.</p>")
        fig, axes = plt.subplots(len(picks), len(columns),
                                 figsize=(6.6 * len(columns), 2.6 * len(picks)),
                                 squeeze=False)
        for row, (target, gap_i) in enumerate(picks):
            start_i, length = gaps[gap_i]
            context = max(3 * length, 144)
            lo = max(0, start_i - context)
            hi = min(len(index), start_i + length + context)
            window = index[lo:hi]
            imputed_window = flag_ws[lo:hi].copy()
            observed = ~imputed_window
            other = imputed_window.copy()
            other[start_i - lo: start_i - lo + length] = False
            for col, (imp_series, color, name) in enumerate(columns):
                ax = axes[row][col]
                values = imp_series["ws100"].reindex(window).to_numpy()
                ax.plot(window[observed], values[observed], color="black", lw=1.0)
                if other.any():
                    ax.plot(window[other], values[other], color="#dd8452",
                            lw=1.0, alpha=0.85)
                ax.plot(index[start_i: start_i + length],
                        imp_series["ws100"].iloc[start_i: start_i + length],
                        color=color, lw=1.7, marker="o", ms=2.6)
                ax.axvspan(index[start_i], index[start_i + length - 1],
                           color=color, alpha=0.08, lw=0)
                if col == 0:
                    ax.set_title(f"alvo {target} -> gap {length} am. "
                                 f"({length / 6:.1f} h) | "
                                 f"{index[start_i]:%d/%m/%Y %H:%M} — {name}",
                                 fontsize=8.5)
                    ax.set_ylabel("m/s")
                else:
                    ax.set_title(name, fontsize=8.5)
        handles = [
            plt.Line2D([], [], color="black", lw=1.0, label="ws100 observado"),
            plt.Line2D([], [], color="#c44e52", lw=1.7, marker="o", ms=3,
                       label="imputado (LightGBM)"),
            plt.Line2D([], [], color="#4c72b0", lw=1.7, marker="o", ms=3,
                       label="imputado (Random Forest)"),
            plt.Line2D([], [], color="#dd8452", lw=1.0, label="outras imputacoes"),
        ]
        axes[0][0].legend(handles=handles, fontsize=7, loc="upper right")
        fig.tight_layout()
        parts.append(fig_to_html(fig))
    return section("2. Dados e imputacao da serie completa", "\n".join(parts), "dados")


def sec_validacao(data):
    summary = data.get("validation", {}).get("summary")
    detailed = data.get("validation", {}).get("detailed")
    if summary is None or summary.empty:
        return section("3. Validacao da imputacao (erro por coluna)",
                       "<p>imputation_validation/summary.csv ausente.</p>", "validacao")

    parts = ["<p>Erro da imputacao LightGBM medido por injecao de gaps artificiais "
             "(30 gaps por comprimento 1-36 passos, seed 42) num segmento completo da "
             "serie. Direcoes usam erro circular. Este e o <strong>erro por coluna "
             "imputada</strong> pedido para a discussao.</p>"]

    summary = summary.assign(family=summary["col"].str.rstrip("0123456789"))
    by_col = summary.groupby(["col", "family"]).agg(
        mae=("mae", "mean"), rmse=("rmse", "mean"), bias=("bias", "mean"),
        r2=("r2", "mean"), samples=("samples", "sum"),
    ).reset_index().sort_values("mae", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [FAMILY_COLORS.get(f, "grey") for f in by_col["family"]]
    ax.barh(by_col["col"], by_col["mae"], color=colors)
    ax.set_xlabel("MAE medio (todos os comprimentos de gap)")
    ax.set_title("Erro de imputacao por coluna (MAE)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    ax.legend(handles, FAMILY_COLORS.keys(), title="familia")
    parts.append(fig_to_html(fig))

    pivot = summary.pivot(index="col", columns="gap_len", values="mae")
    pivot = pivot.reindex(
        sorted(pivot.index, key=lambda c: (c[: -3], int("".join(filter(str.isdigit, c)))))
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("Comprimento do gap (passos de 10 min)")
    ax.set_ylabel("Coluna")
    ax.set_title("MAE de imputacao: coluna x comprimento do gap")
    ax.grid(False)
    plt.colorbar(image, ax=ax, label="MAE")
    parts.append(fig_to_html(fig))

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for metric, ax, title in (
        ("mae", axes[0, 0], "MAE"),
        ("rmse", axes[0, 1], "RMSE"),
        ("bias", axes[1, 0], "Bias (imputado - real)"),
        ("r2", axes[1, 1], "R2"),
    ):
        for family, color in FAMILY_COLORS.items():
            group = summary[summary["family"] == family].groupby("gap_len")[metric].mean()
            ax.plot(group.index, group.values, marker="o", color=color, label=family)
        ax.set_xlabel("Comprimento do gap")
        ax.set_title(title)
        if metric == "bias":
            ax.axhline(0, color="black", lw=0.8, linestyle="--")
        if metric == "r2":
            ax.set_ylim(min(-0.05, ax.get_ylim()[0]), 1.02)
        ax.legend()
    fig.tight_layout()
    parts.append(fig_to_html(fig))

    family_table = summary.groupby("family").agg(
        mae_media=("mae", "mean"), rmse_medio=("rmse", "mean"),
        r2_medio=("r2", "mean"), amostras=("samples", "sum"),
    ).reset_index()
    parts.append("<h3>Resumo por familia de colunas</h3>" + table_html(family_table))

    if detailed is not None and len(detailed):
        sample = detailed.sample(n=min(60_000, len(detailed)), random_state=0)
        sample = sample.assign(family=sample["col"].astype(str).str.rstrip("0123456789"))
        fig, ax = plt.subplots(figsize=(9, 4))
        # boxplot do matplotlib produz estatisticas NaN com qualquer NaN na
        # serie — descarte explicito antes de plotar
        data_box = [
            sample.loc[sample["family"] == f, "abs_error"].dropna()
            for f in FAMILY_COLORS
        ]
        ax.boxplot(data_box, tick_labels=list(FAMILY_COLORS), showfliers=False)
        ax.set_ylabel("|erro| de imputacao")
        ax.set_title("Distribuicao do erro absoluto por familia (sem outliers)")
        parts.append(fig_to_html(fig))
    return section("3. Validacao da imputacao (erro medido por coluna)",
                   "\n".join(parts), "validacao")


def sec_walkforward(data):
    wrappers = {
        key: entry for key, entry in data.get("wrappers", {}).items()
        if entry.get("predictions") is not None
    }
    if not wrappers:
        return section("4. Walk-forward validation",
                       "<p>Nenhum modelo com metrics.json em "
                       "data/results/wfo_lightgbm/. Rode scripts/experiments/wfo_lightgbm_run.py "
                       "(local) ou colete os kernels do Kaggle.</p>", "wfo")

    # Frames "deploy observado": so alvos observados; origens puladas quando o
    # encoder tem gap imputado > 6 h ou > 40% de celulas imputadas.
    row_imputed = None
    if proto.FLAGS_CSV.is_file():
        flags_all = pd.read_csv(proto.FLAGS_CSV, index_col="timestamp",
                                parse_dates=True, dtype=bool)
        row_imputed = flags_all.any(axis=1)
    deploy, horizon_tables = {}, {}
    for key, entry in wrappers.items():
        frame = entry["predictions"]
        if row_imputed is not None:
            gaps = max_input_gap_per_origin(frame, row_imputed)
            filtered = deployment_frame(frame, gaps)
        else:
            filtered = frame[frame["output_imputed"].astype(str) != "True"]
        deploy[key] = filtered
        sq = (filtered["predicted"] - filtered["actual_raw"]) ** 2
        psq = (filtered["persistence"] - filtered["actual_raw"]) ** 2
        table = pd.DataFrame({"horizon": filtered["horizon"], "sq": sq, "psq": psq})
        variances = filtered.groupby("horizon")["actual_raw"].var(ddof=0)
        grouped = table.groupby("horizon").agg(mse=("sq", "mean"),
                                               persistence_mse=("psq", "mean"))
        grouped["var"] = variances
        grouped["r2"] = 1 - grouped["mse"] / grouped["var"]
        grouped["persistence_r2"] = 1 - grouped["persistence_mse"] / grouped["var"]
        horizon_tables[key] = grouped.reset_index().sort_values("horizon")
    skip_note = ('<p class="caption">A partir de 4.2, as curvas usam apenas '
                 '<strong>alvos realmente observados</strong> e pulam origens cujo '
                 'encoder contem gap imputado maior que 6 h ou mais de 40% de '
                 'celulas imputadas (regra "deploy observado").</p>')

    parts = []

    # 4.1 Tabela geral
    rows = []
    for key, entry in wrappers.items():
        overall = entry["metrics"]["overall"]
        filtered = deploy[key]
        mse_deploy = float(((filtered["predicted"] - filtered["actual_raw"]) ** 2).mean())
        pers_deploy = float(((filtered["persistence"] - filtered["actual_raw"]) ** 2).mean())
        var_deploy = float(filtered["actual_raw"].var(ddof=0))
        rows.append({
            "modelo": key,
            "MSE": overall["mse"],
            "RMSE": overall["rmse"],
            "MAE": overall["mae"],
            "R2": overall["r2"],
            "MSE persistencia": overall.get("persistence_mse"),
            "skill": overall.get("skill"),
            "MSE (alvos observados)": overall.get("mse_observed"),
            "MSE (alvos imputados)": overall.get("mse_imputed"),
            "MSE deploy (obs+skip)": mse_deploy,
            "skill deploy": 1 - mse_deploy / pers_deploy if pers_deploy else np.nan,
            "R2 deploy": 1 - mse_deploy / var_deploy if var_deploy else np.nan,
            "janelas": len(entry["metrics"]["windows"]),
        })
    overall_table = pd.DataFrame(rows).sort_values("MSE")
    parts.append("<h3>4.1 Metricas globais (todos os meses de teste)</h3>")
    parts.append(table_html(overall_table))
    parts.append('<p class="caption">Metrica principal: MSE ((m/s)², tabela ordenada '
                 'por MSE). skill = 1 - MSE_modelo / MSE_persistencia (negativo = '
                 'pior que persistencia). Colunas "deploy" usam so alvos observados '
                 'e pulam origens com encoder muito imputado (regra em 4.10).</p>')
    parts.append(skip_note)

    # 4.2 MSE por janela mensal (deploy observado)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    persistence_months = None
    for key, entry in wrappers.items():
        p = deploy[key].copy()
        p["month"] = p["origin"].dt.to_period("M").astype(str)
        p["sq"] = (p["predicted"] - p["actual_raw"]) ** 2
        p["psq"] = (p["persistence"] - p["actual_raw"]) ** 2
        monthly_mse = p.groupby("month")["sq"].mean()
        ax.plot(monthly_mse.index, monthly_mse.values, marker="o",
                color=MODEL_COLORS.get(key), label=key)
        if persistence_months is None:
            persistence_months = p.groupby("month")["psq"].mean()
    if persistence_months is not None:
        ax.plot(persistence_months.index, persistence_months.values,
                color="grey", linestyle="--", marker=".", label="persistencia")
    ax.set_ylabel("MSE ((m/s)²)")
    ax.set_title("MSE por mes de teste — deploy observado "
                 "(janela de treino = 3 meses civis anteriores)")
    ax.legend(ncols=5)
    ax.tick_params(axis="x", rotation=45)
    parts.append(fig_to_html(fig))

    # 4.3 MSE por horizonte + skill (deploy observado)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for key in wrappers:
        horizons = horizon_tables[key]
        axes[0].plot(horizons["horizon"], horizons["mse"], marker="o",
                     color=MODEL_COLORS.get(key), label=key)
        skill = 1 - horizons["mse"] / horizons["persistence_mse"]
        axes[1].plot(horizons["horizon"], skill, marker="o",
                     color=MODEL_COLORS.get(key), label=key)
    reference = horizon_tables[next(iter(wrappers))]
    axes[0].plot(reference["horizon"], reference["persistence_mse"], color="grey",
                 linestyle="--", label="persistencia")
    axes[0].set_xlabel("horizonte (passos de 10 min)")
    axes[0].set_ylabel("MSE ((m/s)²)")
    axes[0].set_title("MSE por horizonte — deploy observado")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("horizonte (passos de 10 min)")
    axes[1].set_ylabel("skill (MSE)")
    axes[1].set_title("Skill vs persistencia por horizonte — deploy observado")
    axes[1].set_ylim(-0.5, 1)
    axes[1].legend(ncols=2)
    fig.tight_layout()
    parts.append(fig_to_html(fig))

    # 4.4 Heatmap MSE modelo x horizonte (deploy observado)
    if len(wrappers) > 1:
        mse_rows, labels = [], []
        for key in wrappers:
            mse_rows.append(horizon_tables[key]["mse"].to_numpy())
            labels.append(key)
        mse_rows.append(horizon_tables[next(iter(wrappers))]["persistence_mse"].to_numpy())
        labels.append("persistencia (baseline)")
        mat = np.vstack(mse_rows)

        fig, ax = plt.subplots(figsize=(13, 2.6 + 0.4 * len(labels)))
        image = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                          vmin=float(np.nanmin(mat)), vmax=float(np.nanmax(mat)),
                          interpolation="nearest")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xticks(range(0, 36, 2), [str(h) for h in range(1, 37, 2)])
        ax.set_xlabel("Horizonte (passos de 10 min)")
        ax.set_title("Heatmap MSE ((m/s)²) por modelo e horizonte — deploy "
                     "observado (verde = menor erro)")
        for j in [0, 5, 11, 17, 23, 29, 35]:
            for i in range(len(labels)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=7.5, color="black",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", alpha=0.75, lw=0))
        ax.grid(False)
        fig.colorbar(image, ax=ax, pad=0.01).set_label("MSE ((m/s)²)")
        parts.append(fig_to_html(fig))

        # Heatmap R2 modelo x horizonte (mesmo layout, verde = melhor)
        r2_rows = [horizon_tables[k]["r2"].to_numpy() for k in wrappers]
        r2_rows.append(horizon_tables[next(iter(wrappers))]["persistence_r2"].to_numpy())
        mat = np.vstack(r2_rows)
        fig, ax = plt.subplots(figsize=(13, 2.6 + 0.4 * len(labels)))
        image = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                          vmin=float(np.nanmin(mat)), vmax=float(np.nanmax(mat)),
                          interpolation="nearest")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xticks(range(0, 36, 2), [str(h) for h in range(1, 37, 2)])
        ax.set_xlabel("Horizonte (passos de 10 min)")
        ax.set_title("Heatmap R² por modelo e horizonte (verde = melhor; "
                     "0 = equivalente a media)")
        for j in [0, 5, 11, 17, 23, 29, 35]:
            for i in range(len(labels)):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center",
                        fontsize=7.5, color="black",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", alpha=0.75, lw=0))
        ax.grid(False)
        fig.colorbar(image, ax=ax, pad=0.01).set_label("R²")
        parts.append(fig_to_html(fig))

    # 4.5 MSE por horizonte, uma linha por janela de teste (deploy observado)
    monthly = {}
    for key, entry in wrappers.items():
        p = deploy[key].copy()
        p["month"] = p["origin"].dt.to_period("M").astype(str)
        p["sq_err"] = (p["predicted"] - p["actual_raw"]) ** 2
        # a primeira origem da janela pode cair no fim do mes anterior
        # (tail de treino) — descarte grupos que nao sao um mes de teste
        counts = p["month"].value_counts()
        p = p[p["month"].isin(counts[counts >= 100].index)]
        grouped = (p.groupby(["month", "horizon"])["sq_err"].mean()
                   .reset_index(name="mse"))
        monthly[key] = grouped
    if monthly:
        n_months = max(len(g["month"].unique()) for g in monthly.values())
        cmap = plt.get_cmap("viridis")
        fig, axes = plt.subplots(len(wrappers), 1,
                                 figsize=(12, 2.9 * len(wrappers)),
                                 sharex=True, sharey=True)
        if len(wrappers) == 1:
            axes = [axes]
        all_mse = np.concatenate([g["mse"].to_numpy() for g in monthly.values()])
        for ax, (key, grouped) in zip(axes, monthly.items()):
            months = sorted(grouped["month"].unique())
            for i, month in enumerate(months):
                color = cmap(i / max(1, n_months - 1))
                line = grouped[grouped["month"] == month].sort_values("horizon")
                ax.plot(line["horizon"], line["mse"], color=color, lw=1.1,
                        alpha=0.9, label=month)
            mean_line = horizon_tables[key]
            ax.plot(mean_line["horizon"], mean_line["mse"], color="black",
                    lw=2.2, linestyle="--", label="geral (media)")
            ax.legend(ncols=3, fontsize=6.5, loc="upper left")
            ax.set_ylabel("MSE ((m/s)²)")
            ax.set_title(f"{key}")
        axes[-1].set_xlabel("horizonte (passos de 10 min)")
        axes[-1].set_ylim(0, np.nanpercentile(all_mse, 99.5) * 1.15)
        fig.suptitle("MSE por horizonte — uma linha por janela de teste "
                     "(tracejado preto = media geral)", y=1.0)
        fig.tight_layout()
        parts.append(fig_to_html(fig))
        parts.append('<p class="caption">Linhas em gradiente: roxo-escuro = janela mais '
                     'antiga (dez/2021), amarelo = mais recente (set/2022). Meses de '
                     'inverno/transicao (2021-12, 2022-01) degradam mais com o horizonte; '
                     'todos convergem perto da variancia do vento em horizontes longos.</p>')

    # 4.4 Bias por horizonte
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for key, entry in wrappers.items():
        horizons = pd.DataFrame(entry["metrics"]["by_horizon"]).sort_values("horizon")
        ax.plot(horizons["horizon"], horizons["nmae"], marker="o",
                color=MODEL_COLORS.get(key), label=key)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("horizonte (passos de 10 min)")
    ax.set_ylabel("NMAE")
    ax.set_title("Erro normalizado (NMAE) por horizonte")
    ax.legend(ncols=4)
    parts.append(fig_to_html(fig))

    # 4.5 Hexbin real vs previsto
    ncols = 2 if len(wrappers) <= 4 else 3
    nrows = -(-len(wrappers) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    for ax, (key, entry) in zip(np.asarray(axes).ravel(), wrappers.items()):
        predictions = entry["predictions"]
        if predictions is None:
            continue
        ax.hexbin(predictions["actual_raw"], predictions["predicted"],
                  gridsize=60, mincnt=1, cmap="viridis", bins="log")
        limit = np.nanpercentile(predictions["actual_raw"], 99.9) * 1.05
        ax.plot([0, limit], [0, limit], color="white", lw=0.8, linestyle="--")
        ax.set_xlabel("ws100 real (m/s)")
        ax.set_ylabel("previsto (m/s)")
        ax.set_title(f"{key} (n={len(predictions):,})")
    for ax in np.asarray(axes).ravel()[len(wrappers):]:
        ax.axis("off")
    fig.tight_layout()
    parts.append(fig_to_html(fig))

    # 4.6 Serie temporal de exemplo (ultimo mes de teste)
    last_month = max(w["window"] for e in wrappers.values() for w in e["metrics"]["windows"])
    fig, axes = plt.subplots(len(wrappers), 1, figsize=(13, 2.4 * len(wrappers)),
                             sharex=True, sharey=True)
    if len(wrappers) == 1:
        axes = [axes]
    for ax, (key, entry) in zip(axes, wrappers.items()):
        predictions = entry["predictions"]
        month = predictions[predictions["origin"].dt.strftime("%Y-%m") == last_month]
        step = month[month["horizon"] == 1].sort_values("timestamp")
        imputed = step["output_imputed"].astype(str) == "True"
        ax.plot(step["timestamp"], step["actual_raw"], color="black", lw=0.8, label="real")
        ax.plot(step["timestamp"], step["predicted"], color=MODEL_COLORS.get(key),
                lw=0.8, label=f"{key} (h=1)")
        ax.scatter(step.loc[imputed, "timestamp"], step.loc[imputed, "actual_raw"],
                   color="red", s=6, zorder=3, label="alvo imputado")
        ax.set_ylabel("m/s")
        ax.legend(loc="upper right", ncols=3)
    axes[0].set_title(f"Ultimo mes de teste ({last_month}): real vs previsto no horizonte 1")
    fig.autofmt_xdate()
    parts.append(fig_to_html(fig))

    # 4.7 Leques: 3 predicoes completas por mes de teste (origens as 06/12/18 h
    # do dia 15), mostrando as 12 h de entrada reais + os 36 passos previstos
    # por modelo + persistencia + a serie real do periodo, com faixas verticais
    # de dado imputado (ws100 e demais colunas).
    if proto.SERIES_CSV.is_file() and proto.FLAGS_CSV.is_file():
        series = pd.read_csv(proto.SERIES_CSV, index_col="timestamp", parse_dates=True)
        flags = pd.read_csv(proto.FLAGS_CSV, index_col="timestamp",
                            parse_dates=True, dtype=bool)
        target_imputed = flags["ws100"]
        other_imputed = flags.drop(columns="ws100").any(axis=1) & ~target_imputed
        first_entry = next(iter(wrappers.values()))
        months = sorted({w["window"] for e in wrappers.values()
                         for w in e["metrics"]["windows"]})
        parts.append("<h3>4.7 Leques de previsao — 3 origens por mes "
                     "(06 h, 12 h e 18 h do dia 15)</h3>")
        parts.append(
            "<p>Cada painel mostra uma unidade completa de previsao: as "
            "<strong>12 h de entrada</strong> observadas (encoder), a origem "
            "(linha vertical cinza) e os <strong>36 passos (6 h) previstos</strong> "
            "por cada modelo e pela persistencia, contra a serie real medida. "
            "Faixas verticais: <span style='color:#c44e52'>vermelho</span> = "
            "celula de ws100 imputada; <span style='color:#dd8452'>ambar</span> = "
            "imputacao em outras colunas. Dias desde o fim do treino indicados "
            "no titulo.</p>"
        )
        for month in months:
            frame = first_entry["predictions"]
            available = frame.loc[
                frame["origin"].dt.strftime("%Y-%m") == month, "origin"
            ].unique()
            available = np.sort(available)
            if not len(available):
                continue
            window = next(w for e in wrappers.values() for w in e["metrics"]["windows"]
                          if w["window"] == month)
            train_end = pd.Timestamp(window["test_start"]) - pd.Timedelta(minutes=10)
            fig, axes = plt.subplots(3, 1, figsize=(12, 8.2))
            plotted = 0
            for ax, hour in zip(axes, (6, 12, 18)):
                wanted = pd.Timestamp(f"{month}-15 {hour:02d}:00")
                pos = int(np.searchsorted(available, wanted.to_datetime64()))
                candidates = available[max(0, pos - 1): pos + 2]
                if not len(candidates):
                    continue
                origin = pd.Timestamp(min(
                    candidates,
                    key=lambda t: abs((t - wanted.to_datetime64())
                                      / np.timedelta64(1, "s")),
                ))
                start = origin - pd.Timedelta(minutes=710)
                end = origin + pd.Timedelta(minutes=360)
                index = pd.date_range(start, end, freq="10min")
                real = series["ws100"].reindex(index)

                for mask, color, alpha in (
                    (target_imputed, "#c44e52", 0.20),
                    (other_imputed, "#dd8452", 0.12),
                ):
                    flagged = mask.reindex(index).fillna(False).to_numpy()
                    run_start = None
                    for i, flag in enumerate(flagged):
                        if flag and run_start is None:
                            run_start = i
                        elif not flag and run_start is not None:
                            ax.axvspan(index[run_start], index[i - 1], color=color,
                                       alpha=alpha, lw=0)
                            run_start = None
                    if run_start is not None:
                        ax.axvspan(index[run_start], index[-1], color=color,
                                   alpha=alpha, lw=0)

                ax.plot(index, real, color="black", lw=1.1, label="ws100 real")
                for key, entry in wrappers.items():
                    fan = entry["predictions"]
                    fan = fan[fan["origin"] == origin].sort_values("horizon")
                    if not len(fan):
                        continue
                    ax.plot(fan["timestamp"], fan["predicted"], lw=1.4,
                            color=MODEL_COLORS.get(key),
                            label=key if plotted == 0 else None)
                    if key == next(iter(wrappers)):
                        ax.plot(fan["timestamp"], fan["persistence"], lw=1.1,
                                ls="--", color="grey",
                                label="persistencia" if plotted == 0 else None)
                ax.axvline(origin, color="grey", ls=":", lw=1)
                days_from_train = (origin - train_end).total_seconds() / 86400
                ax.set_ylabel("m/s")
                ax.set_title(f"{origin:%d/%m %H:%M} — {days_from_train:.1f} dias "
                             "apos o fim do treino")
                plotted += 1
            if plotted:
                handles, labels = axes[0].get_legend_handles_labels()
                axes[0].legend(handles, labels, fontsize=7, ncols=3, loc="upper right")
                fig.suptitle(f"Leques de {month} (entrada 12 h + previsao 6 h)",
                             y=1.0)
                fig.tight_layout()
                parts.append(fig_to_html(fig))

    # 4.7 Histograma de erros
    fig, axes = plt.subplots(1, len(wrappers), figsize=(3.2 * len(wrappers), 3.2),
                             sharey=True)
    if len(wrappers) == 1:
        axes = [axes]
    for ax, (key, entry) in zip(axes, wrappers.items()):
        errors = entry["predictions"]["predicted"] - entry["predictions"]["actual_raw"]
        ax.hist(errors, bins=80, color=MODEL_COLORS.get(key), alpha=0.85)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{key}\nbias={errors.mean():.3f}")
        ax.set_xlabel("erro (m/s)")
    axes[0].set_ylabel("frequencia")
    fig.tight_layout()
    parts.append(fig_to_html(fig))

    # 4.8 Alvos observados vs imputados
    rows = []
    for key, entry in wrappers.items():
        overall = entry["metrics"]["overall"]
        rows.append({
            "modelo": key,
            "MSE observado": overall.get("mse_observed"),
            "MSE imputado": overall.get("mse_imputed"),
            "delta (imp - obs)": (overall.get("mse_imputed") - overall.get("mse_observed"))
            if overall.get("mse_imputed") is not None and overall.get("mse_observed") is not None
            else np.nan,
        })
    parts.append("<h3>4.9 Efeito da imputacao no erro do forecast</h3>"
                 + table_html(pd.DataFrame(rows)))

    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / max(1, len(wrappers))
    for i, (key, entry) in enumerate(wrappers.items()):
        overall = entry["metrics"]["overall"]
        ax.bar(i - width / 2, overall.get("mse_observed", np.nan), width * 0.9,
               color=MODEL_COLORS.get(key), alpha=0.9,
               label=f"{key} observado" if i == 0 else None)
        ax.bar(i + width / 2, overall.get("mse_imputed", np.nan), width * 0.9,
               color=MODEL_COLORS.get(key), alpha=0.45, hatch="//",
               label=f"{key} imputado" if i == 0 else None)
    ax.set_xticks(range(len(wrappers)), list(wrappers))
    ax.set_ylabel("MSE ((m/s)²)")
    ax.set_title("MSE do forecast em alvos observados vs alvos imputados")
    parts.append(fig_to_html(fig))

    # 4.10 ARIMA e persistencia se beneficiam do dado imputado?
    subsets = ("todos os alvos", "so alvos observados", "deploy (obs + skip)")
    subset_rows = []
    subset_mse, subset_persistence = {}, {}
    for key in wrappers:
        frame = wrappers[key]["predictions"]
        observed = frame[frame["output_imputed"].astype(str) != "True"]
        variants = {
            "todos os alvos": frame,
            "so alvos observados": observed,
            "deploy (obs + skip)": deploy[key],
        }
        subset_mse[key] = {}
        for label, sub in variants.items():
            mse = float(((sub["predicted"] - sub["actual_raw"]) ** 2).mean())
            per = float(((sub["persistence"] - sub["actual_raw"]) ** 2).mean())
            subset_mse[key][label] = (mse, per)
        subset_persistence[key] = subset_mse[key]
    for key in wrappers:
        row = {"metodo": key}
        for label in subsets:
            mse, per = subset_mse[key][label]
            row[f"MSE {label}"] = mse
            row[f"persistence MSE {label}"] = per
            row[f"skill {label}"] = 1 - mse / per if per else np.nan
        subset_rows.append(row)
    parts.append("<h3>4.10 ARIMA, persistencia e o subsidio do dado imputado</h3>")
    parts.append(
        "<p>Hipotese: gaps longos imputados viram trechos ~planos que ARIMA e "
        "persistencia preveem quase de graca, inflando suas metricas. Recortes: "
        "(a) todos os alvos; (b) apenas alvos realmente observados; (c) deploy "
        "— observados e SEM origens cujo encoder tem gap imputado &gt; 6 h ou "
        "&gt; 40% de celulas imputadas.</p>"
    )
    parts.append(table_html(pd.DataFrame(subset_rows)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    x = np.arange(len(wrappers))
    width = 0.26
    for i, label in enumerate(subsets):
        values = [subset_mse[key][label][0] for key in wrappers]
        persistence = [subset_mse[key][label][1] for key in wrappers]
        axes[0].bar(x + (i - 1) * width, values, width * 0.9, label=label,
                    color=["#4c72b0", "#55a868", "#dd8452"][i])
        if i == 0:
            axes[0].scatter(x + (i - 1) * width, persistence, marker="_",
                            s=300, color="black", label="persistencia (todos)")
            axes[0].scatter(x + (i - 1) * width + 0.02,
                            [subset_mse[key]["deploy (obs + skip)"][1] for key in wrappers],
                            marker="_", s=300, color="red", label="persistencia (deploy)")
        skills = [subset_mse[key][label][0] / subset_mse[key][label][1] - 1
                  for key in wrappers]
        axes[1].bar(x + (i - 1) * width, skills, width * 0.9, label=label,
                    color=["#4c72b0", "#55a868", "#dd8452"][i])
    for ax, (ylabel, title) in zip(
        axes,
        (("MSE ((m/s)²)", "MSE por recorte de imputacao"),
         ("MSE relativo a persistencia", "Custo relativo vs persistencia "
          "(0 = empata; negativo = melhor)")),
    ):
        ax.set_xticks(x, list(wrappers), rotation=20)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[1].axhline(0, color="black", lw=0.8)
    fig.tight_layout()
    parts.append(fig_to_html(fig))
    parts.append('<p class="caption">Se a hipotese estiver certa, as barras '
                 'verdes/laranja (sem imputacao) mostram MSE maior para ARIMA e '
                 'persistencia do que a azul (todos), e o skill do ARIMA cai no '
                 'recorte deploy.</p>')

    # 4.11 Erro vs fracao imputada da janela de entrada
    fig, ax = plt.subplots(figsize=(10, 4))
    for key, entry in wrappers.items():
        predictions = entry["predictions"].copy()
        predictions["abs_error"] = (
            predictions["predicted"] - predictions["actual_raw"]
        ).abs()
        predictions["frac_bin"] = pd.cut(
            predictions["input_imputed_frac"].astype(float), bins=np.linspace(0, 1, 9)
        )
        grouped = predictions.groupby("frac_bin", observed=True)["abs_error"].mean()
        centers = [interval.mid for interval in grouped.index]
        ax.plot(centers, grouped.values, marker="o", color=MODEL_COLORS.get(key), label=key)
    ax.set_xlabel("fracao de celulas imputadas na janela de entrada (encoder)")
    ax.set_ylabel("MAE (m/s)")
    ax.set_title("Degradacao do forecast conforme a entrada contem mais imputacao")
    ax.legend(ncols=4)
    parts.append(fig_to_html(fig))

    # 4.12 Erro por posicao no conjunto de teste: distancia (em dias) da
    # amostra ate o fim da janela de treino que a gerou. A origem diz ha quantos
    # dias o modelo foi retreinado; o alvo soma o horizonte previsto.
    position_table = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for key, entry in wrappers.items():
        p = deploy[key].copy()
        for w in entry["metrics"]["windows"]:
            mask = ((p["timestamp"] >= pd.Timestamp(w["test_start"]))
                    & (p["timestamp"] <= pd.Timestamp(w["test_end"])))
            p.loc[mask, "train_end"] = (
                pd.Timestamp(w["test_start"]) - pd.Timedelta(minutes=10)
            )
        p = p.dropna(subset=["train_end"])
        p["dist_origin"] = (
            (p["origin"] - p["train_end"]).dt.total_seconds() / 86400
        ).clip(lower=0)
        p["dist_target"] = (
            (p["timestamp"] - p["train_end"]).dt.total_seconds() / 86400
        )
        p["sq_err"] = (p["predicted"] - p["actual_raw"]) ** 2
        # censura de borda: origens nos ultimos ~2,5 dias do mes perdem os
        # horizontes longos (a sequencia cruzaria o fim do mes e nem chega a
        # ser formada) — restrinja a faixas com mistura completa de horizontes
        for col, ax, limit, label in (
            ("dist_origin", axes[0], 27, "origem da previsao"),
            ("dist_target", axes[1], 28, "alvo previsto"),
        ):
            daily = (p[p[col] <= limit]
                     .assign(day=lambda q: np.floor(q[col]).clip(0, 34).astype(int))
                     .groupby("day")["sq_err"].mean())
            ax.plot(daily.index, daily.values, color=MODEL_COLORS.get(key),
                    lw=0.8, alpha=0.3)
            smooth = daily.rolling(3, center=True, min_periods=1).mean()
            ax.plot(smooth.index, smooth.values, marker="o", ms=3.5,
                    color=MODEL_COLORS.get(key), label=key)
            ax.set_xlabel(f"dias entre o {label} e o fim do treino")
        p["week"] = pd.cut(p["dist_origin"], bins=(-0.01, 7, 14, 21, 28, 40),
                           labels=("0-7d", "7-14d", "14-21d", "21-28d", "28-35d"))
        weekly = p.groupby("week", observed=True)["sq_err"].mean()
        position_table.append({
            "modelo": key,
            **{str(k): v for k, v in weekly.items()},
        })
    axes[0].set_ylabel("MSE ((m/s)²)")
    axes[0].set_title("Erro vs distancia da ORIGEM ao treino")
    axes[1].set_title("Erro vs distancia do ALVO ao treino")
    axes[0].legend(ncols=4)
    fig.tight_layout()
    parts.append(fig_to_html(fig))
    parts.append('<p class="caption">Cada dia dentro do mes de teste afasta as '
                 'previsoes do ultimo retreino (linha fina = MSE diario; grossa = '
                 'media movel de 3 dias). Painel direito combina o afastamento da '
                 'origem com o horizonte (1-36 passos = ate +2,5 dias). Faixas '
                 'limitadas a <= 27/28 dias para evitar censura de horizonte na '
                 'borda do mes; 4.11 mostra os mesmos dados em bins semanais.</p>')

    position_df = pd.DataFrame(position_table)
    if len(position_df):
        first_col = position_df.columns[1]
        last_col = position_df.columns[-1]
        parts.append("<h3>4.12 MSE semanal por distancia do treino (deploy)</h3>"
                     + table_html(position_df))
        growth = (position_df[last_col] / position_df[first_col] - 1) * 100
        parts.append('<p class="caption">Variacao do MSE do primeiro ao ultimo bin: '
                     + ", ".join(f"{row['modelo']} {g:+.0f}%"
                                 for (_, row), g in zip(position_df.iterrows(), growth))
                     + ".</p>")

    # 4.13 Tabelas mensais (deploy observado; MSE principal, RMSE secundaria)
    rows = []
    for key in wrappers:
        p = deploy[key]
        month = p["origin"].dt.to_period("M").astype(str)
        mse = ((p["predicted"] - p["actual_raw"]) ** 2).groupby(month).mean()
        rmse = mse.pow(0.5)
        for m in mse.index:
            rows.append({"mes": m, "modelo": key, "MSE": mse[m], "RMSE": rmse[m]})
    windows_df = pd.DataFrame(rows)
    parts.append("<h3>4.13 MSE mensal por modelo (deploy observado)</h3>"
                 + windows_df.pivot_table(index="mes", columns="modelo", values="MSE")
                 .to_html(float_format="%.4f", border=0, classes="data"))
    parts.append("<h3>4.14 RMSE mensal por modelo (secundaria)</h3>"
                 + windows_df.pivot_table(index="mes", columns="modelo", values="RMSE")
                 .to_html(float_format="%.4f", border=0, classes="data"))

    # 4.15 Metricas mensais x presenca de imputacao no mes
    first_pred = next(iter(wrappers.values()))["predictions"]
    month_of = first_pred["origin"].dt.to_period("M").astype(str)
    is_imputed = first_pred["output_imputed"].astype(str) == "True"
    # descarta meses-fantasma (a origem de borda da 1a janela cai no mes anterior)
    month_counts = is_imputed.groupby(month_of).size()
    valid_months = month_counts[month_counts >= 100].index
    imp_frac = (is_imputed.groupby(month_of).mean() * 100).reindex(valid_months)
    deploy_mse = windows_df.pivot_table(index="mes", columns="modelo", values="MSE")
    deploy_mse = deploy_mse.reindex(valid_months)
    if "% alvos imputados" not in deploy_mse.columns:
        deploy_mse.insert(0, "% alvos imputados",
                          imp_frac.reindex(deploy_mse.index).round(2))

    # MSE sobre os ALVOS imputados de cada mes (quando existirem)
    imp_mse_cols = {}
    for key in wrappers:
        p = wrappers[key]["predictions"]
        m = p["origin"].dt.to_period("M").astype(str)
        sel = p["output_imputed"].astype(str) == "True"
        if sel.any():
            imp_mse_cols[f"{key} (imp)"] = (
                ((p.loc[sel, "predicted"] - p.loc[sel, "actual_raw"]) ** 2)
                .groupby(m[sel]).mean()
            )
    parts.append("<h3>4.15 Metricas mensais x presenca de dado imputado</h3>")
    parts.append("<p>Recorte deploy (so alvos observados) de cada mes, cruzado com "
                 "a fracao de alvos imputados no mes. Meses com lacunas "
                 "estruturais (ago/2022: 15 dias) tem a maior fracao imputada e "
                 "tambem o menor MSE observado — o regime plano e 'facil' para "
                 "todos, inclusive para ARIMA e persistencia.</p>")
    parts.append(table_html(deploy_mse.reset_index()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    axes[0].bar(range(len(imp_frac)), imp_frac.values, color="#c44e52", alpha=0.8)
    axes[0].set_xticks(range(len(imp_frac)), imp_frac.index, rotation=45)
    axes[0].set_ylabel("% de alvos imputados no mes")
    axes[0].set_title("Presenca de imputacao no alvo, por mes")
    for key in wrappers:
        xs = deploy_mse["% alvos imputados"]
        ys = deploy_mse[key]
        axes[1].plot(xs, ys, marker="o", lw=1.4, ms=5, alpha=0.85,
                     color=MODEL_COLORS.get(key), label=key)
    axes[1].set_xlabel("% de alvos imputados no mes")
    axes[1].set_ylabel("MSE deploy do mes ((m/s)²)")
    axes[1].set_title("MSE mensal (deploy) vs fracao imputada")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    parts.append(fig_to_html(fig))

    imp_table = pd.DataFrame(imp_mse_cols)
    if len(imp_table):
        parts.append("<p class='caption'>MSE sobre os ALVOS imputados, por mes e "
                     "modelo (onde existirem):</p>")
        parts.append(table_html(imp_table.reset_index()
                                .rename(columns={"index": "mes"})))
    return section("4. Walk-forward validation", "\n".join(parts), "wfo")


def sec_original_vs_wfo(data):
    """Compara o erro dos testes originais (estatico e rolling na serie
    LiDAR, data/results/report_homologado) com o erro do walk-forward na
    serie imputada LiDAR+SODAR."""
    wrappers = {
        key: entry for key, entry in data.get("wrappers", {}).items()
        if entry.get("predictions") is not None
    }
    if not ORIGINAL_EVAL_JSON.is_file() or not wrappers:
        return section("5. Testes originais vs walk-forward",
                       "<p>evaluation.json dos testes originais ausente ou sem "
                       "modelos WFO finalizados.</p>", "original")
    with open(ORIGINAL_EVAL_JSON, encoding="utf-8") as handle:
        original = json.load(handle)

    parts = [
        "<p>Os <strong>testes originais</strong> avaliaram cada modelo uma unica vez "
        "sobre a <strong>serie LiDAR</strong> (dataset.csv, mesma faixa de treino): "
        "particao estatica cronologica (75/20/5, teste = 9.792 amostras) e protocolo "
        "rolling de inferencia (origem a cada amostra, horizonte maximo). O "
        "<strong>WFO</strong> desta avaliacao retreina mensalmente e prevê 10 meses "
        "SODAR imputados (LiDAR+SODAR), origens a cada 6 amostras, horizontes 1-36. "
        "Sao protocolos, periodos e instrumentos diferentes — a comparacao mede a "
        "penalidade conjunta de: troca de instrumento (LiDAR→SODAR), dado imputado "
        "(13,3% das celulas) e meses ainda nao vistos, e nao a qualidade absoluta "
        "dos modelos.</p>",
        '<div class="hl"><strong>Teste complementar (75/25 sem validação)</strong>: '
        "um novo experimento na LiDAR removeu a partição de validação (split 75/25 "
        "treino/teste, 60 épocas fixas, sem early stopping) e está documentado em "
        "<code>data/results/report_7525/relatorio_resultados.html</code>. Os números "
        "dessa linha entram na comparação apenas como referência adicional de "
        "protocolo.</div>",
    ]

    rows = []
    for key, entry in wrappers.items():
        name = entry["name"]
        if name not in original:
            continue
        static_m = original[name]["all"]
        rolling_m = original[name].get("rolling", {})
        wfo_m = entry["metrics"]["overall"]
        rows.append({
            "modelo": key,
            "MSE estatico (LiDAR)": static_m["mse"],
            "MSE rolling (LiDAR)": rolling_m.get("mse"),
            "MSE WFO (SODAR imputado)": wfo_m["mse"],
            "Delta WFO vs estatico": (wfo_m["mse"] / static_m["mse"] - 1) * 100
            if static_m["mse"] else np.nan,
            "RMSE estatico (LiDAR)": static_m["rmse"],
            "RMSE WFO (SODAR imputado)": wfo_m["rmse"],
        })
    table = pd.DataFrame(rows).sort_values("MSE WFO (SODAR imputado)")
    parts.append("<h3>5.1 MSE/RMSE por protocolo</h3>" + table_html(table))
    parts.append('<p class="caption">Delta = (MSE_WFO / MSE_estatico - 1). O teste '
                 'estatico da TCN refere-se a arquitetura anterior (com context '
                 'pooling); a TCN atual do WFO usa mapeamento direto do ultimo '
                 'passo.</p>')

    # Barras agrupadas: estatico | rolling | WFO
    fig, ax = plt.subplots(figsize=(11, 4.2))
    width = 0.26
    model_keys = [row["modelo"] for row in rows]
    for i, (_, row) in enumerate(pd.DataFrame(rows).iterrows()):
        xbase = i
        ax.bar(xbase - width, row["MSE estatico (LiDAR)"], width * 0.9,
               color="#4c72b0", label="estatico (LiDAR)" if i == 0 else None)
        ax.bar(xbase, row["MSE rolling (LiDAR)"], width * 0.9,
               color="#8172b3", label="rolling (LiDAR)" if i == 0 else None)
        ax.bar(xbase + width, row["MSE WFO (SODAR imputado)"], width * 0.9,
               color="#dd8452", label="WFO (SODAR imputado)" if i == 0 else None)
        ax.text(xbase + width, row["MSE WFO (SODAR imputado)"] + 0.03,
                f"{row['Delta WFO vs estatico']:+.0f}%", ha="center", fontsize=8)
    ax.set_xticks(range(len(model_keys)), model_keys)
    ax.set_ylabel("MSE ((m/s)²)")
    ax.set_ylim(0, pd.DataFrame(rows)[["MSE estatico (LiDAR)", "MSE rolling (LiDAR)",
                                       "MSE WFO (SODAR imputado)"]].to_numpy().max() * 1.28)
    ax.set_title("MSE por modelo: testes originais na LiDAR vs walk-forward SODAR imputado")
    ax.legend()
    parts.append(fig_to_html(fig))

    # Decomposicao do aumento: protocolo (estatico->rolling na mesma LiDAR) e
    # dominio/dado (rolling LiDAR -> WFO SODAR imputado)
    parts.append("<h3>5.2 Decomposicao do aumento de erro</h3>")
    rows = []
    for row in table.to_dict(orient="records"):
        static_m = row["MSE estatico (LiDAR)"]
        rolling_m = row["MSE rolling (LiDAR)"]
        wfo_m = row["MSE WFO (SODAR imputado)"]
        rows.append({
            "modelo": row["modelo"],
            "protocolo (estatico->rolling LiDAR)": f"{(rolling_m / static_m - 1) * 100:+.0f}%",
            "dominio+dado (rolling LiDAR->WFO)": f"{(wfo_m / rolling_m - 1) * 100:+.0f}%",
            "total (estatico->WFO)": f"{(wfo_m / static_m - 1) * 100:+.0f}%",
        })
    parts.append(table_html(pd.DataFrame(rows)))

    bullets = []
    medias = pd.DataFrame(rows)
    for col in medias.columns[1:]:
        values = medias[col].str.rstrip("%").astype(float)
        bullets.append(f"{col}: media de {values.mean():+.0f}% entre os modelos.")
    bullets.append(
        "A maior parte do aumento vem da mudanca de dominio (instrumento SODAR + "
        "dado imputado + meses nao vistos), e nao do protocolo de inferencia em si: "
        "o rolling dentro da propria LiDAR custa menos que a migracao para o WFO."
    )
    bullets.append(
        "lstm_bi permanece o melhor modelo nos dois cenarios, indicando que o "
        "ranking das arquiteturas e estavel entre domínios."
    )
    body = "<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"
    parts.append(body)
    return section("5. Testes originais (LiDAR) vs walk-forward (SODAR imputado)",
                   "\n".join(parts), "original")


def sec_discussao(data):
    wrappers = data.get("wrappers", {})
    meta = data.get("meta", {})
    bullets = []
    if wrappers:
        overalls = {key: e["metrics"]["overall"] for key, e in wrappers.items()}
        best = min(overalls, key=lambda k: overalls[k]["mse"])
        bullets.append(
            f"<strong>{best}</strong> apresenta o menor MSE global "
            f"({overalls[best]['mse']:.4f} (m/s)²; RMSE {overalls[best]['rmse']:.4f})."
        )
        ranking = sorted(overalls, key=lambda k: overalls[k]["mse"])
        bullets.append("Ordenacao por MSE (metrica principal): " + " < ".join(
            f"{k} ({overalls[k]['mse']:.4f})" for k in ranking) + ".")
        for key, overall in overalls.items():
            skill = overall.get("skill")
            if skill is not None and skill == skill:
                verdict = "supera" if skill > 0 else "nao supera"
                bullets.append(
                    f"{key} {verdict} a persistencia (skill {skill:.3f} = "
                    f"{skill:.1%} de reducao de MSE)."
                )
        deltas = {
            key: overalls[key].get("mse_imputed") - overalls[key].get("mse_observed")
            for key in overalls
            if overalls[key].get("mse_imputed") is not None
            and overalls[key].get("mse_observed") is not None
        }
        if deltas:
            worst = max(deltas, key=deltas.get)
            bullets.append(
                "Em todos os modelos o erro sobre alvos imputados e maior que sobre "
                "alvos observados; o pior caso e "
                f"{worst} (+{deltas[worst]:.4f} (m/s)² de MSE). Isso quantifica o custo do "
                "dado imputado no alvo e deve entrar na discussao de confiabilidade."
            )
        worst_month = {}
        for key, entry in wrappers.items():
            windows = sorted(entry["metrics"]["windows"], key=lambda w: w["mae"])
            worst_month[key] = windows[-1]
        common_month = max({w["window"] for w in worst_month.values()},
                           key=lambda m: sum(w["window"] == m for w in worst_month.values()))
        bullets.append(
            f"O mes mais dificil converge para {common_month} (pior janela de "
            f"{len([1 for w in worst_month.values() if w['window'] == common_month])} de "
            f"{len(worst_month)} modelos)."
        )
    if meta.get("constant_runs_imputed"):
        total_cols = len(meta["constant_runs_imputed"])
        bullets.append(
            f"A auditoria achou patamares constantes pos-imputacao em {total_cols} "
            "colunas; janelas de teste que coincidem com esses trechos tendem a "
            "superestimar a qualidade em alvos imputados."
        )
    bullets.append(
        "O LiDAR (set-nov/2021) entra apenas no treino das primeiras janelas "
        "(out-dez/2021 para o teste de jan/2022 e nov-dez/2021 para fev/2022); "
        "todos os meses de teste sao SODAR, entao a comparacao entre modelos nao e "
        "contaminada pela troca de instrumento."
    )
    bullets.append(
        "Todas as predicoes individuais (origem, horizonte, real, previsto, "
        "persistencia, flags de imputacao) estao em "
        "data/results/wfo_lightgbm/&lt;modelo&gt;/predictions_all.csv para analises "
        "posteriores."
    )
    body = "<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"
    return section("6. Sintese para discussao", body, "discussao")


# --- Montagem -----------------------------------------------------------------

CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; max-width: 1100px;
       margin: 2rem auto; padding: 0 1.5rem; color: #222; line-height: 1.5; }
h1 { border-bottom: 3px solid #2c3e50; padding-bottom: .4rem; }
h2 { border-bottom: 1px solid #ccc; padding-bottom: .3rem; margin-top: 2.2rem; }
h3 { margin-top: 1.6rem; color: #2c3e50; }
figure { margin: 1.2rem 0; text-align: center; }
figure img { box-shadow: 0 1px 4px rgba(0,0,0,.15); border-radius: 4px; }
.caption { font-size: .85rem; color: #666; margin-top: .4rem; }
table.data { border-collapse: collapse; font-size: .85rem; margin: .8rem 0; }
table.data th, table.data td { border: 1px solid #ddd; padding: .35rem .6rem; }
table.data th { background: #f4f6f8; text-align: right; }
table.data td:nth-child(1) { text-align: left; }
nav a { margin-right: 1rem; }
code { background: #f4f4f4; padding: 0 .3rem; border-radius: 3px; }
.hl { background: #eef4fb; border-left: 4px solid #4c72b0; padding: .6rem .9rem;
      margin: .8rem 0; border-radius: 4px; }
"""


def build_html(data, subtitle=None):
    meta = data.get("meta", {})
    generated = pd.Timestamp.now().strftime("%d/%m/%Y %H:%M")
    title = "Walk-forward validation com série imputada por LightGBM"
    if subtitle:
        title += f" — {subtitle}"
    toc = " ".join(
        f'<a href="#{a}">{t}</a>' for a, t in (
            ("protocolo", "Protocolo"), ("dados", "Dados e imputação"),
            ("validacao", "Validação da imputação"), ("wfo", "Walk-forward"),
            ("original", "Original vs WFO"), ("discussao", "Síntese"),
        )
    )
    header = (
        f"<h1>{title}</h1>"
        f"<p>Gerado em {generated} · {meta.get('period_start', '')} → "
        f"{meta.get('period_end', '')}</p>"
        f"<nav>{toc}</nav>"
    )
    sections = [
        sec_intro(data),
        sec_dados(data),
        sec_validacao(data),
        sec_walkforward(data),
        sec_original_vs_wfo(data),
        sec_discussao(data),
    ]
    return (
        "<!DOCTYPE html><html lang='pt-BR'><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{CSS}</style></head><body>"
        + header + "".join(sections) + "</body></html>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--open", action="store_true", help="abre o HTML no browser")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="diretorio com os resultados (default: data/results/wfo_lightgbm)")
    parser.add_argument("--subtitle", default=None,
                        help="texto adicionado ao titulo (ex.: variacao de protocolo)")
    args = parser.parse_args()

    if args.results_dir:
        proto.RESULTS_DIR = common.resolve(args.results_dir)
        proto.PROGRESS_JSON = proto.RESULTS_DIR / "progress.json"
        proto.REPORT_HTML = (common.resolve("reports/imputacao")
                             / proto.RESULTS_DIR.name.replace("_", "-")
                             / "report.html")

    data = load_inputs()
    if not data["wrappers"]:
        print("AVISO: nenhum modelo finalizado ainda; o relatorio saira parcial "
              "(dados/imputacao apenas).")
    html = build_html(data, subtitle=args.subtitle)
    proto.REPORT_HTML.parent.mkdir(parents=True, exist_ok=True)
    proto.REPORT_HTML.write_text(html, encoding="utf-8")
    print(f"Relatorio: {proto.REPORT_HTML} ({proto.REPORT_HTML.stat().st_size / 1e6:.1f} MB)")
    if args.open:
        webbrowser.open(f"file://{proto.REPORT_HTML}")


if __name__ == "__main__":
    main()
