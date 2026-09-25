"""Análise exploratória dos dados LiDAR de vento (data/dataset.csv).

Dados LiDAR da campanha de medição de setembro a novembro de 2021, com perfil
vertical de 40 m a 260 m (velocidade, direção, velocidade vertical), indicadores
de qualidade do sinal (cis1–cis19), deslocamentos (disp/vdisp) e variáveis
meteorológicas (pressão, umidade, temperatura).

Gera figuras e um relatório HTML em data/eda/.

Executar a partir da raiz do projeto:
    python tests/eda_lidar.py

Saída:
    data/eda/images/*.png
    data/eda/relatorio_eda_lidar.html
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_FILE = PROJECT_ROOT / "data" / "dataset.csv"
OUT_DIR = PROJECT_ROOT / "data" / "eda"
IMG_DIR = OUT_DIR / "images"

HEIGHTS = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
           150, 160, 170, 180, 190, 200, 220, 240, 260]
HUB = 100  # altura de referência (hub height)
METEO = ["temp", "humid", "press"]

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.titlesize": 11,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.autolayout": True,
})


def ws_col(h):
    return f"ws{h}"


def v_col(h):
    return f"v{h}"


def dir_col(h):
    return f"dir{h}"


def disp_col(h):
    return f"disp{h}"


def vdisp_col(h):
    return f"vdisp{h}"


def height_norm():
    return plt.Normalize(min(HEIGHTS), max(HEIGHTS))


def height_color(h):
    return plt.cm.plasma(height_norm()(h))


def savefig(fig, name):
    fig.savefig(IMG_DIR / name, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 1. Visão geral: cobertura temporal e completude
# ----------------------------------------------------------------------------
def plot_overview(df, report):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    completeness = df[df.columns].notna().mean() * 100
    axes[0].barh(range(len(completeness)), completeness.values, color="#1b9e77")
    axes[0].set_yticks(range(len(completeness)))
    axes[0].set_yticklabels(completeness.index, fontsize=6.5)
    axes[0].set_xlim(0, 105)
    axes[0].set_xlabel("Completude (%)")
    axes[0].set_title("Completude por coluna")
    axes[0].invert_yaxis()

    daily_n = df.resample("D").size()
    axes[1].plot(daily_n.index, daily_n.values, lw=1, color="#d95f02")
    axes[1].set_ylabel("Amostras por dia")
    axes[1].set_title("Número de amostras por dia")
    axes[1].set_ylim(0, max(daily_n.max() * 1.1, 200))

    savefig(fig, "01_visao_geral.png")
    report["rows"] = int(len(df))
    report["n_heights"] = len(HEIGHTS)
    report["missing_total"] = int(df.isna().sum().sum())
    report["sampling_min"] = float(df.index.to_series().diff().dt.total_seconds().min() / 60)
    report["sampling_max"] = float(df.index.to_series().diff().dt.total_seconds().max() / 60)


# ----------------------------------------------------------------------------
# 2. Série temporal: vento + meteorologia
# ----------------------------------------------------------------------------
def _daily_gridlines(ax, df):
    daily = pd.date_range(
        df.index.min().normalize(),
        df.index.max().normalize() + pd.Timedelta("1D"),
        freq="D",
    )
    for d in daily:
        ax.axvline(d, color="#cccccc", lw=0.5, zorder=0)


def plot_time_series(df, report):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    ax = axes[0]
    daily_mean = df[[ws_col(h) for h in HEIGHTS]].resample("D").mean()
    for h in HEIGHTS:
        ax.plot(daily_mean.index, daily_mean[ws_col(h)], lw=0.9,
                color=height_color(h), label=f"{h} m")
    ax.set_ylabel("Velocidade média diária (m/s)")
    ax.set_title("Média diária da velocidade do vento por altura — campanha set/nov 2021")
    ax.legend(ncol=5, fontsize=7.5, loc="upper right")

    ax = axes[1]
    ax.plot(df.index, df[ws_col(HUB)], lw=0.4, color="#ff7f0e", alpha=0.8)
    ax.set_ylabel("ws100 (m/s)")
    ax.set_title("Velocidade a 100 m (amostras de 10 min)")

    for ax in axes:
        _daily_gridlines(ax, df)

    savefig(fig, "02_serie_temporal.png")
    report["ws100_mean"] = float(df[ws_col(HUB)].mean())
    report["ws100_min"] = float(df[ws_col(HUB)].min())
    report["ws100_max"] = float(df[ws_col(HUB)].max())
    report["temp_mean"] = float(df["temp"].mean())
    report["humid_mean"] = float(df["humid"].mean())
    report["press_mean"] = float(df["press"].mean())


def plot_meteo_series(df):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    ax = axes[0]
    ax.plot(df.index, df["temp"], lw=0.8, color="#d62728", label="Temperatura (°C)")
    ax.plot(df.index, df["humid"], lw=0.8, color="#1b9e77", label="Umidade (%)")
    ax.set_ylabel("Temp (°C) / Umid (%)")
    ax.set_title("Temperatura e umidade na torre")
    ax.legend(ncol=2)

    ax = axes[1]
    ax.plot(df.index, df["press"], lw=0.8, color="#7570b3")
    ax.set_ylabel("Pressão (hPa)")
    ax.set_title("Pressão atmosférica na torre")

    for ax in axes:
        _daily_gridlines(ax, df)

    savefig(fig, "02_meteo_torre.png")


def plot_vertical_series(df):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(df.index, df[v_col(HUB)], lw=0.5, color="#1b9e77", alpha=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("v100 (m/s)")
    ax.set_title("Componente vertical do vento a 100 m (amostras de 10 min)")
    _daily_gridlines(ax, df)
    savefig(fig, "02_componente_vertical.png")


def plot_direction_series(df):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.scatter(df.index, df[dir_col(HUB)], s=2, color="#d95f02", alpha=0.5)
    ax.set_ylabel("Direção (graus)")
    ax.set_title("Direção do vento a 100 m (amostras de 10 min)")
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_yticklabels(["0 (N)", "90 (L)", "180 (S)", "270 (O)", "360 (N)"])
    _daily_gridlines(ax, df)
    savefig(fig, "02_direcao.png")


def plot_day_heights(df):
    day = "2021-09-21"
    sub = df.loc[day]
    heights = HEIGHTS

    fig, ax = plt.subplots(figsize=(13, 6))
    for h in heights:
        ax.plot(sub.index, sub[ws_col(h)], lw=0.9, color=height_color(h), label=f"{h} m")
    ax.set_ylabel("Velocidade (m/s)")
    ax.set_title(f"Velocidade do vento por altura em {day} (6h–10h)")
    ax.legend(ncol=5, fontsize=7.5, loc="upper right")
    hours = pd.date_range(f"{day} 06:00", f"{day} 10:00", freq="10min")
    ax.set_xticks(hours)
    ax.set_xticklabels([t.strftime("%H:%M") for t in hours], rotation=45)
    ax.set_xlim(pd.Timestamp(f"{day} 06:00"), pd.Timestamp(f"{day} 10:00"))
    savefig(fig, "02b_dia_21_set.png")


# ----------------------------------------------------------------------------
# 3. Perfil vertical médio
# ----------------------------------------------------------------------------
def plot_vertical_profile(df, report):
    fig, ax = plt.subplots(figsize=(6, 6.5))
    means = [df[ws_col(h)].mean() for h in HEIGHTS]
    medians = [df[ws_col(h)].median() for h in HEIGHTS]
    stds = [df[ws_col(h)].std() for h in HEIGHTS]
    ax.errorbar(means, HEIGHTS, xerr=stds, fmt="o-", color="#1b9e77",
                capsize=3, label="Média ± 1σ")
    ax.plot(medians, HEIGHTS, "s--", color="#7570b3", label="Mediana")
    ax.set_xlabel("Velocidade do vento (m/s)")
    ax.set_ylabel("Altura (m)")
    ax.set_title("Perfil vertical médio do vento (LiDAR, 40–260 m)")
    ax.legend()
    savefig(fig, "03_perfil_vertical.png")
    report["profile"] = {h: round(df[ws_col(h)].mean(), 3) for h in HEIGHTS}
    report["ws40_mean"] = float(df[ws_col(40)].mean())
    report["ws260_mean"] = float(df[ws_col(260)].mean())


# ----------------------------------------------------------------------------
# 4. Boxplot por altura
# ----------------------------------------------------------------------------
def plot_boxplot(df):
    fig, ax = plt.subplots(figsize=(12, 5))
    data = [df[ws_col(h)].dropna().values for h in HEIGHTS]
    ax.boxplot(data, tick_labels=HEIGHTS, showfliers=False, patch_artist=True,
               medianprops=dict(color="black", lw=1))
    for patch, h in zip(ax.patches, HEIGHTS):
        patch.set_facecolor(height_color(h))
    ax.set_xlabel("Altura (m)")
    ax.set_ylabel("Velocidade do vento (m/s)")
    ax.set_title("Distribuição da velocidade do vento por altura")
    savefig(fig, "04_boxplot_alturas.png")


# ----------------------------------------------------------------------------
# 5. Histogramas por altura
# ----------------------------------------------------------------------------
def plot_histograms(df):
    n = len(HEIGHTS)
    fig, axes = plt.subplots(4, 5, figsize=(18, 10))
    fig.subplots_adjust(wspace=0.7, hspace=0.4)
    for ax, h in zip(axes.flat, HEIGHTS):
        vals = df[ws_col(h)].dropna().values
        ax.hist(vals, bins=50, color=height_color(h), alpha=0.85, density=True)
        ax.set_xlim(2, 17.5)
        ax.set_title(f"{h} m", fontsize=8)
        ax.set_yticks([])
        ax.text(0.97, 0.95, f"média={vals.mean():.2f}\nσ={vals.std():.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
    fig.suptitle("Histogramas da velocidade do vento por altura")
    savefig(fig, "05_histogramas.png")


# ----------------------------------------------------------------------------
# 6. Rosa dos ventos em 100 m
# ----------------------------------------------------------------------------
def plot_wind_rose(df, report):
    ws = df[ws_col(HUB)].values
    direction = df[dir_col(HUB)].values
    valid = ~np.isnan(ws) & ~np.isnan(direction)
    ws, direction = ws[valid], direction[valid]

    n_sectors = 16
    sector_edges = np.linspace(0, 360, n_sectors + 1)
    speed_bins = [0, 4, 8, 12, 16, np.inf]
    speed_labels = ["0–4", "4–8", "8–12", "12–16", ">16"]
    colors = ["#3288bd", "#66c2a5", "#fee08b", "#d53e4f", "#7b3294"]

    sector_idx = np.digitize(direction, sector_edges) - 1
    sector_idx[sector_idx == n_sectors] = 0
    speed_idx = np.digitize(ws, speed_bins) - 1
    speed_idx[speed_idx > len(speed_bins) - 2] = len(speed_bins) - 2

    counts = np.zeros((n_sectors, len(speed_bins) - 1))
    for s, sp in zip(sector_idx, speed_idx):
        counts[s, sp] += 1
    freq = counts / counts.sum() * 100

    theta = np.deg2rad(np.arange(11.25, 360, 22.5))
    width = np.deg2rad(22.5)
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    bottoms = np.zeros(n_sectors)
    for i, (lab, col) in enumerate(zip(speed_labels, colors)):
        ax.bar(theta, freq[:, i], width=width, bottom=bottoms, color=col,
               edgecolor="white", lw=0.4, label=f"{lab} m/s")
        bottoms += freq[:, i]
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels(["N", "NE", "L", "SE", "S", "SO", "O", "NO"])
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels([f"{t:.0f}%" for t in ax.get_yticks()], fontsize=7)
    ax.set_title(f"Rosa dos ventos a {HUB} m (frequência por setor de 22,5°)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.18, -0.15), ncol=3)

    savefig(fig, "06_rosa_dos_ventos.png")

    report["dir_mode_sector"] = int(np.argmax(counts.sum(axis=1)) * 22.5)
    report["pct_calm"] = float(np.mean(ws < 0.5) * 100)
    report["pct_strong"] = float(np.mean(ws > 12) * 100)


# ----------------------------------------------------------------------------
# 7. Velocidade vertical
# ----------------------------------------------------------------------------
def plot_vertical_velocity(df, report):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    data = [df[v_col(h)].dropna().values for h in HEIGHTS]
    axes[0].boxplot(data, tick_labels=HEIGHTS, showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=1))
    for patch, h in zip(axes[0].patches, HEIGHTS):
        patch.set_facecolor(height_color(h))
    axes[0].set_xlabel("Altura (m)")
    axes[0].set_ylabel("Velocidade vertical (m/s)")
    axes[0].set_title("Velocidade vertical por altura")

    v100 = df[v_col(HUB)]
    axes[1].plot(df.index, v100.rolling("24h", min_periods=1).mean(), lw=0.8, color="#7570b3")
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_ylabel("m/s")
    axes[1].set_title(f"Média móvel 24 h de v{HUB} (movimento vertical)")
    savefig(fig, "07_velocidade_vertical.png")
    report["v100_mean"] = float(v100.mean())
    report["v100_std"] = float(v100.std())
    report["pct_updraft"] = float(np.mean(v100 > 0) * 100)


# ----------------------------------------------------------------------------
# 8. Ciclo diurno (hora do dia x altura)
# ----------------------------------------------------------------------------
def plot_diurnal(df, report):
    hour = df.index.hour
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    pivot = pd.DataFrame({h: df[ws_col(h)].values for h in HEIGHTS}, index=hour)
    heat = pivot.groupby(pivot.index).mean()
    im = axes[0].imshow(heat.T.values, aspect="auto", origin="lower",
                        cmap="viridis", interpolation="nearest",
                        extent=[-0.5, 23.5, min(HEIGHTS) - 5, max(HEIGHTS) + 5])
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].set_xlabel("Hora do dia")
    axes[0].set_ylabel("Altura (m)")
    axes[0].set_title("Velocidade média por hora do dia e altura (m/s)")
    cb = fig.colorbar(im, ax=axes[0], pad=0.02)
    cb.set_label("m/s")

    for h in [40, 100, 200, 260]:
        axes[1].plot(heat[h], marker="o", ms=2.5, lw=1.2, label=f"{h} m")
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].set_xlabel("Hora do dia")
    axes[1].set_ylabel("Velocidade média (m/s)")
    axes[1].set_title("Ciclo diurno em alturas selecionadas")
    axes[1].legend()
    savefig(fig, "08_ciclo_diurno.png")

    hourly_mean = pivot.groupby(pivot.index).mean()[HUB]
    report["diurnal_amplitude"] = float(hourly_mean.max() - hourly_mean.min())
    report["hour_max_ws100"] = int(hourly_mean.idxmax())
    report["hour_min_ws100"] = int(hourly_mean.idxmin())


# ----------------------------------------------------------------------------
# 9. Variação mensal e diária
# ----------------------------------------------------------------------------
def plot_monthly(df, report):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    monthly = df.groupby(df.index.month)[ws_col(HUB)].mean()
    months = ["Set", "Out", "Nov"]
    monthly.index = [months[int(m) - 9] for m in monthly.index]
    axes[0].bar(range(len(monthly)), monthly.values, color="#d95f02")
    axes[0].set_xticks(range(len(monthly)))
    axes[0].set_xticklabels(monthly.index)
    axes[0].set_ylabel("Velocidade média (m/s)")
    axes[0].set_title(f"Velocidade média mensal a {HUB} m")

    daily = df[ws_col(HUB)].resample("D").mean()
    axes[1].plot(daily.index, daily.values, lw=1, color="#1b9e77")
    axes[1].set_ylabel("Velocidade média diária (m/s)")
    axes[1].set_title(f"Velocidade média diária a {HUB} m")
    savefig(fig, "09_variacao_mensal.png")
    report["month_mean"] = {str(k): round(v, 2) for k, v in monthly.items()}


# ----------------------------------------------------------------------------
# 10. Correlação entre alturas
# ----------------------------------------------------------------------------
def plot_correlation(df, report):
    cols = [ws_col(h) for h in HEIGHTS]
    corr = df[cols].corr()
    vmin, vmax = corr.values.min(), corr.values.max()
    pad = 0.01 * (vmax - vmin)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=vmin - pad, vmax=vmax + pad)
    ax.set_xticks(range(len(HEIGHTS)))
    ax.set_yticks(range(len(HEIGHTS)))
    ax.set_xticklabels(HEIGHTS, rotation=90, fontsize=7)
    ax.set_yticklabels(HEIGHTS, fontsize=7)
    ax.set_xlabel("Altura (m)")
    ax.set_ylabel("Altura (m)")
    ax.set_title("Correlação de Pearson entre velocidades por altura "
                 "(escala normalizada)")
    for i in range(len(HEIGHTS)):
        for j in range(len(HEIGHTS)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
    cb.set_label("r (escala normalizada)", fontsize=12)
    cb.ax.tick_params(labelsize=10)
    savefig(fig, "10_correlacao_alturas.png")

    report["corr_ws40_ws100"] = float(df[ws_col(40)].corr(df[ws_col(HUB)]))
    report["corr_ws40_ws260"] = float(df[ws_col(40)].corr(df[ws_col(260)]))


# ----------------------------------------------------------------------------
# 11. Scatter entre alturas
# ----------------------------------------------------------------------------
def plot_scatter_heights(df):
    pairs = [(40, 100), (40, 260), (100, 200)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (h1, h2) in zip(axes, pairs):
        sub = df[[ws_col(h1), ws_col(h2)]].dropna()
        x, y = sub[ws_col(h1)].values, sub[ws_col(h2)].values
        hb = ax.hexbin(x, y, gridsize=40, cmap="YlGnBu", mincnt=1)
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 50)
        ax.plot(xx, m * xx + b, "r-", lw=1.5, label=f"y = {m:.2f}x + {b:.2f}")
        ax.set_xlabel(f"ws{h1} (m/s)")
        ax.set_ylabel(f"ws{h2} (m/s)")
        ax.set_title(f"ws{h1} × ws{h2} (r = {np.corrcoef(x, y)[0, 1]:.3f})")
        ax.legend(loc="lower right")
        fig.colorbar(hb, ax=ax, pad=0.01).set_label("contagem")
    fig.suptitle("Relação entre velocidades em alturas diferentes (hexbin)")
    savefig(fig, "11_scatter_alturas.png")


# ----------------------------------------------------------------------------
# 12. Expoente de shear (lei de potência)
# ----------------------------------------------------------------------------
def plot_shear(df, report):
    ws40 = df[ws_col(40)]
    ws260 = df[ws_col(260)]
    alpha = np.log(ws260 / ws40) / np.log(260 / 40)
    alpha_valid = alpha.replace([np.inf, -np.inf], np.nan).dropna()
    alpha_valid = alpha_valid[(alpha_valid > -1) & (alpha_valid < 1.5)]

    fig = plt.figure(figsize=(13, 7.5))
    axes = [fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2),
            fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4, projection="polar")]
    axes = np.array(axes).reshape(2, 2)
    axes[0, 0].plot(alpha_valid.index, alpha_valid.rolling("24h", min_periods=1).mean(),
                    lw=0.7, color="#1b9e77")
    axes[0, 0].axhline(1 / 7, color="r", ls="--", lw=1, label="α = 1/7 (superfície padrão)")
    axes[0, 0].set_ylabel("α")
    axes[0, 0].set_title("Expoente de shear (média móvel 24 h), ws40→ws260")
    axes[0, 0].legend()

    axes[0, 1].hist(alpha_valid.values, bins=70, color="#1b9e77", alpha=0.85, density=True)
    axes[0, 1].axvline(1 / 7, color="r", ls="--", lw=1)
    axes[0, 1].set_xlabel("α")
    axes[0, 1].set_title(f"Distribuição de α (média={alpha_valid.mean():.3f}, σ={alpha_valid.std():.3f})")

    dir_sec = ((df[dir_col(HUB)] // 22.5).astype("float64") % 16)
    by_dir = pd.DataFrame({"alpha": alpha_valid, "dir": dir_sec}).dropna()
    by_dir["dir"] = by_dir["dir"].astype(int)
    dir_mean = by_dir.groupby("dir")["alpha"].mean().reindex(range(16), fill_value=0)
    theta = np.deg2rad(np.arange(11.25, 360, 22.5))
    axes[1, 1].bar(theta, dir_mean.values, width=np.deg2rad(22.5), color="#7570b3",
                   edgecolor="white")
    axes[1, 1].set_theta_zero_location("N")
    axes[1, 1].set_theta_direction(-1)
    axes[1, 1].set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    axes[1, 1].set_xticklabels(["N", "NE", "L", "SE", "S", "SO", "O", "NO"])
    axes[1, 1].set_title("α médio por setor de direção")

    hourly = alpha_valid.groupby(alpha_valid.index.hour).mean()
    axes[1, 0].plot(hourly.index, hourly.values, "o-", color="#d95f02", ms=3)
    axes[1, 0].set_xticks(range(0, 24, 2))
    axes[1, 0].set_xlabel("Hora do dia")
    axes[1, 0].set_ylabel("α")
    axes[1, 0].set_title("Expoente de shear médio por hora")
    savefig(fig, "12_expoente_shear.png")

    report["alpha_mean"] = float(alpha_valid.mean())
    report["alpha_std"] = float(alpha_valid.std())
    report["alpha_pct_negative"] = float(np.mean(alpha_valid < 0) * 100)


# ----------------------------------------------------------------------------
# 13. Direção vs velocidade e movimento vertical
# ----------------------------------------------------------------------------
def plot_dir_speed_vertical(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sub = df[[dir_col(HUB), ws_col(HUB), v_col(HUB)]].dropna()
    hb = axes[0].hexbin(sub[ws_col(HUB)], sub[dir_col(HUB)], gridsize=50,
                        cmap="viridis", mincnt=1)
    axes[0].set_xlabel(f"Velocidade a {HUB} m (m/s)")
    axes[0].set_ylabel(f"Direção a {HUB} m (graus)")
    axes[0].set_title("Distribuição conjunta velocidade × direção")
    fig.colorbar(hb, ax=axes[0], pad=0.02).set_label("contagem")

    hb2 = axes[1].hexbin(sub[ws_col(HUB)], sub[v_col(HUB)], gridsize=50,
                         cmap="plasma", mincnt=1)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel(f"Velocidade horizontal a {HUB} m (m/s)")
    axes[1].set_ylabel(f"Velocidade vertical a {HUB} m (m/s)")
    axes[1].set_title("Movimento vertical vs velocidade horizontal")
    fig.colorbar(hb2, ax=axes[1], pad=0.02).set_label("contagem")
    savefig(fig, "13_direcao_velocidade_vertical.png")


def plot_dir_vertical_corr(df, report):
    sub = df[[dir_col(HUB), v_col(HUB)]].dropna()
    dirs = sub[dir_col(HUB)].values
    v = sub[v_col(HUB)].values
    theta = np.deg2rad(dirs)
    n = len(sub)
    r_sin, p_sin = stats.pearsonr(np.sin(theta), v)
    r_cos, p_cos = stats.pearsonr(np.cos(theta), v)

    fig = plt.figure(figsize=(13, 5))
    ax_cart = fig.add_subplot(1, 2, 1)
    ax_pol = fig.add_subplot(1, 2, 2, projection="polar")
    hb = ax_cart.hexbin(dirs, v, gridsize=50, cmap="plasma", mincnt=1)
    ax_cart.axhline(0, color="k", lw=0.5)
    ax_cart.set_xlabel(f"Direção a {HUB} m (graus)")
    ax_cart.set_ylabel(f"Velocidade vertical a {HUB} m (m/s)")
    ax_cart.set_title("Componente vertical vs direção (hexbin)")
    fig.colorbar(hb, ax=ax_cart, pad=0.02).set_label("contagem")

    sec = ((dirs // 22.5).astype("float64") % 16).astype(int)
    sector_theta = np.deg2rad(np.arange(11.25, 360, 22.5))
    m = np.array([v[sec == i].mean() for i in range(16)])
    m = np.nan_to_num(m, nan=0.0)
    ax_pol.bar(sector_theta, m, width=np.deg2rad(22.5), color="#1b9e77", edgecolor="white")
    ax_pol.set_theta_zero_location("N")
    ax_pol.set_theta_direction(-1)
    ax_pol.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax_pol.set_xticklabels(["N", "NE", "L", "SE", "S", "SO", "O", "NO"])
    ax_pol.set_title("Velocidade vertical média por setor de direção")

    report["corr_dir100_v100_sin"] = float(r_sin)
    report["corr_dir100_v100_cos"] = float(r_cos)
    savefig(fig, "13b_direcao_vertical_corr.png")


def plot_dir_sin_cos_corr(df, report):
    sub = df[[dir_col(HUB), v_col(HUB)]].dropna()
    dirs = sub[dir_col(HUB)].values
    v = sub[v_col(HUB)].values
    theta = np.deg2rad(dirs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pairs = [(np.sin(theta), "sin(direção)", "corr_dir100_v100_sin"),
             (np.cos(theta), "cos(direção)", "corr_dir100_v100_cos")]
    for ax, (x, lab, key) in zip(axes, pairs):
        ax.scatter(x, v, s=1.5, alpha=0.3, color="#1b9e77")
        m, b = np.polyfit(x, v, 1)
        xx = np.linspace(x.min(), x.max(), 50)
        ax.plot(xx, m * xx + b, "r-", lw=1.2)
        r = float(np.corrcoef(x, v)[0, 1])
        ax.set_xlabel(lab)
        ax.set_ylabel("v100 (m/s)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"v100 × {lab} (r = {r:.3f})")
        report[key] = r
    savefig(fig, "13d_dir_sin_cos_corr.png")


def plot_ws_vert_dir_corr(df, report):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pairs = [("v", "Velocidade vertical (m/s)"), ("dir", "Direção (graus)")]
    for ax, (col, lab) in zip(axes, pairs):
        sub = df[[ws_col(HUB), col + str(HUB)]].dropna()
        x = sub[col + str(HUB)]
        y = sub[ws_col(HUB)]
        ax.scatter(x, y, s=1.5, alpha=0.3, color="#1b9e77")
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 50)
        ax.plot(xx, m * xx + b, "r-", lw=1.2)
        ax.set_xlabel(lab)
        ax.set_ylabel("ws100 (m/s)")
        r = x.corr(y)
        ax.set_title(f"ws100 × {lab} (r = {r:.3f})")
        if col == "v":
            report["corr_ws100_v100"] = float(r)
        else:
            report["corr_ws100_dir100"] = float(r)
    savefig(fig, "13c_ws_vert_dir_corr.png")


# ----------------------------------------------------------------------------
# 14. Ajuste de Weibull
# ----------------------------------------------------------------------------
def plot_weibull(df, report):
    ws = df[ws_col(HUB)].dropna().values
    ws = ws[(ws > 0)]
    shape, loc, scale = stats.weibull_min.fit(ws, floc=0)
    k, A = shape, scale

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ws, bins=60, density=True, alpha=0.75, color="#1b9e77", label="Dados")
    x = np.linspace(0.01, ws.max(), 300)
    pdf = stats.weibull_min.pdf(x, k, loc=0, scale=A)
    ax.plot(x, pdf, "r-", lw=2, label=f"Weibull (k={k:.2f}, A={A:.2f})")
    ax.axvline(ws.mean(), color="k", ls="--", lw=1, label=f"Média observada = {ws.mean():.2f}")
    weibull_mean = A * math.gamma(1 + 1 / k)
    ax.axvline(weibull_mean, color="b", ls=":", lw=1.5,
               label=f"Média Weibull = {weibull_mean:.2f}")
    ax.set_xlabel(f"Velocidade a {HUB} m (m/s)")
    ax.set_ylabel("Densidade de probabilidade")
    ax.set_title(f"Ajuste de Weibull — velocidade a {HUB} m")
    ax.legend()
    savefig(fig, "14_weibull.png")
    report["weibull_k"] = float(k)
    report["weibull_A"] = float(A)
    report["weibull_mean"] = float(weibull_mean)


# ----------------------------------------------------------------------------
# 15. Autocorrelação (persistência)
# ----------------------------------------------------------------------------
def _acf(values, max_lag):
    values = values - np.nanmean(values)
    valid = ~np.isnan(values)
    values = values[valid]
    return np.array([np.corrcoef(values[:-l], values[l:])[0, 1]
                     for l in range(1, max_lag + 1)])


def _pacf(values, max_lag):
    values = values - np.nanmean(values)
    valid = ~np.isnan(values)
    values = values[valid]
    pacf = np.empty(max_lag)
    n = len(values)
    for lag in range(1, max_lag + 1):
        x = np.empty((n - lag, lag))
        for k in range(lag):
            x[:, k] = values[lag - 1 - k:n - 1 - k]
        y = values[lag:]
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        pacf[lag - 1] = coef[-1]
    return pacf


def plot_autocorrelation(df, report):
    max_lag = 144  # 24 horas
    lags_h = np.arange(1, max_lag + 1) / 6
    height_step = 20
    heights = [h for h in HEIGHTS if (h - min(HEIGHTS)) % height_step == 0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8),
                             sharex=True, gridspec_kw={"hspace": 0.12})

    ax = axes[0]
    acf = _acf(df[ws_col(HUB)].values, max_lag)
    ax.bar(lags_h, acf, width=1 / 6, color="#2c7fb8", alpha=0.9)
    ax.set_ylabel("Autocorrelação de ws100")
    ax.set_title("Autocorrelação da velocidade do vento a 100 m (persistência)")

    ax = axes[1]
    for h in heights:
        acf_h = _acf(df[ws_col(h)].values, max_lag)
        ax.plot(lags_h, acf_h, lw=1.2, color=height_color(h),
                label=f"{h} m")
    ax.set_xlabel("Defasagem (horas)")
    ax.set_ylabel("Autocorrelação")
    ax.set_title("Autocorrelação da velocidade do vento por altura (intervalos de 20 m)")
    ax.legend(ncol=5, loc="upper right", fontsize=7.5)

    for ax in axes:
        ax.axhline(0, color="k", lw=0.7)
        for p in [0.5, 1, 2, 4, 8, 24]:
            ax.axvline(p, color="gray", ls="--", lw=0.6)
    axes[1].set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])

    savefig(fig, "15_autocorrelacao.png")
    for lag_h in [1, 6, 24]:
        report[f"acf_{int(lag_h)}h"] = float(acf[int(lag_h * 6) - 1])


def plot_pacf(df, report):
    max_lag = 144  # 24 horas
    lags_h = np.arange(1, max_lag + 1) / 6
    height_step = 20
    heights = [h for h in HEIGHTS if (h - min(HEIGHTS)) % height_step == 0]
    ws = df[ws_col(HUB)].values
    ci = 1.96 / np.sqrt((~np.isnan(ws)).sum())

    fig, axes = plt.subplots(2, 1, figsize=(10, 8),
                             sharex=True, gridspec_kw={"hspace": 0.12})

    ax = axes[0]
    pacf = _pacf(ws, max_lag)
    ax.bar(lags_h, pacf, width=1 / 6, color="#2c7fb8", alpha=0.9)
    ax.set_ylabel("PACF de ws100")
    ax.set_title("Autocorrelação parcial (PACF) da velocidade do vento a 100 m")

    ax = axes[1]
    for h in heights:
        pacf_h = _pacf(df[ws_col(h)].values, max_lag)
        ax.plot(lags_h, pacf_h, lw=1.2, color=height_color(h),
                label=f"{h} m")
    ax.set_xlabel("Defasagem (horas)")
    ax.set_ylabel("PACF")
    ax.set_title("PACF da velocidade do vento por altura (intervalos de 20 m)")
    ax.legend(ncol=5, loc="upper right", fontsize=7.5)

    for ax in axes:
        ax.axhline(0, color="k", lw=0.7)
        ax.axhspan(-ci, ci, color="gray", alpha=0.15, lw=0)
        for p in [0.5, 1, 2, 4, 8, 24]:
            ax.axvline(p, color="gray", ls="--", lw=0.6)
    axes[1].set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])

    savefig(fig, "15b_pacf.png")


def plot_smoothed_acf_pacf(df, report):
    max_lag_h = 24
    lags_h = np.arange(1, max_lag_h + 1)
    height_step = 20
    heights = [h for h in HEIGHTS if (h - min(HEIGHTS)) % height_step == 0]

    series = df[ws_col(HUB)].resample("h").mean().dropna()
    acf = _acf(series.values, max_lag_h)
    pacf = _pacf(series.values, max_lag_h)
    ci = 1.96 / np.sqrt(len(series))

    fig, axes = plt.subplots(3, 1, figsize=(10, 12),
                             sharex=True, gridspec_kw={"hspace": 0.18})

    panels = [
        (axes[0], acf, "Autocorrelação de ws100 (horária)",
         "ACF de ws100 em passos de 1 h (t vs t+1h, t+2h, ...)"),
        (axes[1], pacf, "PACF de ws100 (horária)",
         "PACF de ws100 em passos de 1 h"),
    ]
    for ax, values, ylab, title in panels:
        ax.bar(lags_h, values, width=0.9, color="#2c7fb8", alpha=0.9, zorder=2)
        ax.axhline(0, color="k", lw=0.7, zorder=1)
        ax.axhspan(-ci, ci, color="gray", alpha=0.15, lw=0, zorder=1)
        for p in [1, 2, 3, 4, 6, 8, 12, 24]:
            ax.axvline(p, color="gray", ls="--", lw=0.6, zorder=1)
        ax.set_ylabel(ylab)
        ax.set_title(title)

    ax = axes[2]
    for h in heights:
        s_h = df[ws_col(h)].resample("h").mean().dropna()
        pacf_h = _pacf(s_h.values, max_lag_h)
        ax.plot(lags_h, pacf_h, lw=1.2, color=height_color(h), label=f"{h} m")
    ax.axhline(0, color="k", lw=0.7, zorder=1)
    ax.axhspan(-ci, ci, color="gray", alpha=0.15, lw=0, zorder=1)
    for p in [1, 2, 3, 4, 6, 8, 12, 24]:
        ax.axvline(p, color="gray", ls="--", lw=0.6, zorder=1)
    ax.set_ylabel("PACF")
    ax.set_title("PACF horária da velocidade do vento por altura (intervalos de 20 m)")
    ax.legend(ncol=5, loc="upper right", fontsize=7.5)
    ax.set_xlabel("Defasagem (horas)")
    ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])

    report["acf_smooth_1h"] = float(acf[1 - 1])
    report["acf_smooth_6h"] = float(acf[6 - 1])
    report["pacf_smooth_6h"] = float(pacf[6 - 1])

    savefig(fig, "15d_smoothed_acf_pacf.png")


def plot_hourly_pacf(df, report):
    max_lag_h = 24
    lags_h = np.arange(1, max_lag_h + 1)

    series = df[ws_col(HUB)].resample("h").mean().dropna()
    pacf = _pacf(series.values, max_lag_h)
    ci = 1.96 / np.sqrt(len(series))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(lags_h, pacf, width=0.9, color="#2c7fb8", alpha=0.9, zorder=2)
    ax.axhline(0, color="k", lw=0.7, zorder=1)
    ax.axhspan(-ci, ci, color="gray", alpha=0.15, lw=0, zorder=1)
    for p in [1, 2, 3, 4, 6, 8, 12, 24]:
        ax.axvline(p, color="gray", ls="--", lw=0.6, zorder=1)
    ax.set_ylabel("PACF de ws100 (horária)")
    ax.set_xlabel("Defasagem (horas)")
    ax.set_title("PACF de ws100 em passos de 1 h (t vs t+1h, t+2h, ...)")
    ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])

    savefig(fig, "15e_pacf_horaria.png")


def plot_diurnal_acf(df, report):
    max_lag_h = 24
    series = df[ws_col(HUB)].resample("h").mean().dropna()
    hour = series.index.hour
    vals = series.values

    hours = np.arange(24)
    acf = np.full((len(hours), max_lag_h), np.nan)
    for i, h in enumerate(hours):
        mask = np.where(hour == h)[0]
        for l in range(1, max_lag_h + 1):
            now_i = mask[mask >= l]
            prev_i = now_i - l
            if len(now_i) < 2:
                continue
            acf[i, l - 1] = np.corrcoef(vals[now_i], vals[prev_i])[0, 1]

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(acf.T, aspect="auto", origin="lower", cmap="viridis",
                   extent=[-0.5, 23.5, 0.5, max_lag_h + 0.5])
    ax.set_xticks(range(0, 24, 2))
    ax.set_yticks(range(1, max_lag_h + 1, 3))
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("Defasagem (horas)")
    ax.set_title("Autocorrelação de ws100 por hora do dia e defasagem")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Correlação")
    savefig(fig, "15c_diurnal_acf.png")


# ----------------------------------------------------------------------------
# 16. Meteorologia × vento
# ----------------------------------------------------------------------------
def plot_meteo(df, report):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    pairs = [("temp", "Temperatura (°C)"), ("humid", "Umidade (%)"),
             ("press", "Pressão (hPa)")]
    for ax, (col, lab) in zip(axes.flat[:3], pairs):
        ax.scatter(df[col], df[ws_col(HUB)], s=1.5, alpha=0.3, color="#1b9e77")
        m, b = np.polyfit(df[col], df[ws_col(HUB)], 1)
        xx = np.linspace(df[col].min(), df[col].max(), 50)
        ax.plot(xx, m * xx + b, "r-", lw=1.2)
        ax.set_xlabel(lab)
        ax.set_ylabel("ws100 (m/s)")
        r = df[col].corr(df[ws_col(HUB)])
        ax.set_title(f"ws100 × {lab} (r = {r:.3f})")
        report[f"corr_ws100_{col}"] = float(r)

    corr = df[["temp", "humid", "press"] + [ws_col(h) for h in HEIGHTS]].corr()
    sub = corr.loc[["temp", "humid", "press"], [ws_col(h) for h in HEIGHTS]]
    im = axes[1, 1].imshow(sub.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[1, 1].set_xticks(range(len(HEIGHTS)))
    axes[1, 1].set_xticklabels(HEIGHTS, rotation=90, fontsize=7)
    axes[1, 1].set_yticks(range(3))
    axes[1, 1].set_yticklabels(["temp", "humid", "press"])
    axes[1, 1].set_title("Correlação meteorologia × ws (por altura)")
    for i in range(3):
        for j in range(len(HEIGHTS)):
            axes[1, 1].text(j, i, f"{sub.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=axes[1, 1], pad=0.02).set_label("r")
    savefig(fig, "16_meteorologia.png")


# ----------------------------------------------------------------------------
# 17. Qualidade do sinal LiDAR (CIS)
# ----------------------------------------------------------------------------
def plot_cis(df, report):
    cis_cols = [f"cis{i}" for i in range(1, 20)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    data = [df[c].dropna().values for c in cis_cols]
    axes[0].boxplot(data, tick_labels=range(1, 20), showfliers=False,
                    patch_artist=True, medianprops=dict(color="black", lw=1))
    for patch in axes[0].patches:
        patch.set_facecolor("#a6cee3")
    axes[0].set_xlabel("Gate (cis1–cis19)")
    axes[0].set_ylabel("Carrier-to-noise (CIS)")
    axes[0].set_title("Qualidade do sinal LiDAR por gate")

    cis_mean = df[cis_cols].mean(axis=1)
    axes[1].plot(df.index, cis_mean.rolling("24h", min_periods=1).mean(), lw=0.8, color="#1b9e77")
    axes[1].set_ylabel("CIS médio")
    axes[1].set_title("Qualidade média do sinal ao longo da campanha")
    savefig(fig, "17_cis_qualidade_sinal.png")

    low_q = np.mean(cis_mean < df[cis_cols].quantile(0.1).mean()) * 100
    report["cis_mean_overall"] = float(cis_mean.mean())
    report["cis_pct_low"] = float(low_q)


# ----------------------------------------------------------------------------
# 18. Deslocamentos do LiDAR (disp/vdisp)
# ----------------------------------------------------------------------------
def plot_disp(df, report):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    disp = [df[disp_col(h)].mean() for h in HEIGHTS]
    vdisp = [df[vdisp_col(h)].mean() for h in HEIGHTS]
    axes[0].plot(disp, HEIGHTS, "o-", color="#1b9e77", label="disp")
    axes[0].plot(vdisp, HEIGHTS, "s--", color="#7570b3", label="vdisp")
    axes[0].set_ylabel("Altura (m)")
    axes[0].set_xlabel("Deslocamento médio")
    axes[0].set_title("Perfil vertical de disp/vdisp")
    axes[0].legend()

    r = df[disp_col(HUB)].corr(df[ws_col(HUB)])
    axes[1].scatter(df[disp_col(HUB)], df[ws_col(HUB)], s=1.5, alpha=0.3, color="#d95f02")
    axes[1].set_xlabel(f"disp{HUB}")
    axes[1].set_ylabel(f"ws{HUB} (m/s)")
    axes[1].set_title(f"disp{HUB} × ws{HUB} (r = {r:.3f})")
    savefig(fig, "18_deslocamentos.png")
    report["corr_disp100_ws100"] = float(r)


# ----------------------------------------------------------------------------
# 19. Heatmap perfil vertical ao longo do tempo
# ----------------------------------------------------------------------------
def plot_profile_heatmap(df):
    hourly = df[["ws" + str(h) for h in HEIGHTS]].resample("h").mean().T
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(hourly.values, aspect="auto", origin="lower", cmap="viridis",
                   interpolation="nearest", extent=[0, len(hourly.columns), min(HEIGHTS) - 5, max(HEIGHTS) + 5])
    ax.set_xlabel("Hora desde o início da campanha")
    ax.set_ylabel("Altura (m)")
    ax.set_title("Perfil vertical da velocidade do vento ao longo do tempo (m/s)")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("m/s")
    ticks = np.linspace(0, len(hourly.columns) - 1, 6).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([hourly.columns[t].strftime("%d/%m") for t in ticks])
    savefig(fig, "19_perfil_temporal.png")


# ----------------------------------------------------------------------------
# 20. Tabela de estatísticas (imagem)
# ----------------------------------------------------------------------------
def plot_summary_table(df):
    cols = [ws_col(h) for h in HEIGHTS] + [v_col(HUB), dir_col(HUB)] + METEO
    names = [f"ws{h}" for h in HEIGHTS] + ["v100", "dir100", "temp", "humid", "press"]
    summary = df[cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    summary = summary.round(2)

    fig, ax = plt.subplots(figsize=(10, len(summary) * 0.42 + 0.8))
    ax.axis("off")
    tbl = ax.table(cellText=summary.values, rowLabels=names,
                   colLabels=["média", "σ", "mín", "Q25", "Q50", "Q75", "máx"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    ax.set_title("Estatísticas descritivas (m/s; direção em graus; temp °C; umid %; press hPa)")
    savefig(fig, "20_tabela_estatisticas.png")


# ----------------------------------------------------------------------------
# Relatório HTML
# ----------------------------------------------------------------------------
CSS = """
:root{--accent:#1b6ca8;--bg:#f4f6f9;--card:#ffffff;}
*{box-sizing:border-box}
body{margin:0;font-family:'Segoe UI',system-ui,Arial,sans-serif;background:var(--bg);color:#222}
header{background:linear-gradient(135deg,#0f3d5e,#1b6ca8);color:#fff;padding:28px 32px}
header h1{margin:0 0 6px;font-size:24px}
header p{margin:0;opacity:.9;font-size:14px}
.container{max-width:1100px;margin:0 auto;padding:24px 20px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0}
.card{background:var(--card);border:1px solid #e3e8ee;border-radius:10px;padding:14px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.card .num{font-size:22px;font-weight:700;color:var(--accent)}
.card .lab{font-size:12px;color:#555;margin-top:2px}
section{background:var(--card);border:1px solid #e3e8ee;border-radius:10px;margin:22px 0;padding:20px 24px;box-shadow:0 1px 3px rgba(0,0,0,.06)}
section h2{margin:0 0 4px;font-size:18px;color:#0f3d5e;border-bottom:2px solid #e8eef4;padding-bottom:8px}
section .sub{color:#666;font-size:13px;margin:8px 0 14px;line-height:1.5}
section img{max-width:100%;height:auto;border:1px solid #e3e8ee;border-radius:8px;display:block;margin:12px auto}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
th,td{border:1px solid #dfe5ec;padding:5px 9px;text-align:right}
th{background:#eef3f8;color:#0f3d5e}
td:first-child{text-align:left;font-weight:600}
.btn-copy{background:#1b6ca8;color:#fff;border:none;border-radius:6px;padding:6px 12px;font-size:12px;cursor:pointer;margin:8px 0 0;font-family:inherit}
.btn-copy:hover{background:#0f3d5e}
.btn-copy:active{opacity:.8}
.hl{background:#fff7e0;padding:10px 14px;border-left:4px solid #e0a800;border-radius:0 6px 6px 0;font-size:13px;line-height:1.5}
section ul{margin:8px 0;padding-left:20px}
section li{margin:7px 0;font-size:13px;line-height:1.55;color:#333}
section li b{color:#0f3d5e}
section p{font-size:13px;line-height:1.6;color:#333}
footer{color:#888;text-align:center;font-size:12px;padding:18px}
@media(max-width:760px){.two{grid-template-columns:1fr}}
"""


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}f}".replace(".", ",")


def card(num, lab):
    return f'<div class="card"><div class="num">{num}</div><div class="lab">{lab}</div></div>'


def section(title, sub, body, images=()):
    imgs = "".join(f'<img src="{p}" alt="{title}">' for p in images)
    return (f"<section><h2>{title}</h2><div class='sub'>{sub}</div>"
            f"{body}{imgs}</section>")


def latex_button(latex_source):
    """Return a 'copiar como LaTeX' button carrying the LaTeX source."""
    esc = (latex_source.replace("&", "&amp;")
            .replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;"))
    return (
        '<button type="button" class="btn-copy" data-latex="'
        f'{esc}" onclick="copiarLatex(this)">copiar como LaTeX</button>'
    )


def latex_table(columns, rows):
    """Build (html_table, latex_source) from column headers and row tuples."""
    head_html = "".join(f"<th>{c}</th>" for c in columns)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    html_table = f"<table><tr>{head_html}</tr>{body_html}</table>"

    latex = "\\begin{table}[h]\n\\centering\n"
    latex += "\\begin{tabular}{l" + "c" * (len(columns) - 1) + "}\n"
    latex += " & ".join(columns) + " \\\\ \\hline\n"
    for row in rows:
        latex += " & ".join(str(cell) for cell in row) + " \\\\\n"
    latex += "\\hline\n\\end{tabular}\n\\caption{...}\n\\end{table}"
    return html_table, latex


def build_html(report):
    n_days = (report["end"] - report["start"]).days
    rows_str = f"{report['rows']:,}".replace(",", ".")
    cards = "".join([
        card(f"{n_days}", "dias de dados"),
        card(rows_str, "amostras (10 min)"),
        card(f"{report['n_heights']}", f"alturas medidas ({min(HEIGHTS)}–{max(HEIGHTS)} m)"),
        card(f"{fmt(report['ws100_mean'])} m/s", "velocidade média a 100 m"),
        card(f"{fmt(report['ws100_max'])} m/s", "rajada máx. a 100 m"),
        card(f"{fmt(report['temp_mean'])} °C", "temperatura média"),
        card(f"k = {fmt(report['weibull_k'])}", "parâmetro de forma Weibull"),
        card(f"{fmt(report['alpha_mean'])}", "expoente de shear médio"),
    ])

    profile_rows = [
        (f"{h} m", f"{fmt(report['profile'][h])} m/s")
        for h in HEIGHTS
    ]
    profile_table_html, profile_latex = latex_table(
        ["Altura", "Velocidade média"], profile_rows
    )
    profile_table = latex_button(profile_latex) + profile_table_html

    month_rows = [
        (k, f"{v} m/s") for k, v in report["month_mean"].items()
    ]
    month_table_html, month_latex = latex_table(
        ["Mês", "Velocidade média a 100 m"], month_rows
    )
    month_table = latex_button(month_latex) + month_table_html

    missing_str = f"{report['missing_total']:,}".replace(",", ".")
    parts = []
    parts.append(section(
        "Visão geral e qualidade dos dados",
        (f"Dados de um perfilador LiDAR com medições a cada 10 min nas alturas "
         f"{min(HEIGHTS)}–{max(HEIGHTS)} m (velocidade horizontal, vertical, direção, "
         f"qualidade do sinal e deslocamentos), complementados por meteorologia "
         f"(temperatura, umidade e pressão). Período de <b>{report['start']:%d/%m/%Y}</b> "
         f"a <b>{report['end']:%d/%m/%Y}</b>, com <b>{missing_str}</b> células ausentes "
         "(conjunto já preenchido, sem lacunas)."),
        '<div class="hl">A velocidade média a 100 m é de '
        f"<b>{fmt(report['ws100_mean'])} m/s</b>, variando de {fmt(report['ws100_min'])} "
        f"a {fmt(report['ws100_max'])} m/s. A campanha cobre apenas os meses de "
        "setembro a novembro de 2021.</div>",
        images=["images/01_visao_geral.png", "images/02_serie_temporal.png",
                "images/02_meteo_torre.png", "images/02_componente_vertical.png",
                "images/02_direcao.png", "images/02b_dia_21_set.png"],
    ))

    parts.append(section(
        "Perfil vertical do vento",
        "O vento cresce com a altura ao longo de todo o perfil LiDAR (40–260 m). "
        "O efeito do cisalhamento (shear) é o que motiva o uso de perfiladores LiDAR "
        "em vez de anemômetros de ponto único.",
        profile_table,
        images=["images/03_perfil_vertical.png", "images/04_boxplot_alturas.png"],
    ))

    parts.append(section(
        "Distribuição das velocidades",
        "As distribuições por altura são assimétricas à direita, características do "
        "vento, e deslocam-se para velocidades maiores com a altura.",
        "",
        images=["images/05_histogramas.png"],
    ))

    parts.append(section(
        "Rosa dos ventos e direção",
        (f"Direção dominante no setor {report['dir_mode_sector']}–"
         f"{report['dir_mode_sector'] + 22.5}° (medido a partir do Norte). "
         f"Ventos calmos (&lt; 0,5 m/s) representam {fmt(report['pct_calm'])} % "
         f"e ventos fortes (&gt; 12 m/s) {fmt(report['pct_strong'])} % das amostras."),
        "",
        images=["images/06_rosa_dos_ventos.png", "images/13_direcao_velocidade_vertical.png",
                "images/13b_direcao_vertical_corr.png", "images/13d_dir_sin_cos_corr.png"],
    ))

    parts.append(section(
        "Velocidade vertical",
        (f"O movimento vertical médio a 100 m é de {fmt(report['v100_mean'])} m/s "
         f"(σ = {fmt(report['v100_std'])} m/s), com {fmt(report['pct_updraft'])} % "
         "das amostras em ascensão. Valores próximos de zero indicam vento "
         "predominantemente horizontal, com bolhas de convecção ocasionais. "
         f"A correlação da velocidade com a componente vertical é r = "
         f"{fmt(report['corr_ws100_v100'])} e com a direção é r = "
         f"{fmt(report['corr_ws100_dir100'])}."),
        "",
        images=["images/07_velocidade_vertical.png", "images/13c_ws_vert_dir_corr.png"],
    ))

    parts.append(section(
        "Ciclo diurno",
        (f"A amplitude do ciclo diurno a 100 m é de {fmt(report['diurnal_amplitude'])} m/s, "
         f"com máximas por volta das {report['hour_max_ws100']}:00 e mínimas perto das "
         f"{report['hour_min_ws100']}:00 (hora local)."),
        "",
        images=["images/08_ciclo_diurno.png"],
    ))

    parts.append(section(
        "Variação mensal e diária",
        "A campanha tem apenas ~2 meses e meio, então a 'sazonalidade' mensal resume-se "
        "aos meses de setembro, outubro e novembro de 2021.",
        month_table,
        images=["images/09_variacao_mensal.png"],
    ))

    parts.append(section(
        "Correlação entre alturas",
        (f"As velocidades nas diferentes alturas são fortemente correlacionadas: "
         f"r = {fmt(report['corr_ws40_ws100'])} entre 40 m e 100 m e "
         f"r = {fmt(report['corr_ws40_ws260'])} entre 40 m e 260 m."),
        "",
        images=["images/10_correlacao_alturas.png", "images/11_scatter_alturas.png"],
    ))

    parts.append(section(
        "Cisalhamento (expoente de shear)",
        (f"O expoente de shear médio α (lei de potência, ws40→ws260) é "
         f"<b>{fmt(report['alpha_mean'])}</b> (σ = {fmt(report['alpha_std'])}), "
         f"próximo do valor padrão de superfície 1/7 ≈ 0,143. Em "
         f"{fmt(report['alpha_pct_negative'])} % das amostras o shear é negativo "
         "(vento mais forte em baixo — situações de inversão ou jatos de baixo nível)."),
        "",
        images=["images/12_expoente_shear.png"],
    ))

    parts.append(section(
        "Ajuste de Weibull",
        (f"A distribuição de velocidade a 100 m é bem descrita por uma Weibull com "
         f"parâmetro de forma k = <b>{fmt(report['weibull_k'])}</b> e escala "
         f"A = <b>{fmt(report['weibull_A'])} m/s</b>. A média teórica da Weibull "
         f"({fmt(report['weibull_mean'])} m/s) aproxima a média observada "
         f"({fmt(report['ws100_mean'])} m/s) — útil para estimativas de energia eólica."),
        "",
        images=["images/14_weibull.png"],
    ))

    parts.append(section(
        "Persistência (autocorrelação)",
        (f"A velocidade a 100 m é muito persistente: autocorrelação de "
         f"{fmt(report['acf_1h'])} em 1 h, {fmt(report['acf_6h'])} em 6 h e "
         f"{fmt(report['acf_24h'])} em 24 h. Isso explica o sucesso de modelos "
         "autorregressivos e reforça o valor de janelas de contexto em modelos de previsão."),
        "",
        images=["images/15_autocorrelacao.png", "images/15b_pacf.png",
                "images/15c_diurnal_acf.png", "images/15d_smoothed_acf_pacf.png",
                "images/15e_pacf_horaria.png"],
    ))

    parts.append(section(
        "Meteorologia × vento",
        (f"Correlação com a velocidade a 100 m: temperatura r = "
         f"{fmt(report['corr_ws100_temp'])}; umidade r = {fmt(report['corr_ws100_humid'])}; "
         f"pressão r = {fmt(report['corr_ws100_press'])}."),
        "",
        images=["images/16_meteorologia.png"],
    ))

    parts.append(section(
        "Qualidade do sinal LiDAR (CIS)",
        (f"O carrier-to-noise médio é {fmt(report['cis_mean_overall'])}; valores baixos "
         f"ocorrem em {fmt(report['cis_pct_low'])} % das amostras e indicam dados de "
         "menor confiabilidade (úteis como flags de qualidade em pré-processamento)."),
        "",
        images=["images/17_cis_qualidade_sinal.png"],
    ))

    parts.append(section(
        "Deslocamentos do LiDAR (disp/vdisp)",
        (f"Os deslocamentos apresentam perfil crescente com a altura. O disp100 "
         f"correlaciona {fmt(report['corr_disp100_ws100'])} com a velocidade a 100 m."),
        "",
        images=["images/18_deslocamentos.png"],
    ))

    parts.append(section(
        "Perfil vertical ao longo do tempo",
        "Visualização da campanha inteira: intensidade do vento (cor) por altura e tempo. "
        "Permite identificar episódios de vento forte e a evolução da camada de cisalhamento.",
        "",
        images=["images/19_perfil_temporal.png"],
    ))

    parts.append(section(
        "Estatísticas descritivas",
        "Tabela resumo para todas as alturas (velocidade horizontal), além de v100, "
        "dir100 e variáveis meteorológicas.",
        '<img src="images/20_tabela_estatisticas.png" alt="tabela">',
    ))

    feats = (
        '<p>As <b>features de entrada</b> dos modelos seq2seq derivam das medições '
        'LiDAR e meteorológicas aqui analisadas. O embasamento científico para cada '
        'grupo de variáveis vem da própria EDA:</p>'
        '<ul>'
        f'<li><b>Velocidade do vento em múltiplas alturas (ws40–ws260).</b> O '
        f'perfil vertical cresce monotonicamente (40 m ≈ {fmt(report["ws40_mean"])} m/s, '
        f'260 m ≈ {fmt(report["ws260_mean"])} m/s), e as alturas são fortemente '
        f'correlacionadas (r = {fmt(report["corr_ws40_ws100"])} entre 40 e 100 m; '
        f'r = {fmt(report["corr_ws40_ws260"])} entre 40 e 260 m). A informação do '
        'perfil completo permite ao modelo inferir a estrutura de cisalhamento '
        '(expoente α) e generalizar melhor do que usar só o ponto de hub.</li>'
        f'<li><b>Direção do vento (dir).</b> É uma variável circular; a EDA mostra '
        'relação fraca mas significativa com a velocidade (r = '
        f'{fmt(report["corr_ws100_dir100"])}). Por isso a direção é codificada nas '
        'features como <b>sin(dir)</b> e <b>cos(dir)</b>, eliminando a '
        'descontinuidade 359°→1° e permitindo que o modelo trate a circularidade '
        'corretamente (mesma razão pela qual hora e dia do ano são codificados como '
        'hour_sin/cos e doy_sin/cos).</li>'
        f'<li><b>Componente vertical (v).</b> Mede a atividade convectiva: a EDA '
        f'reporta {fmt(report["pct_updraft"])} % das amostras em ascensão, com '
        f'correlação fraca (r = {fmt(report["corr_ws100_v100"])}) com a velocidade '
        'horizontal. Agrega contexto físico de turbulência que o vento horizontal '
        'sozinho não captura.</li>'
        f'<li><b>Qualidade do sinal (CIS).</b> O carrier-to-noise médio é '
        f'{fmt(report["cis_mean_overall"])}, com {fmt(report["cis_pct_low"])} % das '
        'amostras em sinal baixo. Usado como flag de confiabilidade, evita que o '
        'modelo aprenda com medições degradadas.</li>'
        f'<li><b>Deslocamentos (disp/vdisp).</b> O disp100 correlaciona '
        f'{fmt(report["corr_disp100_ws100"])} com a velocidade, indicando instantes '
        'de medição instável que merecem peso reduzido.</li>'
        f'<li><b>Meteorologia (temp, humid, press).</b> Correlações com a velocidade '
        f'a 100 m de {fmt(report["corr_ws100_temp"])} (temp), '
        f'{fmt(report["corr_ws100_humid"])} (umid) e '
        f'{fmt(report["corr_ws100_press"])} (press). São covariáveis externas que '
        'contextualizam o estado atmosférico de larga escala.</li>'
        f'<li><b>Dinâmica temporal (persistência).</b> A autocorrelação de '
        f'{fmt(report["acf_1h"])} em 1 h e {fmt(report["acf_6h"])} em 6 h confirma '
        'forte persistência — o que justifica janelas de contexto (input_steps) '
        'que alimentem o modelo com o histórico recente.</li>'
        '<li><b>Denoising por wavelets.</b> A suavização com média horária reduz o '
        'ruído de alta frequência e evidencia a queda da autocorrelação, melhorando '
        'a relação sinal-ruído das features sem perder o ciclo diurno.</li>'
        '</ul>'
        f'<p>Em conjunto, estas features representam <b>estado do vento no ponto '
        'de interesse (ws100), estrutura vertical do escoamento, circularidade '
        'direcional, atividade convectiva e contexto atmosférico externo</b> — '
        'fundamentos físicos da dinâmica do vento que dão às previsões base '
        'científica além da simples interpolação estatística da série.</p>'
    )
    parts.append(section(
        "Features criadas e embasamento científico",
        "Justificativa, a partir da EDA, do uso de cada grupo de variáveis na "
        "solução de previsão de velocidade do vento.",
        feats,
        images=["images/03_perfil_vertical.png", "images/10_correlacao_alturas.png",
                "images/13d_dir_sin_cos_corr.png", "images/07_velocidade_vertical.png",
                "images/16_meteorologia.png", "images/15_autocorrelacao.png",
                "images/02_serie_temporal.png"],
    ))

    html = (f"<!DOCTYPE html>\n<html lang=\"pt-BR\">\n<head>\n"
            "<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>EDA — Dados LiDAR de Vento</title>\n<style>{CSS}</style>\n"
            "</head>\n<body>\n<header>\n"
            "<h1>Análise Exploratória — Dados LiDAR de Vento</h1>\n"
            f"<p>Perfilador LiDAR · medições de 10 min · alturas {min(HEIGHTS)}–{max(HEIGHTS)} m ·\n"
            f"   {report['start']:%d/%m/%Y} a {report['end']:%d/%m/%Y}</p>\n"
            "</header>\n<div class=\"container\">\n"
            f"<div class=\"cards\">{cards}</div>\n{''.join(parts)}\n"
            "</div>\n"
            "<footer>Relatório gerado automaticamente por tests/eda_lidar.py</footer>\n"
            "<script>\n"
            "function copiarLatex(btn){\n"
            "  var latex = btn.getAttribute(\"data-latex\");\n"
            "  function done(){ btn.textContent = \"copiado!\"; setTimeout(function(){ btn.textContent = \"copiar como LaTeX\"; }, 1500); }\n"
            "  if(navigator.clipboard && navigator.clipboard.writeText){\n"
            "    navigator.clipboard.writeText(latex).then(done).catch(function(){ fallback(latex); done(); });\n"
            "  } else { fallback(latex); done(); }\n"
            "}\n"
            "function fallback(text){\n"
            "  var ta = document.createElement(\"textarea\");\n"
            "  ta.value = text; document.body.appendChild(ta); ta.select();\n"
            "  try{ document.execCommand(\"copy\"); }catch(e){}\n"
            "  document.body.removeChild(ta);\n"
            "}\n"
            "</script>\n</body>\n</html>")
    return html


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["id"], format="mixed")
    df = df.sort_values("timestamp").set_index("timestamp")
    df = df.apply(pd.to_numeric, errors="coerce")

    report = {
        "start": df.index.min(),
        "end": df.index.max(),
    }

    plot_overview(df, report)
    plot_time_series(df, report)
    plot_meteo_series(df)
    plot_vertical_series(df)
    plot_direction_series(df)
    plot_day_heights(df)
    plot_vertical_profile(df, report)
    plot_boxplot(df)
    plot_histograms(df)
    plot_wind_rose(df, report)
    plot_vertical_velocity(df, report)
    plot_diurnal(df, report)
    plot_monthly(df, report)
    plot_correlation(df, report)
    plot_scatter_heights(df)
    plot_shear(df, report)
    plot_dir_speed_vertical(df)
    plot_dir_vertical_corr(df, report)
    plot_dir_sin_cos_corr(df, report)
    plot_ws_vert_dir_corr(df, report)
    plot_weibull(df, report)
    plot_autocorrelation(df, report)
    plot_pacf(df, report)
    plot_smoothed_acf_pacf(df, report)
    plot_hourly_pacf(df, report)
    plot_diurnal_acf(df, report)
    plot_meteo(df, report)
    plot_cis(df, report)
    plot_disp(df, report)
    plot_profile_heatmap(df)
    plot_summary_table(df)

    html = build_html(report)
    out_file = OUT_DIR / "relatorio_eda_lidar.html"
    out_file.write_text(html, encoding="utf-8")

    print(f"Figuras salvas em: {IMG_DIR}")
    print(f"Relatório gerado em: {out_file}")


if __name__ == "__main__":
    main()