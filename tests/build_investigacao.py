"""Gera o HTML de investigação dos resultados anômalos da sessão.

Coleta os metrics.json de todas as variantes trained_compare_*, produz
gráficos (PNG) e monta um site estático separado do relatório principal:
    data/results/investigacao/index.html

Uso: python tests/build_investigacao.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "data" / "results" / "investigacao"
IMG = OUT / "img"
IMG.mkdir(parents=True, exist_ok=True)

PERSIST_ROLL = 1.1990
PERSIST_ALL_MSE = 2.036
ARIMA_ALL_MSE = 1.706
ARIMA_ROLL = 1.2530

# (rótulo, família, caminho, condição)
RUNS = [
    ("TCN — baseline", "TCN", "trained_compare/Seq2Seq_TCN",
     "level 1, todas as features, teacher forcing, gate off, ctx horizon"),
    ("TCN — sem pós-dropout", "TCN", "trained_compare_nopostdrop/Seq2Seq_TCN",
     "encoder/decoder post-dropout = 0 (resto idêntico)"),
    ("TCN — 9 features, sem pós-dropout", "TCN", "trained_compare_nopostdrop_9feat/Seq2Seq_TCN",
     "features restritas (9 canais) + pós-dropouts zerados"),
    ("TCN — 9feat + lag24h", "TCN", "ab_lag24_9feat/Seq2Seq_TCN",
     "idem anterior + lag de 24h das 5 wavelets (14 canais)"),
    ("TCN — 9feat+dir100+v100, sem norm", "TCN", "ab_nonorm_9feat_dir_v100/Seq2Seq_TCN",
     "13 canais (dir100 senoidal, v100 wavelet), pós-dropout 0, sem normalização"),
    ("TCN — todas feat + dir senoidal, sem norm", "TCN", "ab_nonorm_allfeat_dir_v100/Seq2Seq_TCN",
     "todas as features + v100 wavelet, dir100 só senoidal, pós-dropout 0, sem normalização"),
    ("TCN — fs ciclo4", "TCN", "fs_ciclo4/Seq2Seq_TCN",
     "estudo de features: alvo + 4 senoides dia/hora (5 canais)"),
    ("TCN — fs compl6", "TCN", "fs_compl6/Seq2Seq_TCN",
     "estudo de features: ws40+ws260 + 4 senoides (7 canais)"),
    ("TCN — fs evid10", "TCN", "fs_evid8/Seq2Seq_TCN",
     "estudo de features: ws40/100/260+disp/vdisp40+dir100 senoidal+senoides (11 canais)"),
    ("TCN — fs dirv100-13", "TCN", "fs_dirv100_13/Seq2Seq_TCN",
     "estudo de features: 9feat + dir100 senoidal + v100 wavelet (13 canais)"),
    ("TCN — fs ciclo4-df", "TCN", "fs2_ciclo4/Seq2Seq_TCN",
     "estudo de features: decoder direto, alvo + 4 senoides (5 canais)"),
    ("TCN — fs compl6-df", "TCN", "fs2_compl6/Seq2Seq_TCN",
     "estudo de features: decoder direto, ws40+ws260 + 4 senoides (7 canais)"),
    ("TCN — fs evid10-df", "TCN", "fs2_evid8/Seq2Seq_TCN",
     "estudo de features: decoder direto, 11 canais"),
    ("TCN — fs dirv100-13-df", "TCN", "fs2_dirv100_13/Seq2Seq_TCN",
     "estudo de features: decoder direto, 13 canais"),
    ("TCN — fix1 TF+senoides futuras", "TCN", "fix1_tf_time/Seq2Seq_TCN",
     "decoder com hour/day sin-cos dos instantes-alvo (TF)"),
    ("TCN — fix2 TF+posicional", "TCN", "fix2_tf_pos/Seq2Seq_TCN",
     "decoder com codificação posicional senoidal (TF)"),
    ("TCN — fix3 direto+senoides", "TCN", "fix3_dir_time/Seq2Seq_TCN",
     "decoder direto + senoides futuras"),
    ("TCN — 9feat + lag24h, sem normalização", "TCN", "ab_lag24_nonorm/Seq2Seq_TCN",
     "idem anterior, sem MinMaxScaler (identidade)"),
    ("TCN — wavelet série inteira", "TCN", "trained_compare_wavfull/Seq2Seq_TCN",
     "wavelet aplicada na série completa antes do split"),
    ("TCN — sem TimeDistributed", "TCN", "trained_compare_nodense/Seq2Seq_TCN",
     "cabeça Flatten+Dense no lugar da TimeDistributed"),
    ("TCN — nogate (23/09)", "TCN", "trained_compare_nogate/Seq2Seq_TCN",
     "hp v1, gate off (rodada antiga)"),
    ("TCN — dynagate (23/09)", "TCN", "trained_compare_dynagate/Seq2Seq_TCN",
     "hp v1, gate dinâmico (rodada antiga)"),
    ("TCN — ctxnone (23/09)", "TCN", "trained_compare_ctxnone/Seq2Seq_TCN",
     "hp v1, sem contexto no decoder (rodada antiga)"),
    ("TCN_Bi — baseline", "TCN_Bi", "trained_compare/Seq2Seq_TCN_Bidirectional",
     "level 1, todas as features, teacher forcing"),
    ("TCN_Bi — huber", "TCN_Bi", "trained_compare_tcnbi_huber/Seq2Seq_TCN_Bidirectional",
     "idem baseline, loss huber"),
    ("TCN_Bi — sem teacher forcing", "TCN_Bi", "trained_compare_noteaching/Seq2Seq_TCN_Bidirectional",
     "decoder direct"),
    ("TCN_Bi — sem TimeDistributed", "TCN_Bi", "trained_compare_nodense/Seq2Seq_TCN_Bidirectional",
     "cabeça Flatten+Dense"),
    ("TCN_Bi — nogate (23/09)", "TCN_Bi", "trained_compare_nogate/Seq2Seq_TCN_Bidirectional",
     "hp v1, gate off"),
    ("TCN_Bi — dynagate (23/09)", "TCN_Bi", "trained_compare_dynagate/Seq2Seq_TCN_Bidirectional",
     "hp v1, gate dinâmico"),
    ("LSTM — baseline", "LSTM", "trained_compare/Seq2Seq_LSTM",
     "level 1, todas as features, teacher forcing"),
    ("LSTM — sem teacher forcing", "LSTM", "trained_compare_noteaching/Seq2Seq_LSTM",
     "decoder direct"),
    ("LSTM — wavelet série inteira", "LSTM", "trained_compare_wavfull/Seq2Seq_LSTM",
     "wavelet na série completa (idêntico ao baseline)"),
]


def collect():
    rows = []
    for label, family, rel, cond in RUNS:
        f = PROJECT_ROOT / "pipeline/tmp" / rel / "evaluate" / "metrics.json"
        if not f.is_file():
            continue
        d = json.load(open(f))
        a, r, t = d["all_horizons"], d["rolling"], d["model"]["training"]
        rows.append({"label": label, "family": family, "condition": cond,
                     "all_mae": a["mae"], "all_mse": a["mse"], "all_r2": a["r2"],
                     "roll_mae": r["mae"], "roll_rmse": r["rmse"], "roll_r2": r["r2"],
                     "val_loss": t["best_val_loss"], "epochs": t["epochs_run"]})
    return pd.DataFrame(rows)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / name, dpi=140)
    plt.close(fig)


def chart_run_cards(df):
    """Uma imagem por execução: rolling MAE e MSE all-horizons com referências."""
    import re

    family_color = {"TCN": "#ff7f0e", "TCN_Bi": "#d62728", "LSTM": "#1f77b4"}
    cards = {}
    for _, r in df.iterrows():
        slug = re.sub(r"[^a-z0-9]+", "_", r.label.lower()).strip("_")
        fname = f"run_{slug}.png"
        fig, axes = plt.subplots(1, 2, figsize=(5.4, 1.75))
        color = family_color[r.family]
        axes[0].barh(["roll"], [r.roll_mae], color=color)
        axes[0].axvline(PERSIST_ROLL, color="red", ls="--", lw=1)
        axes[0].text(min(r.roll_mae, ax_lim := max(r.roll_mae * 1.3,
                     PERSIST_ROLL * 1.15)) * 0.97, 0, f"{r.roll_mae:.2f}",
                     va="center", ha="right", fontsize=9, color="white")
        axes[0].set_xlim(0, ax_lim)
        axes[0].set_title("rolling MAE (m/s)", fontsize=8.5)
        axes[1].barh(["all"], [r.all_mse], color=color)
        axes[1].axvline(PERSIST_ALL_MSE, color="red", ls="--", lw=1)
        axes[1].text(min(r.all_mse, ax_lim2 := max(r.all_mse * 1.3,
                     PERSIST_ALL_MSE * 1.15)) * 0.97, 0, f"{r.all_mse:.2f}",
                     va="center", ha="right", fontsize=9, color="white")
        axes[1].set_xlim(0, ax_lim2)
        axes[1].set_title("MSE all-horizons", fontsize=8.5)
        for ax in axes:
            ax.set_yticks([])
            ax.tick_params(labelsize=7)
        savefig(fig, fname)
        cards.setdefault(r.family, []).append((fname, r))
    return cards


def chart_val_vs_test(df):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for family, color, marker in (("TCN", "#ff7f0e", "o"), ("TCN_Bi", "#d62728", "s"),
                                  ("LSTM", "#1f77b4", "^")):
        sub = df[df.family == family]
        ax.scatter(sub.val_loss * 300, sub.roll_mae, color=color, marker=marker,
                   s=70, label=family, alpha=0.85)
    ax.axhline(PERSIST_ROLL, color="red", ls="--", lw=1, label="persistência")
    ax.set_xlabel("val_loss escalada × 300 (aprox. MSE real)")
    ax.set_ylabel("rolling MAE no teste (m/s)")
    ax.set_title("val_loss NÃO prevê o desempenho em teste (cada ponto = 1 execução)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    savefig(fig, "inv_val_vs_teste.png")


def chart_pred_vs_real_cards(dataset):
    """Um PNG por variante: rolling h=36 vs série real; grade montada no HTML."""
    import re

    family_color = {"TCN": "#ff7f0e", "TCN_Bi": "#d62728", "LSTM": "#1f77b4"}
    grids = {}
    for family in ("TCN", "TCN_Bi", "LSTM"):
        variants = []
        for r in RUNS:
            if r[1] != family:
                continue
            f = PROJECT_ROOT / "pipeline/tmp" / r[2] / "evaluate" / "predictions_rolling.csv"
            if not f.is_file():
                continue
            pred = pd.read_csv(f)
            pred["timestamp"] = pd.to_datetime(pred["timestamp"])
            variants.append((r[0], r[2], pred))
        if not variants:
            continue
        cards = []
        for label, rel, pred in variants:
            slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
            fname = f"previsao_{family.replace('_','').lower()}_{slug}.png"
            mae = float(np.mean(np.abs(pred["predicted"] - pred["actual_raw"])))
            fig, ax = plt.subplots(figsize=(6.2, 2.4))
            ax.plot(pred["timestamp"], pred["actual_raw"], color="black",
                    lw=1.0, alpha=0.65, label="real (ws100)")
            ax.plot(pred["timestamp"], pred["predicted"], color=family_color[family],
                    lw=1.0, alpha=0.9, label=f"previsto (MAE {mae:.2f})")
            ax.set_title(f"{label} — rolling h=36", fontsize=9.5)
            ax.set_ylabel("m/s", fontsize=8)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=7, axis="x", rotation=20)
            ax.legend(fontsize=7.5)
            savefig(fig, fname)
            cards.append((fname, label, mae))
        grids[family] = cards
    return grids


def chart_full_prediction(dataset):
    """Um PNG por variante: nuvem da predição completa (origens × horizontes)."""
    import re

    family_color = {"TCN": "#ff7f0e", "TCN_Bi": "#d62728", "LSTM": "#1f77b4"}
    grids = {}
    for label, family, rel, cond in RUNS:
        f = PROJECT_ROOT / "pipeline/tmp" / rel / "evaluate" / "predictions_all_horizons.csv"
        if not f.is_file():
            continue
        df = pd.read_csv(f)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        fname = f"full_{family.replace('_','').lower()}_{slug}.png"
        color = family_color[family]
        fig, ax = plt.subplots(figsize=(6.2, 2.4))
        span = dataset["ws100"].loc[df["timestamp"].min():df["timestamp"].max()]
        ax.plot(span.index, span.values, color="black", lw=1.0, alpha=0.7,
                label="real")
        ax.scatter(df["timestamp"], df["predicted"], s=2, alpha=0.15, color=color,
                   label="predição completa (36 passos)")
        ax.set_title(f"{label} — predição completa", fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7, axis="x", rotation=20)
        ax.legend(fontsize=7)
        savefig(fig, fname)
        mae = float(np.mean(np.abs(df["predicted"] - df["actual_raw"])))
        grids.setdefault(family, []).append((fname, label, mae))
    return grids


def chart_horizon_errors():
    """Heatmap de MSE por modelo e horizonte — todas as variantes (verde = menor erro)."""
    mses, labels = [], []
    for label, family, rel, cond in RUNS:
        f = PROJECT_ROOT / "pipeline/tmp" / rel / "evaluate" / "predictions_all_horizons.csv"
        if not f.is_file():
            continue
        df = pd.read_csv(f)
        g = df.assign(se=(df["predicted"] - df["actual_raw"]) ** 2) \
              .groupby("horizon")["se"].mean()
        mses.append(g.to_numpy())
        labels.append(label)
    bh = pd.read_csv(PROJECT_ROOT / "data/results/baselines/arima_rf/per_horizon_metrics.csv")
    for label, col in (("ARIMA", "arima_rmse"), ("Random Forest", "rf_rmse"),
                       ("Persistência", "persistence_rmse")):
        mses.append((bh[col] ** 2).to_numpy())
        labels.append(f"{label} (baseline)")
    mat = np.vstack(mses)

    fig, ax = plt.subplots(figsize=(13, 4.2 + 0.35 * len(labels)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                   vmin=float(mat.min()), vmax=float(mat.max()),
                   interpolation="nearest")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(0, 36, 2), [str(h) for h in range(1, 37, 2)])
    ax.set_xlabel("Horizonte (passos de 10 min)")
    ax.set_title("MSE ((m/s)²) por modelo e horizonte (verde = menor erro)")
    for j in [0, 5, 11, 17, 23, 29, 35]:
        for i in range(len(labels)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color="black",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", alpha=0.75, lw=0))
    fig.colorbar(im, ax=ax, pad=0.01).set_label("MSE ((m/s)²)")
    savefig(fig, "inv_erros_por_horizonte.png")


def chart_tcnbi_bias(bias=1.523, mse=4.589):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bias_sq = bias ** 2
    resto = mse - bias_sq
    ax.bar(["viés²\n(superprevisão\nde nível)", "variância\n(tracking)"],
           [bias_sq, resto], color=["#d62728", "#1f77b4"], width=0.55)
    for i, v in enumerate([bias_sq, resto]):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=11)
    ax.set_ylabel("MSE ((m/s)²)")
    ax.set_title(f"TCN_Bi: decomposição do MSE all-horizons ({mse:.2f})\n"
                 f"viés médio de +{bias:.2f} m/s sobre o teste")
    savefig(fig, "inv_tcnbi_bias.png")


CSS = """.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}
.card{border:1px solid #ddd;border-radius:8px;padding:8px;background:#fff}
.card img{margin:0;border:none}
.card small{color:#555}
body{font-family:'Segoe UI',system-ui,Arial,sans-serif;max-width:1080px;
margin:24px auto;padding:0 18px;color:#222;line-height:1.45}
h1{color:#0f3d5e} h2{color:#1b6ca8;border-bottom:2px solid #1b6ca8;padding-bottom:4px;margin-top:34px}
h3{color:#0f3d5e} img{max-width:100%;border:1px solid #ddd;border-radius:6px;margin:8px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{border:1px solid #ccc;padding:5px 9px;text-align:left}
th{background:#eef3f8} tr:nth-child(even){background:#fafbfc}
.note{background:#fff7e0;border-left:4px solid #e0a800;padding:8px 12px}
.bad{background:#fdecea;border-left:4px solid #c0392b;padding:8px 12px}
.good{background:#eaf7ea;border-left:4px solid #2e7d32;padding:8px 12px}"""


def main():
    df = collect()
    df.to_csv(OUT / "runs.csv", index=False)
    dataset = pd.read_csv(PROJECT_ROOT / "data/dataset.csv", index_col=0)
    dataset.index = pd.to_datetime(dataset.index, format="mixed")
    cards = chart_run_cards(df)
    run_cards_grid = ""
    for family in ("TCN", "TCN_Bi", "LSTM"):
        items = cards.get(family, [])
        if not items:
            continue
        run_cards_grid += (f"<h3>{family} ({len(items)} execuções)</h3>"
                           f"<div class='grid'>")
        for fname, r in items:
            run_cards_grid += (
                f"<div class='card'><img src='img/{fname}'>"
                f"<div><b>{r.label}</b><br><small>{r.condition}</small><br>"
                f"<small>roll MAE {r.roll_mae:.3f} · all MSE {r.all_mse:.3f} · "
                f"{r.epochs} épocas</small></div></div>")
        run_cards_grid += "</div>"
    chart_val_vs_test(df)
    pred_grids_data = chart_pred_vs_real_cards(dataset)
    pred_grids = ""
    for family, cards in pred_grids_data.items():
        if not cards:
            continue
        pred_grids += f"<h3>{family}</h3><div class='grid'>"
        for fname, label, mae in cards:
            pred_grids += (f"<div class='card'><img src='img/{fname}'>"
                           f"<div><small>{label} · MAE {mae:.3f} m/s</small></div></div>")
        pred_grids += "</div>"
    full_grids_data = chart_full_prediction(dataset)
    full_grids = ""
    for family, cards in full_grids_data.items():
        if not cards:
            continue
        full_grids += f"<h3>{family}</h3><div class='grid'>"
        for fname, label, mae in cards:
            full_grids += (f"<div class='card'><img src='img/{fname}'>"
                           f"<div><small>{label} · MAE {mae:.3f} m/s</small></div></div>")
        full_grids += "</div>"
    chart_horizon_errors()
    chart_tcnbi_bias()

    table_rows = "".join(
        f"<tr><td>{r.label}</td><td>{r.condition}</td><td>{r.all_mae:.3f}</td>"
        f"<td>{r.all_mse:.3f}</td><td>{r.all_r2:+.3f}</td><td>{r.roll_mae:.3f}</td>"
        f"<td>{r.roll_rmse:.3f}</td><td>{r.roll_r2:+.3f}</td><td>{r.val_loss:.5f}</td>"
        f"<td>{r.epochs}</td></tr>"
        for _, r in df.iterrows())

    html = f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8">
<title>Investigação — resultados anômalos</title><style>{CSS}</style></head><body>
<h1>Investigação dos resultados anômalos</h1>
<p>Consolidação das execuções da sessão (protocolo honesto: split 75/20/5,
level 1 do wavelet salvo em cada model.json, métricas contra o ws100 bruto).
Referências: persistência rolling = {PERSIST_ROLL} · ARIMA all-MSE = {ARIMA_ALL_MSE} ·
ARIMA rolling = {ARIMA_ROLL}.</p>

<h2>1. Não-determinismo do treino de TCN (o fator dominante)</h2>
<p>O mesmo hp produziu rolling MAE de <b>1.01 a 1.81</b> no TCN e de
<b>1.07 a 2.02</b> no TCN_Bi ao longo da sessão — variância pura do
autotuning/backward de conv do cuDNN (os LSTMs reproduzem bit a bit).
<b>Nenhum delta &lt; ~0.3 em rolling MAE entre duas rodadas de TCN é
interpretável sem múltiplas seeds.</b></p>
{run_cards_grid}

<h2>2. val_loss não prevê o teste</h2>
<p>Execuções com val_loss quase idênticas (0.0035 vs 0.0036) produziram
rolling MAE de 1.01 e 1.81. A validação (e o objetivo do Optuna) é um
estimador otimista — seleção sobre 100 trials + early stopping.</p>
<img src="img/inv_val_vs_teste.png">

<h2>2b. Previsão × série real (rolling h=36), por variante</h2>
<p>Um painel por variante: preto = ws100 observado; colorido = previsão
autônoma de 6 h. A degradação no dia 8/nov (queda de nível) é o mesmo viés
da seção 3 — o zoom abaixo compara baseline × modificação mais próxima.</p>
{pred_grids}
<img src="img/inv_zoom_8nov.png">

<h2>2c. Predição completa (todas as origens × todos os horizontes)</h2>
<p>Uma imagem por variante, mostrando <b>apenas a predição completa</b>
(36 passos por origem) contra a série real:</p>
{full_grids}

<h2>2d. Erro por horizonte (heatmap)</h2>
<p>Heatmap de MSE ((m/s)²) por modelo e horizonte — <b>todas as variantes
testadas</b> mais os baselines; verde = menor erro.</p>
<img src="img/inv_erros_por_horizonte.png">

<h2>3. TCN_Bi: viés de nível + 4.6 de MSE</h2>
<p>O teste (últimos 2,6 dias) tem queda de regime (7.8 → 5.8 m/s). O TCN_Bi
superprevisa +1.52 m/s em média; <b>viés² = 2.3 dos 4.59 de MSE</b>. E erra
até em h=1 (MAE 1.60 vs 0.91 do TCN) — não rastreia o nível atual. É também
o menor modelo (489k params) e o mais instável entre rodadas.</p>
<img src="img/inv_tcnbi_bias.png">
<div class="note"><b>Achado contrário à expectativa:</b> loss <b>huber</b>
estabilizou o TCN_Bi — melhor execução da família
(all MAE 1.142 · MSE 2.256 · rolling 1.057).</div>

<h2>4. Teacher forcing × decoder direto</h2>
<div class="good"><b>LSTM:</b> direct ganha no all-horizons
(MAE 0.994 vs 1.049; MSE −8%) — treina sob a mesma distribuição do
teste. Empata com o ARIMA em MAE (0.994 vs 0.986).</div>
<div class="bad"><b>TCN_Bi:</b> direct não muda o all-horizons e piora o
rolling (1.48 vs 1.21; R² −1.0). Mantiver teacher forcing.</div>

<h2>5. Escopo da wavelet (por fatia × série inteira)</h2>
<p><b>LSTM: idêntico</b> (deltas ≤ 0.0002) — a hipótese "wavelet em série
pequena explica o erro" foi refutada para ele. <b>TCN: deltas grandes, mas
confundidos com o ruído de treino</b> (all melhorou −0.87 MSE; rolling piorou
+0.31) — exigiria múltiplas seeds para atribuição.</p>

<h2>6. Baselines</h2>
<p><b>ARIMA domina o all-horizons MSE (1.706)</b> contra todos os profundos
nas condições atuais. Nos rollings autônomos, TCN (1.015) e LSTM (1.122 —
decoder direto) vencem o ARIMA (1.253). A implementação antiga (BiLSTM 256,
protocolo com vazamento) media <b>0.0039</b> em escala MinMax; medida com
rigor: all MAE 1.008 (competitivo) mas rolling 1.363 (pior que a
persistência, R² −2.78).</p>

<h2>Tabela completa das execuções</h2>
<table>
<tr><th>Execução</th><th>Condição</th><th>all MAE</th><th>all MSE</th>
<th>all R²</th><th>roll MAE</th><th>roll RMSE</th><th>roll R²</th>
<th>val_loss</th><th>épocas</th></tr>
{table_rows}
</table>
<p class="note">Modelos salvos em <code>models/experiments/&lt;condição&gt;/&lt;Wrapper&gt;/</code>
· carregamento: <code>python scripts/load_experiment_model.py &lt;dir&gt; --rolling --next36</code>
· dados brutos: <code>runs.csv</code>.</p>
</body></html>"""
    out = OUT / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"HTML de investigação: {out}")


if __name__ == "__main__":
    main()
