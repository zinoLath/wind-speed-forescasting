"""Gera o relatório HTML do estudo de features (data/results/features_study/)."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import common

OUT = PROJECT_ROOT / "data/results/features_study"
IMG = OUT / "img"
IMG.mkdir(parents=True, exist_ok=True)

study = json.load(open(OUT / "study_tables.json"))

# --- coleta validação + evidências da sessão --------------------------------
FS = ["fs_ciclo4", "fs_compl6", "fs_evid8", "fs_dirv100_13"]
SLUG_OF = {"ciclo4": "fs_ciclo4", "compl6": "fs_compl6",
           "evid10": "fs_evid8", "dirv100-13": "fs_dirv100_13"}
runs = []
for slug in FS:
    d = json.load(open(PROJECT_ROOT / "pipeline/tmp" / slug / "metrics.json"))
    runs.append(d)
FS2 = ["fs2_ciclo4", "fs2_compl6", "fs2_evid8", "fs2_dirv100_13"]
runs_df = []
for slug in FS2:
    d = json.load(open(PROJECT_ROOT / "pipeline/tmp" / slug / "metrics.json"))
    runs_df.append(d)
FIXES = ["fix1_tf_time", "fix2_tf_pos", "fix3_dir_time"]
runs_fix = []
for slug in FIXES:
    d = json.load(open(PROJECT_ROOT / "pipeline/tmp" / slug / "metrics.json"))
    runs_fix.append(d)
EVID = [
    ("9feat (ref. sessão)", 10, 1.057, 1.771, 0.053, 1.181),
    ("todas feat (ref.)", 57, 0.924, 1.408, None, 1.811),
    ("9feat+lag24h (ref.)", 15, 1.246, 2.326, None, 1.073),
    ("9feat+dir+v100 sem norm (ref.)", 14, 0.995, 1.622, 0.133, 1.373),
]

# --- gráfico comparativo -----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
labels = [f"{d['condicao']}\n({d['n_features']}c)" for d in runs] + \
         [f"{n}\n({n_c}c)" for n, n_c, *_ in EVID]
all_mse = [d["test_all_horizons"]["mse"] for d in runs] + [e[3] for e in EVID]
roll_mae = [d["rolling_h36"]["mae"] for d in runs] + [e[5] for e in EVID]
colors = ["#1f77b4"] * len(runs) + ["#aaaaaa"] * len(EVID)
for ax, vals, title in ((axes[0], all_mse, "MSE all-horizons (teste)"),
                        (axes[1], roll_mae, "MAE rolling h=36 (teste)")):
    ax.bar(range(len(labels)), vals, color=colors)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
fig.suptitle("Validação: conjuntos propostos (azul) vs evidências da sessão (cinza)", fontsize=11)
fig.tight_layout()
fig.savefig(IMG / "comparacao_conjuntos.png", dpi=140)
plt.close(fig)

# --- rolling forecast x série original ----------------------------------------
def rolling_chart(runs_, slug_map, fname, title, palette):
    fig, axes = plt.subplots(len(runs_), 1, figsize=(12.5, 2.75 * len(runs_)),
                             sharex=True)
    for ax, d, color in zip(axes, runs_, palette):
        df = pd.read_csv(PROJECT_ROOT / "pipeline/tmp" / slug_map[d["condicao"]]
                         / "Seq2Seq_TCN" / "evaluate" / "predictions_rolling.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        ax.plot(df["timestamp"], df["actual_raw"], color="black", lw=1.1,
                label="ws100 real")
        ax.plot(df["timestamp"], df["predicted"], color=color, lw=1.1,
                label=f"rolling h=36 ({d['condicao']})")
        ax.set_title(f"{d['condicao']} ({d['n_features']} canais) · "
                     f"roll MAE {d['rolling_h36']['mae']:.3f} · "
                     f"MSE {d['rolling_h36']['mse']:.3f}", fontsize=9.5, loc="left")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25)
    axes[-1].tick_params(axis="x", rotation=15)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(IMG / fname, dpi=140)
    plt.close(fig)


SLUG_OF2 = {"ciclo4-df": "fs2_ciclo4", "compl6-df": "fs2_compl6",
            "evid10-df": "fs2_evid8", "dirv100-13-df": "fs2_dirv100_13"}
rolling_chart(runs, SLUG_OF, "rolling_vs_real.png",
              "Previsão encadeada (rolling h=36) × série original — teacher forcing",
              ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
rolling_chart(runs_df, SLUG_OF2, "rolling_vs_real_direto.png",
              "Previsão encadeada (rolling h=36) × série original — decoder direto",
              ["#9467bd", "#8c564b", "#e377c2", "#17becf"])

# --- amostras de predição completa + horizontes fixos --------------------------
SLUG_ALL = {**SLUG_OF, **SLUG_OF2,
            "fix1: TF + senoides futuras": "fix1_tf_time",
            "fix2: TF + posicional": "fix2_tf_pos",
            "fix3: direto + senoides futuras": "fix3_dir_time"}
RUN_ALL = runs + runs_df
_real = common.load_dataset(PROJECT_ROOT / "data/dataset.csv",
                            keep_raw=("ws100",))["ws100"]
H_SHOW = [1, 3, 6, 12, 18, 24, 30]


def _load_pred(cond):
    df = pd.read_csv(PROJECT_ROOT / "pipeline/tmp" / SLUG_ALL[cond]
                     / "Seq2Seq_TCN" / "evaluate" / "predictions_all_horizons.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["origin"] = pd.to_datetime(df["origin"])
    return df


def fan_samples(cond, fname, color):
    """Origens variadas: input de 72 passos + trajetória 1-36 sobre a série real."""
    df = _load_pred(cond)
    cont = df[df.horizon == 36].set_index("origin")["actual_raw"]
    origins = np.sort(df["origin"].unique())
    picks = sorted({pd.Timestamp(origins[0]), pd.Timestamp(cont.idxmax()),
                    pd.Timestamp(cont.idxmin()),
                    pd.Timestamp(origins[len(origins) // 2]),
                    pd.Timestamp(origins[-1])})
    fig, axes = plt.subplots(len(picks), 1, figsize=(12.5, 2.6 * len(picks)))
    for ax, o in zip(axes, picks):
        t0 = o - pd.Timedelta(minutes=10 * 72)
        t1 = o + pd.Timedelta(minutes=10 * 36)
        seg = _real.loc[t0:t1]
        ax.plot(seg.index, seg.values, color="black", lw=1.0, label="ws100 real")
        p = df[df.origin == o].sort_values("horizon")
        ax.plot(p["timestamp"], p["predicted"], color=color, lw=1.7,
                label="predição (h=1..36)")
        ax.axvspan(t0, o, color="gray", alpha=0.07)
        ax.axvline(o, color="gray", ls="--", lw=0.9)
        mae = float((p["predicted"] - p["actual_raw"]).abs().mean())
        ax.set_title(f"origem {o.strftime('%d/%m %H:%M')} · MAE {mae:.2f} m/s",
                     fontsize=9, loc="left")
        ax.set_ylabel("m/s", fontsize=8)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
    fig.suptitle(f"Amostras de predição completa — {cond} (cinza = janela de "
                 "entrada, 72 passos; tracejado = origem)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(IMG / fname, dpi=140)
    plt.close(fig)


def horizon_chart(cond, fname, color):
    """Um painel por horizonte fixo: predição emitida a h passos vs série real."""
    df = _load_pred(cond)
    fig, axes = plt.subplots(len(H_SHOW), 1, figsize=(12.5, 1.9 * len(H_SHOW)),
                             sharex=True)
    for ax, h in zip(axes, H_SHOW):
        p = df[df.horizon == h]
        mae = float((p["predicted"] - p["actual_raw"]).abs().mean())
        ax.plot(p["timestamp"], p["actual_raw"], color="black", lw=0.9,
                label="ws100 real")
        ax.plot(p["timestamp"], p["predicted"], color=color, lw=1.0,
                label=f"predição h={h}")
        ax.set_title(f"h={h} ({h * 10} min) · MAE {mae:.3f}", fontsize=8.5,
                     loc="left")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
    axes[-1].tick_params(axis="x", rotation=15)
    fig.suptitle(f"Horizontes fixos × série real — {cond}", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(IMG / fname, dpi=140)
    plt.close(fig)


fan_samples("evid10", "samples_tf_evid10.png", "#2ca02c")
fan_samples("evid10-df", "samples_df_evid10.png", "#e377c2")
fan_samples("fix1: TF + senoides futuras", "samples_fix1.png", "#d62728")
fan_samples("fix2: TF + posicional", "samples_fix2.png", "#9467bd")
fan_samples("fix3: direto + senoides futuras", "samples_fix3.png", "#8c564b")
for d in runs:
    horizon_chart(d["condicao"], f"horizontes_{SLUG_ALL[d['condicao']]}.png",
                  "#1f77b4")
for d in runs_df:
    horizon_chart(d["condicao"], f"horizontes_{SLUG_ALL[d['condicao']]}.png",
                  "#9467bd")

# --- tabelas -----------------------------------------------------------------
uni = pd.DataFrame(study["univariate"]["h36"]).set_index("feature")
uni_h1 = pd.DataFrame(study["univariate"]["h1"]).set_index("feature")
imp = pd.DataFrame(study["perm_importance"]).set_index("feature")

uni_rows = "".join(
    f"<tr><td>{f}</td><td>{uni_h1.loc[f,'pearson']:+.3f}</td>"
    f"<td>{uni_h1.loc[f,'mi']:.3f}</td><td>{uni.loc[f,'pearson']:+.3f}</td>"
    f"<td>{uni.loc[f,'spearman']:+.3f}</td><td>{uni.loc[f,'mi']:.3f}</td></tr>"
    for f in uni.index)

val_rows = "".join(
    f"<tr><td>{d['condicao']}</td><td>{d['n_features']}</td><td>{d['epochs']}</td>"
    f"<td>{d['val_all_horizons']['mse']:.3f}</td>"
    f"<td>{d['test_all_horizons']['mae']:.3f}</td>"
    f"<td><b>{d['test_all_horizons']['mse']:.3f}</b></td>"
    f"<td>{d['test_all_horizons']['r2']:+.3f}</td>"
    f"<td><b>{d['rolling_h36']['mae']:.3f}</b></td>"
    f"<td>{d['rolling_h36']['mse']:.3f}</td></tr>"
    for d in runs)

val_df_rows = "".join(
    f"<tr><td>{d['condicao']}</td><td>{d['n_features']}</td><td>{d['epochs']}</td>"
    f"<td>{d['val_all_horizons']['mse']:.3f}</td>"
    f"<td>{d['test_all_horizons']['mae']:.3f}</td>"
    f"<td><b>{d['test_all_horizons']['mse']:.3f}</b></td>"
    f"<td>{d['test_all_horizons']['r2']:+.3f}</td>"
    f"<td><b>{d['rolling_h36']['mae']:.3f}</b></td>"
    f"<td>{d['rolling_h36']['mse']:.3f}</td></tr>"
    for d in runs_df)

fix_rows = "".join(
    f"<tr><td>{d['condicao']}</td>"
    f"<td>{d['test_all_horizons']['mae']:.3f}</td>"
    f"<td>{d['test_all_horizons']['mse']:.3f}</td>"
    f"<td>{d['test_all_horizons']['r2']:+.3f}</td>"
    f"<td>{d['rolling_h36']['mae']:.3f}</td>"
    f"<td>{d['amplitude_pred']:.2f}</td></tr>"
    for d in runs_fix)

evid_rows = "".join(
    f"<tr><td>{n}</td><td>{n_c}</td><td>—</td><td>—</td><td>{mae:.3f}</td>"
    f"<td>{mse:.3f}</td><td>{('%+.3f' % r2) if r2 is not None else '—'}</td>"
    f"<td>{rmae:.3f}</td><td>—</td></tr>"
    for n, n_c, mae, mse, r2, rmae in EVID)

imp_rows = "".join(
    f"<tr><td>{f}</td><td>{imp.loc[f,'perm_importance']:.4f}</td>"
    f"<td>±{imp.loc[f,'perm_std']:.4f}</td></tr>"
    for f in imp.index)

CSS = """
body{font-family:system-ui,sans-serif;max-width:1080px;margin:24px auto;padding:0 16px;color:#222}
h1{border-bottom:3px solid #1f77b4;padding-bottom:8px} h2{margin-top:36px;color:#1f77b4}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}
th,td{border:1px solid #ccc;padding:5px 8px;text-align:right}
th:first-child,td:first-child{text-align:left}
img{max-width:100%;border:1px solid #ddd;border-radius:6px;margin:8px 0}
.reco{background:#eef7ee;border-left:5px solid #2ca02c;padding:12px 16px;border-radius:6px}
.warn{background:#fdf6e3;border-left:5px solid #b58900;padding:10px 14px;border-radius:6px;font-size:13px}
.note{color:#666;font-size:13px}
"""

html = f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8">
<title>Estudo de features — TCN</title><style>{CSS}</style></head><body>
<h1>Estudo de features para a TCN</h1>
<p class="note">Série: data/dataset.csv (7561 amostras de 10 min, split 75/20/5 cronológico).
Alvo de treino: ws100_wavelet (sym18 nível 1); métricas contra ws100 bruto.
Protocolo idêntico ao da investigação: TCN sem pós-dropout, seed 42, huber, norm MinMax
(exceto quando indicado).</p>

<h2>1. Metodologia</h2>
<p>Três análises determinísticas (sem treino profundo, sem aleatoriedade de GPU):
(Pearson/Spearman/informação mútua) de cada candidata em <i>t</i> para
ws100 em <i>t+h</i>; <b>B</b> redundância entre candidatas (Spearman + clustering
hierárquico); <b>C</b> importância multivariada por permutação num RandomForest
determinístico (seed 0, 300 árvores, treinado só no split de treino, avaliado na
validação, h=36). Depois, <b>D</b> validação empírica com 4 conjuntos treinados.</p>

<h2>2. Parte A — poder preditivo univariado</h2>
<img src="img/univariado_h36.png">
<img src="img/decaimento_horizonte.png">
<table>
<tr><th>feature</th><th>r (h=1)</th><th>MI (h=1)</th>
<th>r (h=36)</th><th>ρ (h=36)</th><th>MI (h=36)</th></tr>
{uni_rows}
</table>
<p><b>Leituras:</b> em h=1 a persistência domina (r=+0.95 do ws100_wavelet).
Em h=36 <b>ws260_wavelet passa à frente</b> (r=+0.51, MI 0.177) — o vento a 260 m
<b>antecede</b> o de 100 m; o ciclo diurno (day_sin/cos) tem correlação ~0 mas MI alta
(relação não-linear); dir100_sin/cos carregam sinal moderado (|r|≈0.3).</p>

<h2>3. Parte B — redundância</h2>
<img src="img/redundancia.png">
<p>Correlações altas: ws40↔ws100 (ρ=0.93), ws100↔ws260 (ρ=0.90), mas
<b>ws40↔ws260 apenas 0.72</b> — o par 40/260 é o mais complementar. Único cluster
|ρ|&gt;0.95: dir100_sin↔dir100_cos (estrutura cíclica, ambos necessários).</p>

<h2>4. Parte C — permutation importance (RF determinístico)</h2>
<img src="img/perm_importance.png">
<p>R² do RF na validação em h=36: <b>{study['r2_rf_val_h36']:.3f}</b>.</p>
<table>
<tr><th>feature</th><th>queda de R²</th><th>desvio</th></tr>
{imp_rows}
</table>
<p><b>ws260_wavelet domina</b> (queda 0.323 ao permutar), seguido do ciclo diurno
(day_cos 0.138, hour_cos 0.055). ws100_wavelet quase não acrescenta dado ws260
(+0.020) — fortemente subsumido. v100, lag24h, disp/vdisp: contribuições pequenas
(&lt;0.02).</p>

<h2>5. Parte D — validação empírica</h2>
<p>Quatro conjuntos treinados (alvo ws100_wavelet sempre presente como entrada;
"n" conta canais de entrada):</p>
<table>
<tr><th>conjunto</th><th>n</th><th>épocas</th><th>val MSE</th>
<th>all MAE</th><th>all MSE</th><th>all R²</th><th>roll MAE</th><th>roll MSE</th></tr>
{val_rows}
</table>
<p>Evidências anteriores da sessão (mesma receita):</p>
<table>
<tr><th>condição</th><th>n</th><th></th><th></th><th>all MAE</th><th>all MSE</th>
<th>all R²</th><th>roll MAE</th><th></th></tr>
{evid_rows}
</table>
<img src="img/comparacao_conjuntos.png">
<p><b>Leituras:</b></p>
<ul>
<li><b>evid10 (11 canais) tem o melhor MSE all-horizons global (1.381)</b> — supera o
todas-features (1.408) com 5× menos canais e o 9feat (1.771) com 22% menos erro.</li>
<li><b>ciclo4 (5 canais) tem o melhor rolling da família normalizada (1.125)</b> —
o comportamento de previsão encadeada melhora conforme o conjunto encolhe
(1.125 → 1.422 → 1.780 com 5 → 11 → 13 canais).</li>
<li>dirv100-13 (9feat+dir100+v100, com norm) não melhora sobre evid10 —
<b>v100 não paga o custo</b> (MI baixa, importância de permutação ~0.008), coerente
com a Parte C.</li>
<li>compl6 decepcionou no all-horizons (3.61) apesar das boas escolhas univariadas —
ver aviso sobre variância abaixo.</li>
</ul>

<div class="warn"><b>Aviso — variância da TCN:</b> cada conjunto foi treinado uma
única vez (seed 42); execuções repetidas da mesma configuração variaram o rolling
MAE entre 1.0 e 2.0 nesta sessão. As ordens de grandeza e os contrastes grandes
(evid10 vs dirv100-13; ciclo4 rolling) são robustos, diferenças &lt;10% não são.</div>

<h2>5b. Rolling forecast × série original</h2>
<p>Previsão encadeada de 36 passos (6 h), avançando de hora em hora na janela de
teste, contra a série real (preto). A janela de teste contém a queda de regime
(vento médio diário 7.8 → 5.8 m/s).</p>
<img src="img/rolling_vs_real.png">

<h2>5c. Réplica com decoder direto (sem teacher forcing no treino)</h2>
<p>Mesmos 4 conjuntos, mesma seed e HP, com <code>decoder_mode="direct"</code>:</p>
<table>
<tr><th>conjunto</th><th>n</th><th>épocas</th><th>val MSE</th>
<th>all MAE</th><th>all MSE</th><th>all R²</th><th>roll MAE</th><th>roll MSE</th></tr>
{val_df_rows}
</table>
<p><b>Comparação TF vs direto</b> (all MSE / roll MAE):</p>
<ul>
<li>ciclo4: 1.637/1.125 (TF) vs 2.605/1.436 (direto)</li>
<li>compl6: 3.608/1.295 (TF) vs 2.535/1.340 (direto)</li>
<li>evid10: <b>1.381</b>/1.422 (TF) vs 1.757/<b>1.255</b> (direto)</li>
<li>dirv100-13: 1.645/1.780 (TF) vs 2.370/<b>1.281</b> (direto)</li>
</ul>
<p><b>Leituras:</b> o teacher forcing segue melhor no agregado all-horizons
(3 de 4 conjuntos). O decoder direto reduz o spread entre conjuntos e melhora o
rolling em 3 de 4 — mas <b>não recupera a amplitude da série</b>: o achatamento
para a média é limitação de informação no horizonte de 6 h, não artefato do
teacher forcing. evid10 permanece o melhor conjunto também sem TF (1.757) —
a recomendação é robusta ao modo do decoder.</p>
<img src="img/rolling_vs_real_direto.png">

<h2>5d. Amostras de predição completa e comportamento por horizonte</h2>
<p>Predições <b>completas</b> (h=1..36 a partir de uma origem): cinza = janela de
entrada (72 passos = 12 h), tracejado = origem, preto = série real, colorido =
trajetória prevista. Amostras: início/fim da janela, pico, vale e meio do teste.</p>
<img src="img/samples_tf_evid10.png">
<img src="img/samples_df_evid10.png">
<p>Comportamento da predição em <b>horizontes fixos</b> (h = 1, 3, 6, 12, 18, 24, 30)
para todos os conjuntos — a curva colorida é o que o modelo disse <i>h</i> passos
antes de cada instante:</p>
<h3>Teacher forcing</h3>
<img src="img/horizontes_fs_ciclo4.png">
<img src="img/horizontes_fs_compl6.png">
<img src="img/horizontes_fs_evid8.png">
<img src="img/horizontes_fs_dirv100_13.png">
<h3>Decoder direto</h3>
<img src="img/horizontes_fs2_ciclo4.png">
<img src="img/horizontes_fs2_compl6.png">
<img src="img/horizontes_fs2_evid8.png">
<img src="img/horizontes_fs2_dirv100_13.png">
<p><b>Leitura esperada:</b> em h=1–3 a predição praticamente cobre a série (segue
a persistência); conforme h cresce perde amplitude e fase — h=18–30 já parecem
uma versão suavizada/atrasada, ilustrando o colapso para a média discutido
acima.</p>

<h2>6. Correções do decoder testadas</h2>
<p>Diagnóstico: na inferência o decoder recebe entrada constante (TF: zeros +
último valor no passo 0; direto: último valor repetido), então um decoder
convolucional só produz trajetórias constantes (repetição em pares por efeito
de borda). Três correções testadas sobre evid10:</p>
<table>
<tr><th>correção</th><th>all MAE</th><th>all MSE</th><th>all R²</th>
<th>roll MAE</th><th>ampl. trajetória</th></tr>
{fix_rows}
<tr><td>evid10 original (ref.)</td><td>0.916</td><td><b>1.381</b></td>
<td>+0.262</td><td>1.422</td><td>0.17</td></tr>
<tr><td><i>série real</i></td><td></td><td></td><td></td><td></td>
<td><i>2.84</i></td></tr>
</table>
<img src="img/samples_fix1.png">
<img src="img/samples_fix2.png">
<img src="img/samples_fix3.png">
<p><b>Conclusão:</b> nenhuma correção recuperou a amplitude (0.17–0.26 vs
2.84 real). Com teacher forcing, o modelo aprende o atalho de 1 passo
(copiar o decoder input real) e ignora as senoides — que na inferência não
compensam a falta do sinal real. No modo direto + senoides (fix3) a amplitude
melhora pouco e as métricas ficam iguais ao direto puro. <b>A trajetória reta
não é limitação do decoder: é a ausência de sinal previsível no horizonte de
6 h</b> — coerente com o R²=0.325 do estudo determinístico (Parte C). Caminho
recomendado: saída probabilística/quantílica para representar a incerteza,
em vez de forçar amplitude numa predição pontual.</p>

<h2>7. Recomendação</h2>
<div class="reco">
<p><b>Conjunto recomendado (11 canais):</b> ws40_wavelet, ws100_wavelet (alvo),
ws260_wavelet, disp40_wavelet, vdisp40_wavelet, dir100_sin, dir100_cos,
hour_sin, hour_cos, day_sin, day_cos.</p>
<ul>
<li><b>Mantém:</b> o trio 40/100/260 m (par complementar 40↔260 + persistência do
alvo), dispersões 40 m, direção 100 m em senoides, ciclo diurno completo.</li>
<li><b>Remove:</b> v100 (MI/importância marginais), lag24h (importância 0.016 e MSE
2.33 na validação anterior), ws50–ws140/alturas intermediárias (ρ&gt;0.9 com o trio),
doy_sin/cos (série de ~52 dias não completa um ciclo anual).</li>
<li><b>Tarefa sensível a rolling:</b> considerar ciclo4 (alvo + 4 senoides) —
melhor rolling observado (1.125) ao custo de MSE all-horizons pior.</li>
<li><b>Candidatos futuros com evidência:</b> turbulência local (desvio móvel do
resíduo da wavelet) e shear 40–260 — aparecem nas análises univariadas mas não foram
validados em treino.</li>
</ul>
</div>
<p class="note">Gerado por tests/feature_study.py (análises) e
tests/feature_sets_validation.py (validação). Métricas cruas em
data/results/features_study/study_tables.json e pipeline/tmp/fs_*/metrics.json.</p>
</body></html>"""

(OUT / "index.html").write_text(html, encoding="utf-8")
print(f"Relatório: {OUT / 'index.html'}")
