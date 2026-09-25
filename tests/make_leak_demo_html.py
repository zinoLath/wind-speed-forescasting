"""Generate a self-contained HTML page (docs/leak_demo.html) that explains the
data leakage in the leaky rolling-forecast protocol, using real wind data.

Run:  python tests/make_leak_demo_html.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "leak_demo.html"

INPUT_STEPS = 12
OUTPUT_STEPS = 12
SEG_ROWS = 132


def main():
    df = pd.read_csv(ROOT / "data" / "dataset.csv")
    ws = df["ws100"].to_numpy()
    seg = ws[-SEG_ROWS:]

    labels = []
    for ts in df["id"].iloc[-SEG_ROWS:]:
        s = str(ts)
        if len(s) >= 16:
            labels.append(f"{s[5:7]}-{s[8:10]} {s[11:16]}")
        else:
            labels.append(s)

    test_start = INPUT_STEPS
    n = len(seg)
    K = n - test_start - OUTPUT_STEPS  # number of forecast steps
    x = np.arange(n)

    metrics = {
        "genuine": {"mae": None, "rmse": None},
        "leaky": {"mae": None, "rmse": None},
    }

    def mae(a, b):
        return float(np.mean(np.abs(a - b)))

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    iterations = []
    for k in range(K):
        actual = float(seg[test_start + k + OUTPUT_STEPS - 1])

        gen_start = test_start - INPUT_STEPS + k
        gen_end = test_start + k - 1
        gen_window = [float(v) for v in seg[gen_start:gen_end + 1]]
        gen_pred = float(seg[gen_end])  # persistence

        if k == 0:
            leak_end = test_start - 1
            leak_pred = float(seg[leak_end])
        else:
            leak_end = test_start + k + OUTPUT_STEPS - 2
            leak_pred = float(seg[leak_end])  # persistence on contaminated window
        leak_start = leak_end - INPUT_STEPS + 1
        leak_window = [float(v) for v in seg[leak_start:leak_end + 1]]

        leaked_start = test_start + k
        leaked_end = test_start + k + OUTPUT_STEPS - 2
        leaked_vals = [float(v) for v in seg[leaked_start:leaked_end + 1]]

        iterations.append({
            "k": k,
            "actual": actual,
            "gen_start": int(gen_start), "gen_end": int(gen_end),
            "gen_window": gen_window,
            "gen_pred": gen_pred,
            "leak_start": int(leak_start), "leak_end": int(leak_end),
            "leak_window": leak_window,
            "leaked_start": int(leaked_start), "leaked_end": int(leaked_end),
            "leaked_vals": leaked_vals,
            "leak_pred": leak_pred,
        })

    gen_preds = np.array([it["gen_pred"] for it in iterations])
    leak_preds = np.array([it["leak_pred"] for it in iterations])
    actuals = np.array([it["actual"] for it in iterations])
    metrics["genuine"]["mae"] = mae(gen_preds, actuals)
    metrics["genuine"]["rmse"] = rmse(gen_preds, actuals)
    metrics["leaky"]["mae"] = mae(leak_preds, actuals)
    metrics["leaky"]["rmse"] = rmse(leak_preds, actuals)

    real_model = {
        "test_split": {
            "genuine": {"mae": 0.9381, "rmse": 1.1748, "r2": 0.286},
            "leaky": {"mae": 0.7655, "rmse": 1.0284, "r2": 0.453},
        },
        "full_dataset": {
            "genuine": {"mae": 0.9025, "rmse": 1.2190, "r2": 0.469},
            "leaky": {"mae": 0.7279, "rmse": 0.9796, "r2": 0.657},
        },
    }

    payload = {
        "seg": [float(v) for v in seg],
        "labels": labels,
        "input_steps": INPUT_STEPS,
        "output_steps": OUTPUT_STEPS,
        "test_start": test_start,
        "iterations": iterations,
        "metrics": metrics,
        "real_model": real_model,
    }

    html = TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
    OUT.write_text(html, encoding="utf-8")
    print(f"written {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"persistence demo MAE  genuine={metrics['genuine']['mae']:.4f} "
          f"leaky={metrics['leaky']['mae']:.4f}")
    print(f"persistence demo RMSE genuine={metrics['genuine']['rmse']:.4f} "
          f"leaky={metrics['leaky']['rmse']:.4f}")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vazamento de dados no rolling forecast — demonstração didática</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
<style>
  :root{--ok:#2e7d32;--bad:#c62828;--acc:#1565c0;--bg:#f6f8fb;--card:#fff;--ink:#1a1a2e}
  *{box-sizing:border-box}
  body{margin:0;font:15px/1.6 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
       background:var(--bg);color:var(--ink)}
  header{background:linear-gradient(135deg,#1a1a2e,#2b3a67);color:#fff;padding:34px 24px 26px}
  header h1{margin:0 0 6px;font-size:26px}
  header p{margin:0;opacity:.9;max-width:900px}
  main{max-width:1180px;margin:0 auto;padding:22px 18px 60px}
  .card{background:var(--card);border:1px solid #e2e8f0;border-radius:12px;
        padding:20px 22px;margin:22px 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}
  h2{font-size:19px;margin:0 0 12px;border-left:4px solid var(--acc);padding-left:10px}
  h3{font-size:15px;margin:16px 0 6px}
  code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
  pre{background:#0f172a;color:#e2e8f0;padding:14px 16px;border-radius:8px;
      overflow-x:auto;font-size:13px;line-height:1.5}
  pre .leak{color:#ff8a80;font-weight:600}
  pre .fix{color:#a5d6a7;font-weight:600}
  pre .cm{color:#64748b;font-style:italic}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  @media(max-width:820px){.grid2{grid-template-columns:1fr}}
  .legend{font-size:13px;display:flex;flex-wrap:wrap;gap:16px;margin:10px 0 4px}
  .legend span{display:inline-flex;align-items:center;gap:6px}
  .dot{width:14px;height:14px;border-radius:4px;display:inline-block}
  .slide{width:100%;margin-top:8px}
  .slide output{font-weight:700;color:var(--acc)}
  table{border-collapse:collapse;width:100%;font-size:13.5px;margin-top:8px}
  th,td{border:1px solid #dbe3ee;padding:7px 10px;text-align:right}
  th{background:#eef2f9;text-align:center}
  td.l,th.l{text-align:left}
  .ok{color:var(--ok);font-weight:700}
  .bad{color:var(--bad);font-weight:700}
  .note{font-size:13px;color:#475569;margin-top:8px}
  .pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:700}
  .pill-ok{background:#e3f4e6;color:var(--ok)}
  .pill-bad{background:#fdeaea;color:var(--bad)}
  footer{color:#64748b;font-size:13px;text-align:center;padding:20px}
  .steps{list-style:none;counter-reset:s;padding:0;margin:0}
  .steps li{counter-increment:s;padding:9px 0 9px 44px;position:relative;border-bottom:1px dashed #e2e8f0}
  .steps li::before{content:counter(s);position:absolute;left:6px;top:10px;width:26px;height:26px;
       background:var(--acc);color:#fff;border-radius:50%;display:flex;align-items:center;
       justify-content:center;font-size:13px;font-weight:700}
</style>
</head>
<body>
<header>
  <h1>Vazamento de dados no <em>rolling forecast</em></h1>
  <p>Por que o protocolo "leaky" (encoder que ingere o valor real futuro) infla as
  métricas — demonstrado com dados reais de <code>data/dataset.csv</code> (vento ws100).</p>
</header>
<main>

  <div class="card">
    <h2>1. A linha culpada</h2>
    <p>No protocolo com vazamento, ao final de cada iteração o histórico é atualizado com o
    <b>valor real do horizonte que acabou de ser previsto</b>:</p>
<pre>
for i in range(len(test_data) - output_steps + 1):
    ...
    last_actual = test_data[i + output_steps - 1]   <span class="cm"># real value at the forecasted horizon</span>
    actuals.append(last_actual)

    <span class="cm"># Vazamento: injeta o GABARITO (valor real futuro) no histórico do encoder</span>
    window = np.vstack([window, <span class="leak">test_data[i + output_steps - 1]</span>])
    window = window[1:]
</pre>
    <p><span class="pill pill-bad">LEAKY</span> Na iteração <code>i+1</code> o encoder passa a "ver"
    o valor real futuro como se fosse passado conhecido. É <em>teacher forcing</em> / <em>look-ahead bias</em>:
    para <code>output_steps=12</code>, até 11 amostras reais futuras ficam escondidas dentro do histórico.</p>
    <p><span class="pill pill-ok">GENUÍNO</span> O correto é avançar a janela apenas com o valor recém-observado:</p>
<pre>
    window = np.vstack([window, <span class="fix">test_data[i]</span>])   <span class="cm"># só o valor que acabou de acontecer</span>
    window = window[1:]
</pre>
  </div>

  <div class="card">
    <h2>2. Veja a janela deslizando (interativo)</h2>
    <p>O gráfico mostra a série de vento <code>ws100</code> (segmento real). Arraste o
    <b>deslizador</b> para andar pelas iterações.</p>
    <div class="legend">
      <span><span class="dot" style="background:#2e7d32"></span> encoder window (histórico conhecido)</span>
      <span><span class="dot" style="background:#c62828"></span> valores futuros reais vazados (o "gabarito")</span>
      <span><span class="dot" style="background:#1565c0"></span> ponto de previsão (persistência)</span>
      <span><span class="dot" style="background:#f9a825"></span> alvo real (actual)</span>
    </div>
    <div class="grid2">
      <div><h3>GENUÍNO — janela só com o passado</h3><div id="chart-gen" style="height:340px"></div></div>
      <div><h3>LEAKY — janela contaminada com futuro</h3><div id="chart-leak" style="height:340px"></div></div>
    </div>
    <input type="range" id="slider" class="slide" min="0" max="0" value="0">
    <div style="display:flex;justify-content:space-between;font-size:13px;color:#475569">
      <span>iteração <output id="slider-out">0</output></span>
      <span>← deslize para "vazar" cada vez mais futuro →</span>
    </div>
    <p class="note">No LEAKY, note o trecho <b>vermelho</b> à frente da origem da previsão:
    é informação real do futuro dentro da entrada do modelo. Na primeira iteração a janela ainda é
    honesta; a contaminação cresce a partir da segunda.</p>
  </div>

  <div class="card">
    <h2>3. Consequência prática</h2>
    <p>Sem treinar nenhuma rede, usamos <b>persistência</b> (prever o último valor da janela) como
    "modelo" para isolar o efeito do vazamento. No LEAKY o último valor da janela é quase o próprio
    gabarito — até um modelo que não aprendeu nada fica parecendo excelente.</p>
    <div class="grid2">
      <div><h3>Previsões vs. reais (persistência)</h3><div id="chart-preds" style="height:340px"></div></div>
      <div>
        <h3>Métricas da demo (persistência)</h3>
        <div id="chart-metrics" style="height:220px"></div>
        <h3>Métricas do modelo real (tcn_bi, test split)</h3>
        <table>
          <tr><th class="l">métrica</th><th>GENUÍNO</th><th>LEAKY</th></tr>
          <tr><td class="l">MAE (m/s)</td><td class="bad">0.9381</td><td class="ok">0.7655</td></tr>
          <tr><td class="l">RMSE (m/s)</td><td class="bad">1.1748</td><td class="ok">1.0284</td></tr>
          <tr><td class="l">R²</td><td class="bad">0.286</td><td class="ok">0.453</td></tr>
        </table>
        <p class="note">O vazamento corta ~18% do MAE e sobe o R² de 0.29 → 0.45.
        No dataset completo (docs/rolling_forecast_profundo.md): MAE 0.9025 → 0.7279, R² 0.469 → 0.657.</p>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>4. Por que isso é perigoso</h2>
    <ul class="steps">
      <li>O modelo parece melhor do que é — e você decide implantá-lo com base em números falsos.</li>
      <li>Em produção não existe o valor real futuro: a janela só tem o passado. O desempenho real cai.</li>
      <li>Vaza por "padrão" em todos os meses e horas (docs): o ganho é artificial, não é habilidade do modelo.</li>
      <li>Números históricos antigos do projeto (ex. MAE 0.17) foram produzidos sob protocolo leaky —
          por isso os valores honestos atuais (~0.9) pareciam "piores".</li>
    </ul>
  </div>

  <div class="card">
    <h2>5. Como garantir avaliação honesta</h2>
    <ol>
      <li>Cada janela do encoder deve ser <b>função apenas do histórico conhecido</b> até a origem da previsão.</li>
      <li>Atualizar o histórico apenas com <code>test_data[i]</code> (valor recém-observado), nunca com o valor do horizonte.</li>
      <li>Usar a implementação batched <code>rolling_forecast</code> do
          <code>TorchSeq2SeqWrapper</code> (src/models/s2s_wrapper.py) — janelas via <code>sliding_window_view</code>, sem futuro.</li>
      <li>Comparar sempre contra a <b>persistência</b> e reportar os números genuínos como referência.</li>
    </ol>
  </div>

</main>
<footer>
  Gerado por <code>tests/make_leak_demo_html.py</code> · dados: data/dataset.csv (ws100, últimas 132 linhas) ·
  input_steps=12, output_steps=12 · plotly.js via CDN
</footer>

<script>
const D = __PAYLOAD__;
const seg = D.seg, lab = D.labels, n = seg.length;
const I = D.input_steps, O = D.output_steps, TS = D.test_start;
const K = D.iterations.length;
const x = seg.map((_, i) => i);

const baseLineGen = {x, y: seg, mode: 'lines', line: {color: '#94a3b8', width: 1.5},
                     showlegend: false, hoverinfo: 'skip'};
const baseLineLeak = {x, y: seg, mode: 'lines', line: {color: '#94a3b8', width: 1.5},
                      showlegend: false, hoverinfo: 'skip'};

const predTraces = [
  {x: lab.slice(TS + O - 1), y: D.iterations.map(t => t.actual),
   mode: 'lines', name: 'real (actual)', line: {color: '#f9a825', width: 2}},
  {x: lab.slice(TS + O - 1), y: D.iterations.map(t => t.gen_pred),
   mode: 'lines', name: 'previsão GENUÍNA', line: {color: '#2e7d32', width: 2.5}},
  {x: lab.slice(TS + O - 1), y: D.iterations.map(t => t.leak_pred),
   mode: 'lines', name: 'previsão LEAKY', line: {color: '#c62828', width: 2.5}},
];

function band(x0, x1, color) {
  const y0 = Math.min(...seg) - 0.5, y1 = Math.max(...seg) + 0.5;
  return {x: [x0, x0, x1, x1], y: [y0, y1, y1, y0], fill: 'toself',
          fillcolor: color, line: {width: 0}, showlegend: false, hoverinfo: 'skip'};
}
function segLine(lo, hi, color) {
  return {x: x.slice(lo, hi + 1), y: seg.slice(lo, hi + 1), mode: 'lines',
          line: {color, width: 4}, showlegend: false, hoverinfo: 'skip'};
}
function vline(pos, color) {
  return {x: [pos, pos], y: [Math.min(...seg) - 0.5, Math.max(...seg) + 0.5],
          mode: 'lines', line: {color, width: 1.5, dash: 'dot'},
          showlegend: false, hoverinfo: 'skip'};
}
function mark(pos, val, color) {
  return {x: [pos], y: [val], mode: 'markers',
          marker: {color, size: 11, symbol: 'circle', line: {color: '#fff', width: 1}},
          showlegend: false, hoverinfo: 'skip'};
}

// Per-iteration traces: gen part (5) + leak part (7)
const perIterGen = [], perIterLeak = [];
D.iterations.forEach((t, i) => {
  const g = [
    band(t.gen_start, t.gen_end, 'rgba(46,125,50,0.13)'),
    segLine(t.gen_start, t.gen_end, '#2e7d32'),
    vline(t.gen_end, '#1565c0'),
    vline(t.gen_end + O, '#f9a825'),
    mark(t.gen_end, t.gen_pred, '#1565c0'),
  ];
  const l = [
    band(t.leak_start, t.leak_end, 'rgba(46,125,50,0.13)'),
    segLine(t.leak_start, t.leak_end, '#2e7d32'),
    band(t.leaked_start, t.leaked_end, 'rgba(198,40,40,0.22)'),
    segLine(t.leaked_start, t.leaked_end, '#c62828'),
    vline(t.leak_end, '#1565c0'),
    vline(t.leak_end + 1, '#f9a825'),
    mark(t.leak_end, t.leak_pred, '#1565c0'),
  ];
  g.forEach((tr, j) => { tr.visible = (i === 0); });
  l.forEach((tr, j) => { tr.visible = (i === 0); });
  perIterGen.push(g);
  perIterLeak.push(l);
});

const genTraces = [baseLineGen].concat(perIterGen.flat());
const leakTraces = [baseLineLeak].concat(perIterLeak.flat());

// Metrics bar chart (persistence demo)
const m = D.metrics;
const metTraces = [
  {x: ['MAE', 'RMSE'], y: [m.genuine.mae, m.genuine.rmse],
   name: 'GENUÍNO', type: 'bar', marker: {color: '#2e7d32'},
   text: [m.genuine.mae.toFixed(3), m.genuine.rmse.toFixed(3)], textposition: 'outside'},
  {x: ['MAE', 'RMSE'], y: [m.leaky.mae, m.leaky.rmse],
   name: 'LEAKY', type: 'bar', marker: {color: '#c62828'},
   text: [m.leaky.mae.toFixed(3), m.leaky.rmse.toFixed(3)], textposition: 'outside'},
];

const xTicks = {};
for (let i = 0; i < n; i += 6) xTicks[i] = lab[i];
const tickXs = Object.keys(xTicks), tickLs = Object.values(xTicks);

const common = {
  paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
  font: {family: 'ui-monospace, Menlo, Consolas, monospace', size: 11},
  hovermode: 'closest',
};
const genLayout = {
  ...common,
  xaxis: {tickvals: tickXs, ticktext: tickLs, range: [0, n - 1]},
  yaxis: {title: 'ws100 (m/s)'},
  margin: {l: 46, r: 14, t: 20, b: 40},
  showlegend: false,
};
const leakLayout = JSON.parse(JSON.stringify(genLayout));
const predsLayout = {
  ...common,
  xaxis: {tickangle: -35, tickfont: {size: 10}},
  yaxis: {title: 'ws100 (m/s)'},
  margin: {l: 46, r: 14, t: 20, b: 60},
  showlegend: true,
  legend: {orientation: 'h', y: -0.28},
};
const metLayout = {
  ...common,
  xaxis: {tickfont: {size: 11}},
  yaxis: {title: 'm/s', range: [0, Math.max(m.genuine.rmse, m.leaky.rmse) * 1.35]},
  margin: {l: 46, r: 14, t: 20, b: 40},
  showlegend: true,
  legend: {orientation: 'h', y: -0.25},
};

Plotly.newPlot('chart-gen', genTraces, genLayout, {responsive: true});
Plotly.newPlot('chart-leak', leakTraces, leakLayout, {responsive: true});
Plotly.newPlot('chart-preds', predTraces, predsLayout, {responsive: true});
Plotly.newPlot('chart-metrics', metTraces, metLayout, {responsive: true});

const slider = document.getElementById('slider');
slider.max = K - 1;
const out = document.getElementById('slider-out');

function applyStep(i) {
  const gvis = genTraces.map((_, idx) => idx === 0 || (idx >= 1 + i * 5 && idx < 1 + (i + 1) * 5));
  const lvis = leakTraces.map((_, idx) => idx === 0 || (idx >= 1 + i * 7 && idx < 1 + (i + 1) * 7));
  Plotly.restyle('chart-gen', {visible: gvis});
  Plotly.restyle('chart-leak', {visible: lvis});
  out.value = i;
}

slider.addEventListener('input', () => applyStep(parseInt(slider.value, 10)));
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()