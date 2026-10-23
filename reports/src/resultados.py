"""Renderizador padrao dos relatorios de resultados LiDAR.

Converte a run do banco de predições (data/predictions.db, resolvida por
report_id) ou, na falta dela, o diretorio de saida do protocolo antigo
(data/results/report_*/: evaluation.json + images/) em uma pagina
padronizada do hub, com tabelas e figuras exportaveis para LaTeX
(steam/sbsr/generic).

Usado pelos report.py em reports/lidar/<experimento>/.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src import components as C  # noqa: E402

# Popups LaTeX das páginas de resultados abrem em SBSR por default (o
# formato steam usa \rowcolor/colortbl, que o template do SBSR não carrega).
FORMATS = ("sbsr", "steam", "generic")

FIGURE_CAPTIONS = {
    "01_split_temporal": "Divisão temporal do dataset LiDAR (treino/validação/teste)",
    "02_comparacao_mse_rmse": "Comparação de MSE e RMSE (todos os horizontes) — modelos e baselines",
    "02b_erro_geral": "Resumo de erro (MAE, MSE, R²) e loss final de treino",
    "03_mse_r2": "MSE e R² — todos os horizontes vs rolling (h=36)",
    "04_radar": "Perfil dos métodos (eixos normalizados)",
    "05_mae_por_horizonte_real": "MAE por horizonte vs dado real (ws100 bruto)",
    "05_mae_por_horizonte_wavelet": "MAE por horizonte vs alvo wavelet",
    "06_heatmap_mse_horizonte": "Heatmap de MSE (m/s)² por modelo e horizonte",
    "06_heatmap_r2_horizonte": "Heatmap de R² por modelo e horizonte",
    "07_fan_chart": "Previsões de 6 h à frente em 6 origens do teste (fan chart)",
    "08_rolling_forecast_real": "Previsão autônoma (rolling h=36) vs observado — teste",
    "08_rolling_forecast_wavelet": "Previsão autônoma (rolling h=36) vs alvo wavelet",
    "09_diagnostico_acf_pacf": "ACF/PACF dos resíduos rolling (h=36)",
    "09_diagnostico_dispersao": "Previsto vs observado por modelo (todos os horizontes)",
    "09_diagnostico_histograma": "Distribuição dos erros por família de modelo",
    "09_diagnostico_vies": "Viés por regime de vento (erro médio por faixa)",
    "10_piores_predicoes_lstm": "Piores previsões completas da LSTM (36 passos)",
    "11_early_stopping": "Épocas executadas vs orçamento e val_loss (best vs final)",
    "12_tempo_treinamento": "Tempo total de treinamento por modelo",
    "13_inferencia": "Inferência em lote, latência single-shot e tamanho do modelo",
    "14_custo_beneficio": "Custo de treinamento × erro (área = nº de parâmetros)",
    "15_arquitetura": "Diagrama de arquitetura",
    "16_significancia_dm": "Teste de Diebold–Mariano por horizonte (matrizes linha × coluna)",
    "16b_significancia_dm_pvalor": "p-valores do teste de Diebold–Mariano",
    "16_wavelet_vs_real": "Erro vs alvo wavelet vs erro ws100 bruto",
    "16c_significancia_dm_ab_loss": "DM das variantes de loss (A/B de função de perda)",
    "16d_mcs": "Model Confidence Set — p-valor de exclusão por modelo (agregado sobre os 36 horizontes)",
}

METRIC_ORDER = ("mse", "rmse", "mae", "r2")

# Grupos temáticos de figuras com o texto que estrutura a análise.
FIGURE_THEMES = [
    ("Protocolo e divisão dos dados",
     ("01_split_temporal",),
     "A série LiDAR de 10 min é dividida <strong>sem embaralhamento</strong> "
     "(80/20 na reamostragem, 75/20/5 no protocolo homologado): o treino "
     "vai sempre antes do teste, simulando o deployment real. Ao ler a "
     "figura, observe que o regime de vento do período de teste pode ser "
     "mais calmo ou mais volátil que o do treino — parte da diferença de "
     "erro entre execuções vem daí, não dos modelos."),
    ("Comparação global de erro",
     ("02_comparacao_mse_rmse", "02b_erro_geral", "03_mse_r2", "04_radar",
      "16_wavelet_vs_real"),
     "Aqui os modelos disputam em duas condições: <strong>todos os "
     "horizontes</strong> (cada previsão avaliada no passo em que acerta o "
     "timestamp) e <strong>rolling autônomo</strong> (o modelo alimenta a "
     "própria previsão por 6 h — condição mais difícil, com acúmulo de "
     "erro). Barras hachuradas são baselines (ARIMA, Random Forest, "
     "persistência); menor MSE/RMSE e maior R² indicam melhor modelo. No "
     "radar, cada eixo é normalizado e aponta para fora quando melhor — "
     "área maior = perfil mais consistente. O quadro typical: TCN à frente "
     "em erro, bidirecionalidade com ganho pequeno."),
    ("Comportamento por horizonte e significância",
     ("05_mae_por_horizonte_real", "05_mae_por_horizonte_wavelet",
      "06_heatmap_mse_horizonte", "06_heatmap_r2_horizonte",
      "16_significancia_dm", "16b_significancia_dm_pvalor",
      "16c_significancia_dm_ab_loss", "16d_mcs"),
     "O erro cresce com o horizonte — o esperado —, mas o ponto crítico é "
     "<strong>onde cada modelo cruza a persistência</strong>: passos curtos "
     "(1–6) tendem a ficar aquém do ingênuo, e a vantagem aparece nos "
     "horizontes médios (6–24). Os heatmaps mostram a leitura célula a "
     "célula (verde = melhor). As matrizes de Diebold–Mariano fecham o "
     "critério estatístico: célula positiva = modelo da linha com perda "
     "maior (pior); com poucas células significativas (p &lt; 0,05), as "
     "diferenças entre wrappers são empate técnico — o que orienta a "
     "escolha por custo de treino/inferência em vez de só por MSE."),
    ("Trajetórias e previsões autônomas",
     ("07_fan_chart", "08_rolling_forecast_real",
      "08_rolling_forecast_wavelet", "10_piores_predicoes_lstm"),
     "O fan chart abre 6 origens do teste e mostra a trajetória completa "
     "de 6 h de cada modelo — repare na divergência após os primeiros "
     "passos e na âncora comum (última observação). O rolling contínuo "
     "comprime 6 semanas de previsões autônomas; o painel inferior (resíduo "
     "médio móvel de 2 h) expõe <strong>quando o modelo vicia</strong>: "
     "faixas persistentes acima de zero = subestimação daquele evento. As "
     "piores trajetórias da LSTM são, quase sempre, rampas de vento "
     "(mudanças rápidas), o limite conhecido de modelos treinados em MSE."),
    ("Diagnóstico dos resíduos",
     ("09_diagnostico_dispersao", "09_diagnostico_histograma",
      "09_diagnostico_vies", "09_diagnostico_acf_pacf"),
     "Quatro lentes sobre o erro: (1) dispersão previsto × observado — "
     "nuvem colada na diagonal = calibrado, achatada = regressão à média; "
     "(2) histograma dos erros — centrado em 0 e simétrico = sem viés "
     "global (μ e σ no rótulo); (3) viés por regime — erro médio por faixa "
     "de vento, onde fica claro se o modelo superestima vento fraco e "
     "subestima vento forte; (4) ACF/PACF dos resíduos rolling — barras "
     "fora do intervalo de confiança indicam estrutura temporal que o "
     "modelo não capturou."),
    ("Custos de treino, inferência e arquitetura",
     ("11_early_stopping", "12_tempo_treinamento", "13_inferencia",
      "14_custo_beneficio", "15_arquitetura"),
     "Critério de desempate prático: épocas executadas vs orçamento (sem "
     "early stopping na reamostragem, o orçamento é sempre integral), "
     "tempo de GPU por modelo, latência single-shot (requisito de "
     "operação), throughput em lote e tamanho do <code>model.keras</code>. "
     "O gráfico custo × erro resume a decisão: pontos à esquerda e embaixo "
     "dominam (menos treino, menos erro); a área marca o número de "
     "parâmetros. Os diagramas de arquitetura documentam shapes e "
     "ativações de cada bloco."),
]


def _fmt(x, nd=3):
    if x is None:
        return "—"
    try:
        if x != x:  # NaN
            return "—"
    except TypeError:
        return str(x)
    return f"{x:.{nd}f}".replace(".", ",")


def _figure_caption(png: Path) -> str:
    stem = png.stem
    caption = FIGURE_CAPTIONS.get(stem) or FIGURE_CAPTIONS.get(
        "_".join(stem.split("_")[:2]))
    if caption is None:
        caption = stem.replace("_", " ").strip()
        return caption[0].upper() + caption[1:]
    return caption


def _themed_figures(img_dir: Path, prefix: str) -> list[str]:
    """Seções temática: cada grupo de figuras com o texto da análise."""
    prefix = prefix.rstrip("/") + "/"
    pngs = sorted(img_dir.glob("*.png"))
    by_stem = {p.stem: p for p in pngs}
    used: set[str] = set()
    sections = []
    for title, stems, text in FIGURE_THEMES:
        group = []
        for key in stems:
            matches = [s for s in by_stem if s == key
                       or s.startswith(key + "_")]
            for stem in matches:
                if stem in used:
                    continue
                used.add(stem)
                png = by_stem[stem]
                label = f"fig:{stem}"
                group.append(C.figure(
                    f"{prefix}{png.name}", _figure_caption(png), label=label,
                    width=0.8 if stem.startswith(("15", "16")) else 0.95))
        if group:
            sections.append(C.section(
                title, C.grid("".join(group)), sub=text))
    leftover = [p for p in pngs if p.stem not in used]
    if leftover:
        group = [C.figure(f"{prefix}{p.name}", _figure_caption(p),
                          label=f"fig:{p.stem}")
                 for p in leftover]
        sections.append(C.section("Outras figuras",
                                  C.grid("".join(group)),
                                  sub="figuras auxiliares do relatório "
                                      "original.",
                                  sec_id=None))
    return sections


def _summary_text(models: list) -> str:
    """Parágrafo-síntese gerado a partir das métricas."""
    if not models:
        return ""
    best = models[0]
    worst = models[-1]
    a, b = best["all"], worst["all"]

    def br(x, nd=3):
        return f"{x:.{nd}f}".replace(".", ",")

    base = (
        f"<strong>{C.esc(best.get('short', '?'))}</strong> lidera com MSE "
        f"{br(a['mse'])} (m/s)² e R² {br(a.get('r2', 0))} em todos os "
        f"horizontes, contra {br(b['mse'])} / {br(b.get('r2', 0))} do pior "
        f"({C.esc(worst.get('short', '?'))})."
    )
    rolling = best.get("rolling") or {}
    if not rolling.get("mse"):
        return base
    queda = br(100 * (1 - rolling["mse"] / a["mse"]), 0) if a["mse"] else "—"
    return (
        f"{base} Na condição mais dura — rolling autônomo de 6 h — o "
        f"melhor MSE é {br(rolling['mse'])}, e a diferença relativa "
        f"({queda}%) quantifica o custo do acúmulo de erro na alimentação "
        "autônoma. Com diferenças entre wrappers frequentemente não-"
        "significativas pelo teste DM, a leitura recomendada é conjunta: "
        "erro (tabelas abaixo), significância (matrizes DM) e custo "
        "(treino/inferência)."
    )


def _hp_section(models: list) -> str:
    """Tabela de hiperparâmetros por modelo (requer run do banco)."""
    entries = []
    for m in models:
        hp = m.get("hp") or {}
        feats = m.get("features") or []
        entries.append((m.get("short", m.get("name", "?")), hp, len(feats),
                        m.get("seed"), m.get("loss"), feats))
    if not any(e[1] for e in entries):
        return ""

    def cell(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.6g}".replace(".", ",")
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(cell(x) for x in v) + "]"
        return str(v)

    keys = []
    for _, hp, *_ in entries:
        for k in hp:
            if k not in keys:
                keys.append(k)
    rows = [("seed", *(cell(e[3]) for e in entries)),
            ("colunas de entrada",
             *(str(e[2]) if e[2] else "—" for e in entries)),
            ("loss", *(cell(e[4]) for e in entries))]
    rows += [(k, *(cell(e[1].get(k)) for e in entries)) for k in keys]
    hp_table = C.table(
        ["Parâmetro", *(e[0] for e in entries)], rows,
        "Hiperparâmetros de treino de cada modelo da rodada "
        "(seed, colunas de entrada e o dicionário gravado no "
        "model.json / banco de runs).", formats=FORMATS,
        label="tab:hiperparametros")

    feat_rows = [
        ("ws40–ws140", "velocidade horizontal do vento em 11 alturas "
                       "(40–140 m, passo de 10 m)"),
        ("v40–v140", "componente v do vento nas mesmas 11 alturas"),
        ("dir_h_sin/cos", "direção do vento codificada em seno/cosseno nas "
                          "11 alturas (22 colunas)"),
        ("hour_sin/cos", "hora do dia codificada ciclicamente (2 colunas)"),
        ("cis1–cis19", "cisalhamento vertical do perfil do vento (19 "
                       "colunas) — somente nos LSTMs"),
        ("humid, temp", "umidade relativa e temperatura de superfície "
                        "(2 colunas)"),
    ]
    feat_table = C.table(
        ["Variável", "Descrição"], feat_rows,
        "Grupos de variáveis de entrada (44 + 2 colunas comuns a todos; "
        "os LSTMs somam o cisalhamento, totalizando 67 colunas contra 48 "
        "das TCNs).", formats=FORMATS,
        label="tab:entradas")

    return C.section(
        "Hiperparâmetros",
        hp_table + feat_table,
        sub="TCNs herdam os HPs da canônica 16/10 (dilatações estáticas C′ "
            "com <code>encoder_dil_depth</code>/<code>decoder_dil_depth</code> "
            "e kernel/stacks fixos); LSTMs usam os HPs do HPO 80/20. A perda "
            "é MSE pura em todos os wrappers. As colunas de entrada seguem o "
            "<code>feature_keep</code> por wrapper do lineup misto "
            "(48/48/67/67); sobre elas o wrapper ainda acrescenta os canais "
            "de alvo (<code>ws100_wavelet</code> e último observado) ao "
            "encoder.",
        sec_id="hiperparametros")


def _mcs_section(src: Path) -> str:
    """Seção do Model Confidence Set (quando mcs.json existe na fonte)."""
    path = src / "mcs.json"
    if not path.is_file():
        return ""
    mcs = json.loads(path.read_text(encoding="utf-8"))
    if not mcs.get("exclusion_p"):
        return ""
    rows = []
    for label, p in sorted(mcs["exclusion_p"].items(), key=lambda kv: kv[1]):
        inside = p >= mcs.get("alpha", 0.10)
        p_txt = _fmt(p, 3) if p >= 0.001 else "&lt; 0,001"
        rows.append((label, _fmt(mcs.get("mse", {}).get(label), 3), p_txt,
                     "pertence ao MCS" if inside else "excluído"))
    note = (
        f"perda de cada origem = média do erro quadrático nos 36 horizontes "
        f"(mesmo agregado do DM); p-valores por <i>stationary bootstrap</i> "
        f"({mcs.get('n_boot', 2000)} reamostragens, bloco médio "
        f"{mcs.get('mean_block', 36)}); α = "
        f"{_fmt(mcs.get('alpha', 0.10), 2)}."
    )
    if mcs.get("n_origins"):
        note += (f" Comparação justa entre todos os métodos: "
                 f"{mcs['n_origins']} origens comuns (inclui ARIMA/RF).")
    return C.section(
        "Model Confidence Set (agregado sobre a predição inteira)",
        C.table(["Modelo", "MSE (m/s)²", "p-valor de exclusão", "Status"],
                rows, "Model Confidence Set (Hansen, Lunde & Nason, 2011): "
                "modelos no conjunto são estatisticamente indistinguíveis "
                "quando a comparação agrega toda a trajetória de 36 passos.",
                formats=FORMATS, label="tab:mcs",
                highlight_rows=[i for i, r in enumerate(rows)
                                if "pertence" in r[3]],
                note=note),
        sub="A leitura conjunta com o DM: o DM pareado pergunta 'A difere de "
            "B?'; o MCS elimina de uma vez todos os modelos que perdem e "
            "devolve o conjunto dos empatados — nele, a escolha é por custo "
            "(treino/inferência), não por MSE. A figura do MCS está na "
            "seção de significância, abaixo.",
        sec_id="mcs")


def _run_from_db(report_key: str):
    """Run canônica do banco de predições para este relatório.

    Retorna (run_row, results_dict) ou (None, None) quando o banco não
    existe / não tem run para o ``report_id`` (ex.: ``lidar/8020``).
    """
    if not report_key:
        return None, None
    from src import predictions_db

    try:
        conn = predictions_db.connect(read_only=True)
    except (FileNotFoundError, sqlite3.Error):
        return None, None
    try:
        run = predictions_db.run_by_report(conn, report_key)
        if run is None:
            return None, None
        return run, predictions_db.results_dict(conn, run["id"])
    finally:
        conn.close()


def build_report(ctx, *, source_dir: str, title: str, subtitle: str = "",
                 badges=(), protocol_note: str = "") -> str:
    """Gera o corpo HTML padronizado do relatorio de resultados.

    A fonte primária é o banco de runs (data/predictions.db, resolvido por
    ``<categoria>/<nome>``); sem run gravada, cai para o evaluation.json da
    pasta fonte (compatibilidade com rodadas antigas).
    """
    src = PROJECT_ROOT / source_dir
    report_key = f"{ctx.category}/{ctx.name}"
    run_row, db_data = _run_from_db(report_key)
    evaluation_path = src / "evaluation.json"
    if db_data is None and not evaluation_path.is_file():
        return (f'<section><h2>Sem dados</h2><p>Nenhuma run de '
                f'<code>{C.esc(report_key)}</code> no banco '
                f'(<code>data/predictions.db</code>) e <code>{C.esc(source_dir)}/'
                f'evaluation.json</code> não encontrado. Use o botão '
                f'<strong>Regenerar</strong> (servidor ativo) ou rode '
                f'<code>tests/relatorio_resultados.py</code>.</p></section>')

    # figuras: copia para a pasta do relatorio (artefatos locais da pagina)
    fig_prefix = "figures"
    local_figs = ctx.dir / fig_prefix
    if (src / "images").is_dir():
        if local_figs.exists():
            shutil.rmtree(local_figs)
        shutil.copytree(src / "images", local_figs)

    data = db_data if db_data is not None \
        else json.loads(evaluation_path.read_text(encoding="utf-8"))
    models = sorted(data.values(), key=lambda m: m.get("all", {})
                    .get("mse", 9e9))

    # ---- cards
    best = models[0]
    total_params = sum(m.get("params", 0) for m in models)
    extra_badges = [source_dir]
    if run_row is not None:
        extra_badges.append(
            f"banco: run #{run_row['id']} · {run_row['category']}")
    parts = [C.page_header(
        title, subtitle, badges=list(badges) + extra_badges),
        C.cards([
            (len(models), "modelos seq2seq"),
            (f"{best.get('short', '?')} · {_fmt(best['all']['mse'])}",
             f"melhor MSE ({best.get('all', {}).get('samples', '?'):,} "
             f"amostras)".replace(",", ".")),
            (_fmt(min(m["all"]["r2"] for m in models), 2) + " – "
             + _fmt(max(m["all"]["r2"] for m in models), 2),
             "faixa de R² (todos os horizontes)"),
            (f"{total_params:,}".replace(",", "."), "parâmetros somados"),
        ])]

    if protocol_note:
        parts.append(C.note(protocol_note))
    if run_row is not None:
        generator = (run_row["source_file"] or run_row["source"] or "—")
        repro = (f"Modelos: <code>{run_row['models_dir'] or '—'}</code> · "
                 f"gerador: <code>{generator}</code>"
                 + (f" · saída: <code>{run_row['out_file']}</code>"
                    if run_row["out_file"] else ""))
        if run_row["source_cmd"]:
            repro += (f'<div style="font-size:11px;word-break:break-all">'
                      f"<code>{run_row['source_cmd']}</code></div>")
        parts.append(C.note(
            f"Fonte de dados: <code>data/predictions.db</code> — run "
            f"<strong>#{run_row['id']}</strong> ({run_row['category']}), "
            f"criada em {str(run_row['created_at']).replace('T', ' ')[:19]}. "
            f"Métricas de <code>run_model_stath</code>; predições linha a "
            f"linha em <code>run_model_prediction</code>.<br>{repro}"))

    # ---- síntese da análise
    parts.append(C.section(
        "Síntese da análise",
        f"<div class='markdown'>{_summary_text(models)}</div>",
        sub="leitura guiada: comece pela síntese, confirme nas tabelas e "
            "use as seções temáticas para o detalhamento figura a figura "
            "(clique nas imagens para ampliar).",
        sec_id="sintese"))

    # ---- tabela: todos os horizontes
    rows = []
    for m in models:
        a = m["all"]
        rows.append((m.get("short", m.get("name", "?")),
                     m.get("loss") or "—",
                     _fmt(a.get("mse")), _fmt(a.get("rmse")),
                     _fmt(a.get("mae")), _fmt(a.get("r2"))))
    best_mse = min(float(r[2].replace(",", ".")) for r in rows)
    highlight = [i for i, r in enumerate(rows)
                 if float(r[2].replace(",", ".")) == best_mse]
    parts.append(C.section(
        "Métricas — todos os horizontes (1–36 passos, vs ws100 bruto)",
        C.table(["Modelo", "Loss", "MSE (m/s)²", "RMSE (m/s)", "MAE (m/s)",
                 "R²"], rows,
                "Métricas de erro dos modelos seq2seq em todos os horizontes.",
                formats=FORMATS, label="tab:metricas_geral",
                highlight_rows=highlight,
                note="menor MSE destacado; avaliação sobre as janelas do "
                     "conjunto de teste."),
        sub="Cada previsão é avaliada no timestamp que ela almeja: o passo "
            "1 confere com a observação 10 min à frente, o passo 36 com a "
            "de 6 h. MSE/RMSE penalizam erros grandes (rampas de vento); "
            "MAE é a leitura em escala natural; R² mede a variância "
            "explicada — perto de 0,4–0,5 neste sítio é bom desempenho "
            "multi-passo, pois o vento a 6 h tem pouca previsibilidade "
            "determinística.",
        sec_id="metricas"))

    # ---- hiperparâmetros (requer hp gravado — run do banco)
    hp_html = _hp_section(models)
    if hp_html:
        parts.append(hp_html)

    # ---- tabela: rolling
    if all("rolling" in m for m in models):
        rows = [(m.get("short", "?"), _fmt(m["rolling"].get("mse")),
                 _fmt(m["rolling"].get("rmse")), _fmt(m["rolling"].get("mae")),
                 _fmt(m["rolling"].get("r2")))
                for m in models]
        best_roll = min(float(r[1].replace(",", ".")) for r in rows)
        highlight = [i for i, r in enumerate(rows)
                     if float(r[1].replace(",", ".")) == best_roll]
        parts.append(C.section(
            "Rolling forecast autônomo (h = 36 passos ≈ 6 h)",
            C.table(["Modelo", "MSE (m/s)²", "RMSE (m/s)", "MAE (m/s)", "R²"],
                    rows, "Métricas do rolling forecast autônomo (horizonte "
                    "de 36 passos).", formats=FORMATS,
                    label="tab:metricas_rolling",
                    highlight_rows=highlight,
                    note="o modelo vê apenas o histórico; nenhuma entrada "
                         "futura real."),
            sub="Aqui cada previsão de 6 h parte de uma fronteira e a "
                "janela anda passo a passo sem nunca ver dados reais do "
                "futuro — o erro de um passo contamina os seguintes. É a "
                "métrica mais próxima da operação real e a que mais "
                "diferencia arquiteturas: escalas de erro maiores que a "
                "tabela anterior são normais e desejáveis de monitorar.",
            sec_id="rolling"))

    # ---- tempos
    if all("timing" in m for m in models):
        rows = [(m.get("short", "?"),
                 f"{m['timing'].get('single_shot_ms', float('nan')):.1f} ms"
                 .replace(".", ","),
                 f"{m['timing'].get('per_window_ms', float('nan')):.2f} ms"
                 .replace(".", ","),
                 f"{m['timing'].get('throughput_wps', float('nan')):.0f}",
                 f"{m.get('params', 0):,}".replace(",", "."),
                 f"{m.get('size_mb', 0):.1f} MB")
                for m in models]
        parts.append(C.section(
            "Inferência e tamanho",
            C.table(["Modelo", "Latência (1 janela)", "Por janela (lote)",
                     "Janelas/s", "Parâmetros", "model.keras"], rows,
                    "Custos de inferência e tamanho dos modelos.",
                    formats=FORMATS, label="tab:inferencia"),
            sub="Latência single-shot = custo de uma chamada do modelo para "
                "uma janela (cenário de operação online, 1 previsão de 6 h "
                "a cada ciclo); \"por janela (lote)\" é o custo amortizado "
                "processando o teste inteiro. Todos os wrappers ficam na "
                "ordem de dezenas de ms — folga larga para um ciclo de "
                "10 min em CPU/GPU embarcada.",
            sec_id="inferencia"))

    # ---- Model Confidence Set (quando o relatório de origem tem mcs.json)
    mcs_html = _mcs_section(src)
    if mcs_html:
        parts.append(mcs_html)

    # ---- figuras com análise estruturada
    if local_figs.is_dir():
        theme_sections = _themed_figures(local_figs, fig_prefix)
        parts.append(C.section(
            "Análise gráfica",
            "".join(theme_sections) or "<p>sem figuras</p>",
            sub="cada grupo traz o texto de leitura antes das figuras — "
                "clique em qualquer imagem para ampliá-la; o botão "
                "<strong>LaTeX</strong> exporta no formato do artigo.",
            sec_id="figuras"))
    return "\n".join(parts)


def report_module(*, name: str, title: str, description: str,
                  source_dir: str, subtitle: str = "", badges=(),
                  protocol_note: str = "", regenerate=None, retrain=None,
                  info=None):
    """Fábrica usada pelos report.py dos experimentos LiDAR."""
    REPORT = {
        "title": title, "description": description,
        "regenerate": regenerate, "retrain": retrain, "info": info or [],
        "source_dir": source_dir,
    }

    def build(ctx):
        return build_report(ctx, source_dir=source_dir, title=title,
                            subtitle=subtitle, badges=badges,
                            protocol_note=protocol_note)

    return REPORT, build
