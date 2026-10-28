"""Relatório da rodada do artigo SBSR — perfil 40–160 m (reval).

Lineup do artigo (decisão do autor 27/10/2026, finding
`estudos-ablacoes`): TCN/TCN-Bi 56 col (perfil 40–160 + meteo),
LSTM/LSTM-Bi 68 col (+ cis1..12), MSE pura, HPs canônicos com dilatações
C′, seed 42, split 72,5/12,5/15 com EarlyStopping + ReduceLROnPlateau.
Página padronizada a partir de data/results/report_reval_h160 +
testes de hipótese (DM/Friedman/Wilcoxon-Holm/MCS) e CD por horizonte.

    python reports/lidar/reval-h160/report.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src import components as C  # noqa: E402
from reports.src.resultados import report_module  # noqa: E402

SOURCE_DIR = PROJECT_ROOT / "data" / "results" / "report_reval_h160"
ALPHAS = (0.25, 0.10, 0.05, 0.01)


def _num(x, dec=4):
    return f"{float(x):.{dec}f}".replace(".", ",")


def _p(x):
    txt = f"{float(x):.3g}"
    if "e" in txt:
        mant, exp = txt.split("e")
        txt = f"{mant.replace('.', ',')}e{int(exp):+d}"
    return txt.replace(".", ",", 1) if "e" not in txt else txt


def _load(name):
    path = SOURCE_DIR / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _dm_table(testes):
    pares = testes.get("dm_pares") or []
    if not pares:
        return ""
    rows = [
        (p["par"], _num(p["dMSE"]), _num(p["DM"], 3), _p(p["p"]),
         _p(p["p_holm"]), p["melhor"],
         "sim" if p["rejeita"] == "sim" else "não",
         "sim" if p["rejeita_holm"] == "sim" else "não")
        for p in pares
    ]
    n = len(pares)
    return C.table(
        ["Par", "ΔMSE", "DM", "p bruto", "p (Holm)", "Melhor",
         "Rejeita H₀ (5%)", "Rejeita (Holm)"],
        rows,
        f"Teste de Diebold–Mariano pareado entre os 4 modelos da rodada "
        f"({n} pares), perdas por origem (MSE médio nos 36 passos), ajuste "
        f"HLN + HAC com lag 35; α = 0,05.",
        formats=("sbsr", "steam", "generic"), label="tab:dm_pares",
        note="H₀: E(d_t) = 0 — igual acurácia preditiva esperada entre os "
             "dois modelos. Ordenado por p-valor bruto.",
    )


def _wilcoxon_table(testes):
    pares = testes.get("wilcoxon_pares") or []
    if not pares:
        return ""
    rows = [
        (p["par"], _p(p["p"]), _p(p["p_holm"]), p["melhor"],
         "sim" if p["rejeita_holm"] == "sim" else "não")
        for p in pares
    ]
    n = len(pares)
    return C.table(
        ["Par", "p bruto", "p (Holm)", "Melhor", "Rejeita H₀ (Holm, 5%)"],
        rows,
        f"Wilcoxon signed-rank pareado por origem entre os 4 modelos "
        f"({n} pares) com correção de Holm; α = 0,05.",
        formats=("sbsr", "steam", "generic"), label="tab:wilcoxon_pares",
        note="H₀: as perdas dos dois modelos vêm da mesma distribuição "
             "(pareadas por origem). Ordenado por p-valor bruto.",
    )


def _friedman_table(testes):
    fr = testes.get("friedman")
    if not fr:
        return ""
    ranks = fr.get("ranks") or {}
    rows = [(k, _num(v, 3)) for k, v in
            sorted(ranks.items(), key=lambda kv: kv[1])]
    return C.table(
        ["Modelo", "Rank médio"], rows,
        f"Friedman sobre as perdas por origem ({testes.get('n_origens')} "
        f"blocos): χ² = {_num(fr['chi2'], 1)}; p = {_p(fr['p'])} — H₀ "
        f"{'rejeitada' if float(fr['p']) < 0.05 else 'não rejeitada'}.",
        formats=("sbsr", "steam", "generic"), label="tab:friedman_ranks",
        note="H₀: ranks médios iguais entre os 4 modelos (cada origem é "
             "um bloco).",
    )


def _mcs_table(multi):
    if not multi:
        return ""
    rows = []
    for r in multi.get("tabela", []):
        p = r.get("p_exclusao")
        p_txt = "≥ 0,25" if p is None else _num(p, 3)
        rows.append((r["modelo"], p_txt,
                     *[("dentro" if r.get(f"alpha_{a}") == "dentro" else
                        "fora") for a in ALPHAS]))
    return C.table(
        ["Modelo", "p-exclusão",
         *[f"α = {str(a).replace('.', ',')}" for a in ALPHAS]],
        rows,
        "Model Confidence Set em diferentes níveis de confiança "
        "(stationary bootstrap, 2.000 reamostragens, bloco médio 36).",
        formats=("sbsr", "steam", "generic"), label="tab:mcs_multialpha",
        note="H₀ do MCS: os modelos do conjunto têm igual poder preditivo. "
             "“dentro” = não excluído no nível α; p-exclusão “≥ 0,25” = "
             "nunca entrou em rodada de eliminação.",
    )


def _tests_section():
    testes = _load("testes_hipotese.json")
    multi = _load("mcs_multialpha.json")
    if not testes:
        return ""
    n = testes.get("n_origens", "—")
    fr = testes.get("friedman", {})
    return C.section(
        "Testes de hipótese",
        _dm_table(testes)
        + (_friedman_table(testes) if fr else "")
        + _wilcoxon_table(testes)
        + _mcs_table(multi),
        sub=f"Perdas por origem = MSE médio nos 36 passos ({n} origens "
            "comuns aos 4 modelos; protocolo reval 72,5/12,5/15, trim 72, "
            "baselines na mesma grade).",
        sec_id="testes-hipotese")


def _friedman_h_section():
    data = _load("friedman_horizonte.json")
    if not data:
        return ""
    rows = data.get("por_horizonte") or []
    sel = [1, 3, 6, 12, 18, 24, 30, 36]
    by_h = {r["horizon"]: r for r in rows}
    table_rows = []
    for h in sel:
        r = by_h.get(h)
        if not r:
            continue
        rank_keys = {m: r[f"rank_{m}"] for m in
                     ("LSTM", "LSTM-Bi", "TCN", "TCN-Bi", "ARIMA",
                      "Persistência")}
        best = min(rank_keys, key=rank_keys.get)
        table_rows.append((
            str(h), _num(r["chi2"], 1), _p(r["p_holm"]), best,
            _num(rank_keys["LSTM"], 2),
            "sim" if r["p_holm"] < 0.05 else "não"))
    nsig = sum(1 for r in rows if r["p_holm"] < 0.05)
    body = C.figure(
        "figures_static/friedman_horizonte.png",
        "Painel superior: MSE médio por horizonte (escala log). Painel "
        "inferior: rank médio de Friedman por horizonte (1 = melhor).",
        label="fig:friedman_horizonte")
    body += C.table(
        ["Horizonte", "χ²", "p (Holm)", "Melhor rank", "Rank LSTM",
         "H₀ rejeitada"],
        table_rows,
        f"Friedman pareado por horizonte ({data.get('n_origens')} origens "
        f"na grade comum; blocos = origens; perda = MSE no par "
        f"(origem, h)): H₀ rejeitada em {nsig}/36 horizontes (Holm 5%). "
        f"Inclui ARIMA e persistência (baselines na mesma grade, trim 72); "
        f"RF fora (sem predições por origem no diagrama).",
        formats=("sbsr", "steam", "generic"), label="tab:friedman_horizonte",
        note="Leitura: no curtíssimo prazo TCN-Bi e LSTM alternam a ponta "
             "(persistência ainda competitiva em h = 1); do meio em diante "
             "(h ≥ 12) a LSTM assume o melhor rank médio e sustenta até "
             "h = 36 — é ali que ela constrói a vantagem agregada.",
    )
    return C.section(
        "Friedman pareado por horizonte", body,
        sub="Gerado por <code>friedman_horizonte.py</code> a partir das "
            "predições do banco (run 12 = reval-h160) + baselines "
            "<code>arima_rf_reval160</code>.",
        sec_id="friedman-horizonte")


def _cd_section():
    data = _load("friedman_horizonte.json")
    cd = (data or {}).get("nemenyi")
    if not cd:
        return ""
    grupos = [" < ".join(g) for g in cd.get("grupos_nao_separaveis", [])]
    ranks = cd.get("mean_ranks", {})
    rank_rows = [(m, _num(v, 3)) for m, v in
                 sorted(ranks.items(), key=lambda kv: kv[1])]
    body = C.figure(
        "figures_static/critical_difference.png",
        "Diagrama de diferença crítica (Nemenyi, α = 0,05): modelos à "
        "esquerda têm melhor rank médio; barras ligam grupos cuja "
        "diferença de ranks é menor que a CD — estatisticamente "
        "indistinguíveis.",
        label="fig:cd_nemenyi")
    body += C.table(
        ["Método", "Rank médio"], rank_rows,
        f"Ranks médios por origem (perda = MSE médio nos 36 passos; "
        f"{cd.get('n_origens')} origens; CD de Nemenyi = "
        f"{_num(cd['cd'], 3)} para k = {cd.get('k')} métodos).",
        formats=("sbsr", "steam", "generic"), label="tab:cd_ranks",
        note="Grupos não-separáveis: " + "; ".join(grupos) + ". No topo, "
             "{TCN-Bi, LSTM} separam-se do cluster de baselines "
             "({ARIMA, Persistência}); a LSTM-Bi fecha a fileira.",
    )
    return C.section(
        "Diagrama de diferença crítica (Nemenyi)", body,
        sub="Mesma grade do Friedman global; gerado junto com a análise "
            "por horizonte (<code>friedman_horizonte.py</code>).",
        sec_id="cd-nemenyi")


def _cd_horizonte_section():
    data = _load("cd_horizonte.json")
    if not data:
        return ""
    por_h = data.get("por_horizonte") or []
    lstm_best = sum(1 for r in por_h
                    if min(r.get("mean_ranks", {}),
                           key=r.get("mean_ranks", {}).get) == "LSTM")
    body = C.figure(
        "figures_static/cd_horizonte_selecionados.png",
        "Diagrama de Diferença Crítica nos horizontes de 1 h a 6 h "
        "(h = 6, 12, 18, 24, 30, 36) com post-hoc de Nemenyi: pontos no "
        "eixo de ranks médios de Friedman (1 = melhor, à esquerda), "
        "barras ligam métodos cuja diferença de ranks médios é menor "
        "que a CD de Nemenyi do horizonte — estatisticamente "
        "indistinguíveis (α = 0,05).",
        label="fig:cd_horizonte_selecionados")
    body += C.figure(
        "figures_static/cd_horizonte.png",
        "Grade completa do mesmo diagrama para todos os horizontes "
        "(h = 1..36).",
        label="fig:cd_horizonte")
    return C.section(
        "Diferença crítica por horizonte (Friedman + Nemenyi)", body,
        sub=f"A LSTM tem o melhor rank em {lstm_best}/36 horizontes; em "
            f"h = 1 nenhum par sobrevive ao Holm (todos os métodos se "
            f"separam); nos horizontes finais o cluster de topo é "
            f"{{LSTM, TCN-Bi, TCN}} e a Persistência afunda. Gerado por "
            f"<code>reports/lidar/ab-alturas/cd_horizonte.py</code> "
            f"(run 12 do banco + baselines <code>arima_rf_reval160</code>, "
            f"trim 72).",
        sec_id="cd-horizonte")


REPORT, _base_build = report_module(
    name="reval-h160",
    title="LiDAR reval — perfil 40–160 m (rodada do artigo SBSR)",
    description="Lineup do artigo (TCN/TCN-Bi 56 col, LSTM/LSTM-Bi 68 "
                "col com cis1..12, MSE pura, HPs canônicos) no regime "
                "reval: 72,5/12,5/15 com EarlyStopping + "
                "ReduceLROnPlateau. Rodada escolhida pelo autor em "
                "27/10/2026 para alimentar o paper.",
    source_dir="data/results/report_reval_h160",
    subtitle="A/B de alturas decidiu: o corte intermediário 40–160 m "
             "(13 alturas) empata com a canônica 40–140 nas TCNs e LSTM "
             "e evita o colapso de largura da LSTM-Bi visto no 40–260. "
             "Baselines (ARIMA/RF/Persistência) calculados na MESMA "
             "grade reval (data/results/baselines/arima_rf_reval160) e "
             "trimados em 72 origens junto com as DL.",
    badges=("72,5/12,5/15", "60 épocas", "seed 42", "MSE pura",
            "EarlyStopping", "ReduceLROnPlateau", "perfil 40–160 m",
            "RODADA DO PAPER"),
    protocol_note="Modelos treinados em <code>pipeline/tmp/"
                  "trained_reval_h160</code> por <code>scripts/"
                  "lidar_reval_h160_train.py</code> (feature_keep por "
                  "wrapper, HPs da canônica <code>trained_8020_itildeq</"
                  "code>). As primeiras 72 origens do teste são "
                  "descartadas (janela cruzando a fronteira) de TODOS os "
                  "métodos.",
    regenerate=[".venv/bin/python", "reports/src/relatorio_resultados.py",
                "--models-dir", "pipeline/tmp/trained_reval_h160",
                "--out-dir", "data/results/report_reval_h160",
                "--trim-boundary-origins", "72",
                "--baselines-dir", "data/results/baselines/arima_rf_reval160",
                "--run-title",
                "LiDAR reval — perfil 40–160 m (56/68 col, oficial)"],
    retrain=[".venv/bin/python", "scripts/lidar_reval_h160_train.py"],
    info=[("Protocolo", "72,5/12,5/15 com ES+RLRP"),
          ("EarlyStopping", "val_loss, patience 12, restore_best"),
          ("ReduceLROnPlateau", "fator 0,5, patience 6, min_lr 1e-6"),
          ("HPs", "canônicos (trained_8020_itildeq)"),
          ("Loss", "MSE pura (todos)"),
          ("TCN/TCN-Bi", "perfil 40–160 (13 alturas) + humid/temp (56 col)"),
          ("LSTM/LSTM-Bi", "56 + cis1..12 (68 col)"),
          ("Baselines", "arima_rf_reval160 (mesma grade, trim 72)"),
          ("Épocas", "60 (com ES)"), ("Seed", "42"),
          ("Modelos", "trained_reval_h160")],
)


def _std_erro_figure():
    data = _load("std_erro_horizonte.json")
    if not data:
        return ""
    return C.figure(
        "figures_static/std_erro_horizonte.png",
        "Desvio padrão do erro (previsto − real, m/s) por modelo e "
        "horizonte, entre origens da grade rolling oficial (trim 72): "
        "verde = erros consistentes entre origens; vermelho = dispersão "
        "alta. Complemento do heatmap de MSE, que mistura viés² e "
        "variância — o MSE pode crescer por viés ou por dispersão, e "
        "este mapa separa os dois efeitos. Gerado por "
        "<code>std_erro_horizonte.py</code> (run 12 + baselines "
        "<code>arima_rf_reval160</code>).",
        label="fig:std_erro_horizonte")


def build(ctx):
    html = _base_build(ctx) + _tests_section() + _friedman_h_section() \
        + _cd_section() + _cd_horizonte_section()
    # insere o heatmap de desvio padrão logo abaixo do heatmap de R²
    # (que fica imediatamente embaixo do de MSE na seção "Análise gráfica")
    std_fig = _std_erro_figure()
    if std_fig:
        marker = "06_heatmap_r2_horizonte.png"
        pos = html.find(marker)
        if pos != -1:
            end = html.find("</figure>", pos)
            if end != -1:
                end += len("</figure>")
                html = html[:end] + std_fig + html[end:]
    return html


if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
