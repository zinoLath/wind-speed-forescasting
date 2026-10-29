"""Relatório de revalidação SBSR — 72,5/12,5/15 com ES + RLRP.

Mesmo lineup canônico do artigo (finding `rodada-final-sbsr`: TCN/TCN-Bi
48 col com meteo, LSTM/LSTM-Bi featurex 67 col, MSE pura, HPs canônicos,
dilatações C′, seed 42), treinado no regime com validação real: split
72,5/12,5/15, EarlyStopping(val_loss, patience 12, restore_best_weights)
+ ReduceLROnPlateau — o regime que o HPO usa e o protocolo final seco
não tem. Página padronizada a partir de data/results/report_reval_sbsr +
testes de hipótese (DM/Friedman/Wilcoxon-Holm/MCS) lidos de
`testes_hipotese.json` e `mcs_multialpha.json`.

    python reports/lidar/reval_sbsr/report.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src import components as C  # noqa: E402
from reports.src.resultados import report_module  # noqa: E402

SOURCE_DIR = PROJECT_ROOT / "data" / "results" / "report_reval_sbsr"
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
            "comuns aos 4 modelos; protocolo da revalidação 72,5/12,5/15 "
            "com ES+RLRP).",
        sec_id="testes-hipotese")


REPORT, _base_build = report_module(
    name="reval_sbsr",
    title="LiDAR 72,5/12,5/15 — revalidação SBSR (ES+RLRP)",
    description="Lineup canônico do artigo (TCN/TCN-Bi 48 col, "
                "LSTM/LSTM-Bi featurex 67 col, MSE pura, HPs canônicos) "
                "revalidado com validação real, EarlyStopping e "
                "ReduceLROnPlateau.",
    source_dir="data/results/report_reval_sbsr",
    subtitle="Mesmos modelos do artigo SBSR, agora com split 72,5/12,5/15 "
             "COM validação real e EarlyStopping(val_loss, patience 12, "
             "restore_best_weights) + ReduceLROnPlateau (fator 0,5, "
             "patience 6) — o regime que o HPO usa e o protocolo final "
             "seco (80/20, 60 épocas sem callbacks) não tem. ATENÇÃO: a "
             "fatia de teste aqui é a última 15% (vs 20% do artigo) — os "
             "números NÃO são diretamente comparáveis ao 8020-sbsr; a "
             "comparação de regime válida exige mesma grade.",
    badges=("72,5/12,5/15", "60 épocas", "seed 42", "MSE pura",
            "EarlyStopping", "ReduceLROnPlateau", "lineup misto"),
    protocol_note="Modelos treinados em <code>pipeline/tmp/"
                  "trained_reval_sbsr</code> por <code>scripts/"
                  "lidar_reval_sbsr_train.py</code> (feature_keep por "
                  "wrapper, HPs da canônica <code>trained_8020_itildeq</"
                  "code>). As primeiras 72 origens do teste são "
                  "descartadas (janela cruzando a fronteira), como na "
                  "rodada do artigo.",
    regenerate=[".venv/bin/python", "reports/src/relatorio_resultados.py",
                "--models-dir", "pipeline/tmp/trained_reval_sbsr",
                "--out-dir", "data/results/report_reval_sbsr",
                "--trim-boundary-origins", "72",
                "--run-title",
                "LiDAR 72,5/12,5/15 — revalidação SBSR (ES+RLRP)"],
    retrain=[".venv/bin/python", "scripts/lidar_reval_sbsr_train.py"],
    info=[("Protocolo", "72,5/12,5/15 com ES+RLRP"),
          ("EarlyStopping", "val_loss, patience 12, restore_best"),
          ("ReduceLROnPlateau", "fator 0,5, patience 6, min_lr 1e-6"),
          ("HPs", "canônicos (trained_8020_itildeq)"),
          ("Loss", "MSE pura (todos)"),
          ("TCN/TCN-Bi", "46 + humid/temp (48 col)"),
          ("LSTM/LSTM-Bi", "46 + cis1..19 + humid/temp (67 col)"),
          ("Épocas", "60 (com ES)"), ("Seed", "42"),
          ("Modelos", "trained_reval_sbsr")],
)


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
                     ("LSTM", "BiLSTM", "TCN", "BiTCN", "ARIMA",
                      "Pers.")}
        best = min(rank_keys, key=rank_keys.get)
        table_rows.append((
            str(h), _num(r["chi2"], 1), _p(r["p_holm"]), best,
            _num(rank_keys["LSTM"], 2),
            "sim" if r["p_holm"] < 0.05 else "não"))
    nsig = sum(1 for r in rows if r["p_holm"] < 0.05)
    body = C.figure(
        "figures_static/friedman_horizonte.png",
        "Painel superior: MSE médio por horizonte (escala log). Painel "
        "inferior: −log₁₀(p) do Friedman pareado por horizonte, bruto e "
        "com Holm; abaixo da linha tracejada H₀ não é rejeitada.",
        label="fig:friedman_horizonte")
    body += C.table(
        ["Horizonte", "χ²", "p (Holm)", "Melhor rank", "Rank LSTM",
         "H₀ rejeitada"],
        table_rows,
        f"Friedman pareado por horizonte ({data.get('n_origens')} origens "
        f"na grade comum; blocos = origens; perda = MSE no par "
        f"(origem, h)): H₀ rejeitada em {nsig}/36 horizontes (Holm 5%). "
        f"Inclui ARIMA e persistência (cache na mesma grade); RF fora "
        f"(sem predições por origem armazenadas).",
        formats=("sbsr", "steam", "generic"), label="tab:friedman_horizonte",
        note="Leitura: curtíssimo prazo favorece TCN-Bi/ARIMA (persistência "
             "ainda competitiva em h=1); do meio em diante (h ≥ 18) a LSTM "
             "canônica tem o melhor rank médio — é ali que ela constrói a "
             "vantagem agregada.",
    )
    return C.section(
        "Friedman pareado por horizonte", body,
        sub="Gerado por <code>friedman_horizonte.py</code> a partir das "
            "predições do banco (run reval-sbsr) + cache ARIMA.",
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
        note="Grupos não-separáveis: " + "; ".join(grupos) + ". TCN-Bi e "
             "TCN separam-se por margem mínima (≈ CD); o cluster de topo "
             "({TCN-Bi, LSTM, TCN}) domina o cluster de baselines "
             "({Persistência, ARIMA, LSTM-Bi}).",
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
        "Diagrama de Diferença Crítica (Wilcoxon-Holm) nos horizontes de "
        "1 h a 6 h (h = 6, 12, 18, 24, 30, 36): pontos no eixo de ranks "
        "médios (1 = melhor, à esquerda), barras ligam métodos cuja "
        "diferença de perdas por origem não é significativa (Wilcoxon "
        "signed-rank + Holm dentro do horizonte, α = 0,05).",
        label="fig:cd_horizonte_selecionados")
    body += C.figure(
        "figures_static/cd_horizonte.png",
        "Grade completa do mesmo diagrama para todos os horizontes "
        "(h = 1..36).",
        label="fig:cd_horizonte")
    return C.section(
        "Diagrama de diferença crítica por horizonte", body,
        sub=f"Estilo do diagrama de referência (Wilcoxon-Holm): a LSTM "
            f"canônica tem o melhor rank em {lstm_best}/36 horizontes "
            f"(a partir de h ≈ 10, persistindo até h = 36); no curtíssimo "
            f"prazo (h ≤ 9) a TCN-Bi lidera com ARIMA próximo. A "
            f"persistência afunda para a última posição de h ≥ 23 em "
            f"diante. Gerado por <code>cd_horizonte.py</code>.",
        sec_id="cd-horizonte")


def _h36_section():
    data = _load("tabela_h36.json")
    if not data:
        return ""
    rows = []
    for r in data["rows"]:
        rows.append((r["modelo"], _num(r["r2"]), _num(r["mse"]),
                     _num(r["rmse"]), _num(r["mae"])))
    return C.section(
        "Métricas no h = 36 (6 h), com baselines",
        C.table(
            ["Modelo", "R²", "MSE", "RMSE", "MAE"],
            rows,
            f"Métricas exclusivamente no horizonte de 36 passos (6 h), "
            f"mesma grade de {data.get('n_origens')} origens comuns do "
            f"banco de predições (DL + ARIMA/RF/persistência).",
            formats=("sbsr", "steam", "generic"), label="tab:h36_baselines",
            note="No h36 TCN e LSTM empatam tecnicamente (ΔMSE ≈ 0,4%); "
                 "a vantagem agregada da LSTM vem dos horizontes mais "
                 "curtos.",
        ),
        sec_id="h36-baselines")


def build(ctx):
    html = (_base_build(ctx) + _tests_section() + _friedman_h_section()
            + _cd_section() + _cd_horizonte_section())
    h36 = _h36_section()
    if h36:
        # encaixa a tabela h36 logo após a seção de métricas globais
        parts = html.split("</section>")
        idx = next((i for i, p in enumerate(parts)
                    if 'id="sec-metricas"' in p), None)
        if idx is not None:
            parts.insert(idx + 1, h36)
            html = "</section>".join(parts)
    return html


if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
