"""Relatório da rodada FINAL do artigo SBSR — LiDAR 80/20.

Lineup canônico unificado (finding `rodada-final-sbsr`): TCN/TCN-Bi com
46 canônicas + humid/temp (48 col), LSTM/LSTM-Bi com featurex (67 col),
MSE pura em todos, HPs canônicos, seed 42, 60 épocas, baselines na mesma
grade (scripts/lidar_8020_sbsr_train.py → pipeline/tmp/trained_8020_sbsr).
Página padronizada a partir de data/results/report_8020_sbsr + seção de
testes de hipótese (DM/Friedman/Wilcoxon-Holm/MCS multi-α) lida de
`testes_hipotese.json` e `mcs_multialpha.json` no diretório de origem.

    python reports/lidar/8020-sbsr/report.py   # reconstrói a página
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src import components as C  # noqa: E402
from reports.src.resultados import report_module  # noqa: E402

SOURCE_DIR = PROJECT_ROOT / "data" / "results" / "report_8020_sbsr"
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
    return C.table(
        ["Par", "ΔMSE", "DM", "p bruto", "p (Holm)", "Melhor",
         "Rejeita H₀ (5%)", "Rejeita (Holm)"],
        rows,
        "Teste de Diebold–Mariano pareado entre os 7 modelos da rodada "
        "(21 pares), perdas por origem (MSE médio nos 36 passos), ajuste "
        "HLN + HAC com lag 35; α = 0,05.",
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
    return C.table(
        ["Par", "p bruto", "p (Holm)", "Melhor", "Rejeita H₀ (Holm, 5%)"],
        rows,
        "Wilcoxon signed-rank pareado por origem entre os 7 modelos "
        "(21 pares) com correção de Holm; α = 0,05.",
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
    body = C.table(
        ["Modelo", "Rank médio"], rows,
        f"Friedman sobre as perdas por origem ({testes.get('n_origens')} "
        f"blocos): χ² = {_num(fr['chi2'], 1)}; p = {_p(fr['p'])} — H₀ "
        "rejeitada.",
        formats=("sbsr", "steam", "generic"), label="tab:friedman_ranks",
        note="H₀: ranks médios iguais entre os 7 modelos (cada origem é "
             "um bloco).",
    )
    return body


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
    n = testes.get("n_origens", 1405)
    fr = testes.get("friedman", {})
    parts = [C.section(
        "Testes de hipótese",
        _dm_table(testes)
        + (_friedman_table(testes) if fr else "")
        + _wilcoxon_table(testes)
        + _mcs_table(multi),
        sub=f"Perdas por origem = MSE médio nos 36 passos ({n} origens "
            "comuns aos 7 modelos; protocolo da rodada 80/20-sbsr). Leitura: "
            "o DM com correção HAC não distingue perdas médias entre "
            "qualquer par (empate técnico geral, consistente com o MCS); o "
            "Friedman global rejeita igualdade de ranks e o Wilcoxon + Holm "
            "estabelece a ordem distribucional — TCN-Bi no topo, LSTM-Bi no "
            "fundo.",
        sec_id="testes-hipotese")]
    return "\n".join(parts)


REPORT, _base_build = report_module(
    name="8020-sbsr",
    title="LiDAR 80/20 — rodada final SBSR",
    description="Lineup do artigo: TCN/TCN-Bi 46+meteo (48 col), "
                "LSTM/LSTM-Bi featurex (67 col), MSE pura em todos, "
                "seed 42, com baselines na mesma grade.",
    source_dir="data/results/report_8020_sbsr",
    subtitle="Rodada canônica do artigo SBSR: split 80/20 sem validação, "
             "60 épocas fixas, seed 42, perda MSE pura em todos os "
             "modelos (ITILDE-Q descartada), alvo ws100_wavelet. TCNs "
             "consumem o perfil LiDAR + meteorologia (48 col); LSTMs "
             "acrescentam cis1..19 (featurex, 67 col). Entrada de 72 "
             "passos (12 h) → horizonte de 36 passos (6 h), 10 min.",
    badges=("80/20", "60 épocas", "seed 42", "MSE pura", "lineup misto",
            "com baselines", "sem origens de fronteira"),
    protocol_note="Modelos treinados em <code>pipeline/tmp/"
                  "trained_8020_sbsr</code> por <code>scripts/"
                  "lidar_8020_sbsr_train.py</code> (feature_keep por "
                  "wrapper). Avaliação com baselines na mesma grade por "
                  "<code>reports/src/relatorio_resultados.py</code>; as "
                  "primeiras 72 origens do teste (janela de entrada "
                  "cruzando a fronteira treino/teste) são descartadas "
                  "para todos os métodos.",
    regenerate=[".venv/bin/python", "reports/src/relatorio_resultados.py",
                "--models-dir", "pipeline/tmp/trained_8020_sbsr",
                "--out-dir", "data/results/report_8020_sbsr",
                "--trim-boundary-origins", "72",
                "--run-title", "LiDAR 80/20 — rodada final SBSR"],
    retrain=[".venv/bin/python", "scripts/lidar_8020_sbsr_train.py"],
    info=[("Protocolo", "80/20 sem validação"), ("Épocas", "60 fixas"),
          ("Seed", "42"), ("Perda", "MSE pura (todos)"),
          ("TCN/TCN-Bi", "46 + humid/temp (48 col)"),
          ("LSTM/LSTM-Bi", "46 + cis1..19 + humid/temp (67 col)"),
          ("Baselines", "ARIMA, RF, persistência (mesma grade)"),
          ("Janela", "72 → 36 passos (10 min)"),
          ("Modelos", "trained_8020_sbsr")],
)


def build(ctx):
    return _base_build(ctx) + _tests_section()


if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
