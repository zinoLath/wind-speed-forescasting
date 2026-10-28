"""Relatório — erro normalizado + Diebold–Mariano ajustado (8020-itildeq).

Página metodológica da rodada mista 80/20-ITILDE-Q. Duas ideias centrais:

1. **Erro normalizado pelo intrínseco do dado** — MASE, skill vs.
   persistência e razão local |erro|/|erro da persistência|, no geral e por
   estrato (regime diurno/noturno, faixa de intensidade), mais o erro
   relativo ao piso de suavização wavelet. Serve para separar "o dado é
   difícil aqui" de "o modelo é ruim aqui": os modelos erram nas MESMAS
   áreas em que a persistência também erra.
2. **Diebold–Mariano com ajustes** — DM(HLN+HAC), DM(NW automático),
   p-valores de Holm e Clark–West, com a leitura de cada variante.

Fontes dos números:
    data/results/metodos_itildeq/metodos_itildeq.json + images/*.png
Gerador: studies/metodos_itildeq.py

    .venv/bin/python reports/lidar/8020-itildeq-metodos/report.py
"""

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src import components as C  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "results" / "metodos_itildeq"
META_JSON = DATA_DIR / "metodos_itildeq.json"
IMG_DIR = DATA_DIR / "images"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIGS = ("fig_err_norm.png", "fig_skill_horizon.png", "fig_dm_variants.png",
        "fig_strata_normalized.png")


def fmt(x, nd=3):
    if x is None:
        return "—"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f != f:
        return "—"
    return f"{f:.{nd}f}".replace(".", ",")


def copy_figures():
    if IMG_DIR.is_dir():
        FIG_DIR.mkdir(exist_ok=True)
        for name in FIGS:
            src = IMG_DIR / name
            if src.is_file():
                shutil.copy2(src, FIG_DIR / name)


def build(ctx):  # noqa: ARG001 (contrato do builder)
    copy_figures()
    if not META_JSON.is_file():
        return C.page_header(
            "Erro normalizado + DM ajustado",
            "Análise ainda não executada.") + C.note(
            "Rode <code>studies/metodos_itildeq.py --models-dir "
            "pipeline/tmp/trained_8020_itildeq</code> (ou o botão "
            "<strong>Regenerar</strong> com o servidor ativo).")

    d = json.loads(META_JSON.read_text(encoding="utf-8"))
    n = d.get("n_origins", 0)
    alpha = d.get("alpha", 0.05)

    parts = [C.page_header(
        "Erro normalizado + Diebold–Mariano ajustado",
        "Rodada mista 80/20-ITILDE-Q: como o erro do modelo se compara ao "
        "erro intrínseco do dado (persistência) e como quatro variantes do "
        "teste de Diebold–Mariano respondem à mesma pergunta de precisão.",
        badges=("erro normalizado", "MASE", "DM(HLN+HAC)", "DM(NW auto)",
                "Holm", "Clark–West", f"{n} origens"))]

    parts.append(C.cards([
        (f"{n * 36:,}".replace(",", "."), "amostras (origens × 36 h)"),
        (f"{n}", "origens comuns (DL + baselines)"),
        (fmt(d.get("floor_mae"), 3) + " m/s", "piso de suavização (MAE)"),
        (fmt(d.get("floor_rmse"), 3) + " m/s", "piso de suavização (RMSE)"),
    ]))

    # ------------------------------------------------------------------
    # 1) Erro normalizado pelo intrínseco
    # ------------------------------------------------------------------
    norm_rows = [(r["modelo"], fmt(r["MAE"]), fmt(r["MASE"]),
                  fmt(r["skill"]), fmt(r["ratio_medio"]),
                  fmt(r["ratio_mediano"]))
                 for r in d.get("norm", [])]
    parts.append(C.section(
        "Erro normalizado pelo intrínseco do dado",
        "<p><b>A ideia.</b> Os modelos erram sobretudo <i>onde o dado é "
        "difícil</i> — rampas e extremos — e nesses mesmos instantes a "
        "persistência também erra muito. Comparar MSE absoluto entre "
        "métodos mistura “o dado é difícil aqui” com “o modelo é ruim "
        "aqui”. Normalizando pelo erro da persistência <b>na mesma célula/"
        "estrato</b>, o piso de dificuldade do dado sai da conta e sobra o "
        "ganho genuíno do modelo.</p>"
        "<ul>"
        "<li><b>MASE</b> = MAE<sub>modelo</sub>/MAE<sub>persist.</sub> "
        "(&lt;1 bate o ingênuo; Hyndman &amp; Koehler 2006);</li>"
        "<li><b>Skill</b> = 1 − MSE<sub>modelo</sub>/MSE<sub>persist.</sub> "
        "(0 = empata; &lt;0 = pior que repetir o último valor);</li>"
        "<li><b>Razão local</b> = |erro<sub>modelo</sub>| / "
        "|erro<sub>persist.</sub>| por célula (mediana robusta a caudas);</li>"
        "<li><b>Erro/piso</b> = MAE<sub>modelo</sub>/piso wavelet — fração "
        "do erro irredutível do alvo (bruto vs suavizado) que o modelo "
        f"carrega. Piso = {fmt(d.get('floor_mae'), 3)} m/s (MAE).</li>"
        "</ul>"
        + C.figure("figures/fig_err_norm.png",
                   "MASE, skill vs. persistência e razão mediana |erro|/"
                   "|erro persist.| por método (linha tracejada = empate "
                   "com a persistência).", label="fig:norm_bars")
        + C.table(["Modelo", "MAE (m/s)", "MASE", "Skill",
                   "Razão (média)", "Razão (mediana)"], norm_rows,
                  "Normalização do erro pelo erro intrínseco (persistência) "
                  "na mesma grade de origens.", label="tab:norm_geral",
                  note="MASE<1 e skill>0 indicam ganho real sobre o ingênuo; "
                       "razão mediana <1 significa vencer a persistência em "
                       "mais da metade das células.")
        + C.figure("figures/fig_skill_horizon.png",
                   "Skill vs. persistência por horizonte (IC 95% stationary "
                   "bootstrap): o ganho cresce com o horizonte — a "
                   "persistência domina nos primeiros passos.",
                   label="fig:norm_skill", width=1.0),
        sub="Tabela e figuras calculadas sobre as origens comuns (DL + "
            "baselines) da rodada mista.", sec_id="norm"))

    strata = d.get("strata", {})
    reg_rows = [(r["modelo"], r["regime"], fmt(r["ratio_MAE"]),
                 fmt(r["skill"]))
                for r in strata.get("regime", [])]
    band_rows = [(r["modelo"], str(r["banda"]), int(r["n"]),
                  fmt(r["ratio_MAE"]), fmt(r["skill"]))
                 for r in strata.get("banda", [])]
    parts.append(C.section(
        "Erro normalizado por estrato",
        "<p>Onde os modelos erram em conjunto, a razão MAE"
        "<sub>modelo</sub>/MAE<sub>persist.</sub> fica perto de 1: o dado "
        "é que é difícil, não o modelo. Onde a razão cai abaixo de 1, o "
        "modelo agrega informação que a persistência não tem. O MBE por "
        "faixa, no relatório de métodos complementares, mostra que os "
        "extremos (≥13 m/s) são subestimados por todos — o erro intrínseco "
        "que nenhum modelo desta rodada resolve.</p>"
        + C.figure("figures/fig_strata_normalized.png",
                   "MAE do modelo dividido pelo MAE da persistência, por "
                   "regime diurno/noturno e por faixa de intensidade "
                   "(linha tracejada = empate).",
                   label="fig:norm_strata", width=1.0)
        + C.table(["Modelo", "Regime", "MAE/MAE_persist.", "Skill"],
                  reg_rows, "Erro normalizado por regime diurno × noturno.",
                  label="tab:norm_regime")
        + C.table(["Modelo", "Faixa", "n", "MAE/MAE_persist.", "Skill"],
                  band_rows, "Erro normalizado por faixa de intensidade.",
                  label="tab:norm_banda",
                  note="Faixas operacionais 0–3 / 3–13 / ≥13 m/s "
                       "(xianfeng2026data); quartis de volatilidade como "
                       "alternativa (aslam2026physics)."),
        sub="Estratos usando o regime diurno (queda semidiurna 06h/18h já "
            "descrita no EDA) e a faixa de intensidade do vento real.",
        sec_id="norm_strata"))

    # ------------------------------------------------------------------
    # 2) DM ajustado
    # ------------------------------------------------------------------
    dm_rows = []
    for r in d.get("dm", []):
        dm_rows.append((
            f"{r['a']} vs {r['b']}", fmt(r["dMSE"]),
            fmt(r["dm_hln"], 2), fmt(r["p_hln"], 4),
            fmt(r["p_holm"], 4),
            fmt(r["dm_nw"], 2), fmt(r["p_nw"], 4),
            "sim" if r.get("sig_holm") else "não",
            r["melhor"],
        ))
    cw_rows = [(r["modelo"], fmt(r["dMSE"]), fmt(r["cw"], 2),
                fmt(r["p"], 4), r["favorece"])
               for r in d.get("clark_west", [])]
    parts.append(C.section(
        "Diebold–Mariano com ajustes",
        "<p>O DM compara duas previsões pela série de diferenças de perda "
        "d<sub>t</sub> = L(A) − L(B), H₀: E[d<sub>t</sub>] = 0. Como as "
        "janelas de 6 h se sobrepõem, a variância precisa de correção HAC e "
        "— em amostra finita — a estatística pede ajuste. As quatro "
        "variantes respondem à mesma pergunta por caminhos diferentes:</p>"
        "<ul>"
        "<li><b>DM(HLN+HAC)</b> — <i>o padrão do artigo</i>. Variância HAC "
        "com defasagem h−1 (Newey–West) + correção de pequena amostra de "
        "Harvey–Leybourne–Newbold (1997), avaliada contra t(n−1). É o que o "
        "relatório de resultados já usa.</li>"
        "<li><b>DM(NW auto)</b> — mesma diferença de perda, mas a "
        "defasagem do HAC vem da regra automática de Newey–West "
        "(floor(4·(n/100)^(2/9))) e a estatística é comparada à normal "
        "N(0,1). <i>Checagem de robustez:</i> veredito que muda entre as "
        "duas aponta sensibilidade à escolha da banda.</li>"
        "<li><b>Holm</b> — os p-valores do DM(HLN+HAC) ajustados por "
        "multiplicidade (Holm 1979). Com 21 pares, ~1 falso positivo é "
        "esperado a α=5%; o ajuste controla a taxa de erro de família. "
        "Aqui o critério honesto é: significativo depois de Holm.</li>"
        "<li><b>Clark–West</b> — MSPE-ajustado (2007) para o benchmark "
        "<i>aninhado</i> (a persistência). Ajusta o viés do DM quando o "
        "modelo maior “contém” o menor; CW&gt;0 favorece o modelo.</li>"
        "</ul>"
        + C.figure("figures/fig_dm_variants.png",
                   "Esquerda: estatística DM(HLN+HAC) × DM(NW auto) por "
                   "par. Direita: significância bruta × ajustada por Holm "
                   "(linha vermelha em α).",
                   label="fig:dm_variants", width=1.0)
        + C.table(["Par", "ΔMSE", "DM HLN", "p HLN", "p Holm",
                   "DM NW", "p NW", "Holm sig.?", "Melhor"], dm_rows,
                  "Diebold–Mariano agregado por origem (perda quadrática) "
                  "nas quatro variantes de ajuste.", label="tab:dm_ajustado",
                  note="Fontes: Diebold & Mariano (1995), "
                       "doi:10.1080/07350015.1995.10524599; Harvey, "
                       "Leybourne & Newbold (1997), "
                       "doi:10.1016/S0169-2070(96)00719-4; Newey & West "
                       "(1994); Holm (1979); Clark & West (2007).")
        + C.table(["Modelo", "ΔMSE", "CW", "p", "Favorece"], cw_rows,
                  "Clark–West MSPE-ajustado — cada rede contra a "
                  "persistência (benchmark aninhado).", label="tab:clark_west",
                  note="CW>0 favorece o modelo; CW<0 favorece a "
                       "persistência."),
        sub="Convenção: ΔMSE e DM positivos ⇒ o primeiro modelo do par tem "
            "perda maior (pior). O veredito “Melhor” usa o p de Holm.",
        sec_id="dm_ajustado"))

    parts.append(C.section(
        "Como ler (síntese)",
        "<div class='markdown'><ul>"
        "<li><b>Erro normalizado</b> é a métrica que responde à pergunta "
        "“quanto do erro é do dado e quanto é do modelo”: MASE, skill e "
        "razão local descontam o erro que a persistência também comete.</li>"
        "<li>Use <b>Holm</b> para afirmações de superioridade entre pares "
        "(multiplicidade); use <b>Clark–West</b> na comparação com a "
        "persistência; use <b>NW auto</b> como checagem de robustez do DM "
        "canônico.</li>"
        "<li>Divergência entre variantes não invalida o resultado — "
        "<i>quantifica a incerteza do veredito</i> e deve ser relatada "
        "(prática de binrong2026whole e wang2026adaptive).</li>"
        "</ul></div>",
        sec_id="leitura"))

    return "".join(parts)


REPORT = {
    "title": "Erro normalizado + DM ajustado (8020-itildeq)",
    "description": "MASE/skill/razão pelo erro intrínseco (persistência), "
                   "por estrato, e Diebold–Mariano ajustado (HLN+HAC, NW "
                   "auto, Holm, Clark–West).",
    "regenerate": [".venv/bin/python", "studies/metodos_itildeq.py",
                   "--models-dir", "pipeline/tmp/trained_8020_itildeq",
                   "--out-dir", "data/results/metodos_itildeq"],
    "retrain": None,
    "source_dir": "data/results/metodos_itildeq",
    "info": [
        ("Rodada", "trained_8020_itildeq (rodada mista)"),
        ("Normalização", "MASE · skill · razão |erro|/|erro persist.ª|"),
        ("DM ajustado", "HLN+HAC · NW auto · Holm · Clark–West"),
        ("Fontes", "docs/revisions/sbr/guia_metricas.ai.md"),
    ],
}


if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
