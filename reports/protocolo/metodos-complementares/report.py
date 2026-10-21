"""Relatório de protocolo — análises metodológicas complementares (SBSR).

Consolida as análises pós-modelo geradas por
``tests/metodos_complementares.py`` (Diebold–Mariano com HAC+HLN e IC por
bootstrap estacionário, skill vs. persistência, estratos por
regime/intensidade, métricas de cauda, custo computacional, KS/ECDF,
métricas de fase, A/B de loss e diagnóstico de resíduos com Page–Hinkley),
com textos interpretativos e inseríveis com popup LaTeX.

Fontes:
    data/results/metodos_complementares/metodos.json
    data/results/metodos_complementares/images/*.png
    data/results/metodos_complementares/skill_por_horizonte.csv

Executar a partir da raiz:
    .venv/bin/python reports/protocolo/metodos-complementares/report.py
"""

import csv
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src import components as C  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "results" / "metodos_complementares"
META_JSON = DATA_DIR / "metodos.json"
SKILL_CSV = DATA_DIR / "skill_por_horizonte.csv"
IMG_DIR = DATA_DIR / "images"
FIG_DIR = Path(__file__).resolve().parent / "figures"

SHORTS = {
    "Seq2Seq_LSTM": "LSTM", "Seq2Seq_LSTM_Bidirectional": "LSTM-Bi",
    "Seq2Seq_TCN": "TCN", "Seq2Seq_TCN_Bidirectional": "TCN-Bi",
    "__persistence__": "Persistência", "__arima__": "ARIMA",
    "__rf__": "Random Forest",
}
FIGS = ("fig_dm_concept.png", "skill_vs_persistencia.png",
        "fig_strata_regime.png", "fig_strata_band.png", "fig_tails.png",
        "fig_cost.png", "fig_ecdf_annot.png", "fig_phase_traj.png",
        "fig_loss_forest.png", "fig_resid.png", "fig_breaks.png")


def fmt(x, nd=3):
    """float -> string pt-BR (vírgula); None/NaN -> '—'."""
    if x is None:
        return "—"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f != f:  # NaN
        return "—"
    return f"{f:.{nd}f}".replace(".", ",")


def short(name):
    return SHORTS.get(name, name)


def copy_figures():
    """Traz as figuras da análise para figures/ (o site copia daqui)."""
    if IMG_DIR.is_dir():
        FIG_DIR.mkdir(exist_ok=True)
        for name in FIGS:
            src = IMG_DIR / name
            if src.is_file():
                shutil.copy2(src, FIG_DIR / name)


def load_skill_agg():
    """Agregados do skill por modelo (médio, h6, h36) a partir do CSV."""
    if not SKILL_CSV.is_file():
        return []
    per = {}
    with open(SKILL_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            g = per.setdefault(row["modelo"], {})
            h = int(row["horizon"])
            g[h] = float(row["skill"])
    rows = []
    for name, g in per.items():
        vals = list(g.values())
        if not vals:
            continue
        rows.append((short(name), fmt(sum(vals) / len(vals)),
                     fmt(g.get(6)), fmt(g.get(36))))
    rows.sort(key=lambda r: float(r[3].replace(",", ".")), reverse=True)
    return rows


def build(ctx):  # noqa: ARG001 (contrato do builder)
    copy_figures()
    if not META_JSON.is_file():
        return C.page_header(
            "Análises metodológicas complementares",
            "Rodada ainda não executada.") + C.note(
            "Execute <code>tests/metodos_complementares.py</code> (ou use o "
            "botão <strong>Regenerar</strong> com o servidor ativo) para "
            "gerar <code>metodos.json</code>.")

    d = json.loads(META_JSON.read_text(encoding="utf-8"))
    n = d.get("n_origins", 0)

    parts = [C.page_header(
        "Análises metodológicas complementares",
        "Testes estatísticos, estratificações e diagnósticos pós-modelo "
        "sobre a rodada canônica 80/20 — DL e baselines (ARIMA, Random "
        "Forest, Persistência) avaliados na mesma grade de origens do "
        "rolling forecast.",
        badges=("DM + HLN", "skill score", "bootstrap estacionário",
                f"{n} origens")),
        C.cards([
            (f"{n * 36:,}".replace(",", "."), "amostras avaliadas"),
            (f"{n}", "origens comuns (DL + baselines)"),
            ("7", "métodos comparados"),
            ("21", "pares no teste DM"),
        ])]

    # ---------------- resumo executivo ----------------
    parts.append(C.section(
        "Resumo executivo",
        "<div class='markdown'><ul>"
        "<li><strong>TCN &gt; LSTM e LSTM-Bi é significativo</strong> "
        "(DM: p=0,012 e p=0,0002).</li>"
        "<li><strong>A bidirecionalidade piora a TCN</strong> (p=0,024) e "
        "não ajuda a LSTM (p=0,30) — contraria a narrativa de ganho da "
        "bidirecionalidade.</li>"
        "<li><strong>TCN × ARIMA é empate estatístico</strong> no agregado "
        "(p=0,85); a vantagem aparece no horizonte de 6 h (skill TCN "
        "+0,27 vs +0,18 do ARIMA).</li>"
        "<li><strong>LSTM-Bi é pior que a persistência</strong> (p=0,026) — "
        "skill negativo reportado com honestidade.</li>"
        "<li><strong>Extremos (≥13 m/s) subestimados em ~4 m/s</strong> por "
        "todos os modelos (MBE ≈ −4).</li>"
        "<li><strong>Ablação de perda</strong>: só a TCN-Bi difere — a "
        "favor do MSE puro (p=0,028).</li></ul>"
        "<p><strong>Achado operacional:</strong> o "
        "<code>mcs.json</code> do relatório LiDAR usa apenas <strong>163 "
        "origens</strong> — as origens dos baselines têm drift de "
        "microssegundos e não casam com a grade do dataset. A correção "
        "(<code>origin.dt.round(\"min\")</code>) está aplicada aqui e "
        "falta aplicar em <code>tests/relatorio_resultados.py</code>.</p>"
        "</div>",
        sub="Números desta página vêm de metodos.json; cada seção traz o "
            "conceito, como ler o gráfico e as fontes (DOIs verificados "
            "em 14/10/2026 — ver docs/revisions/sbr/guia_metricas.ai.md).",
        sec_id="resumo"))

    # ---------------- A1 DM ----------------
    dm_rows = []
    for r in d.get("dm", [])[:12]:
        dm_rows.append((f"{short(r['a'])} vs {short(r['b'])}",
                        fmt(r["mean_dMSE"]), fmt(r["IC95_inf"]),
                        fmt(r["IC95_sup"]), fmt(r["DM"], 2), fmt(r["p"], 4),
                        r["melhor"], r["sig"]))
    parts.append(C.section(
        "Diebold–Mariano (HAC + correção HLN)",
        C.figure("figures/fig_dm_concept.png",
                 "Série d_t por origem (perda A − perda B) para um par "
                 "significativo (acima) e um empate (abaixo), com média e "
                 "IC 95% do bootstrap estacionário.",
                 label="fig:metodos_dm_concept")
        + C.table(
            ["Par", "ΔMSE", "IC95 inf", "IC95 sup", "DM", "p", "Melhor",
             "Sig."],
            dm_rows,
            "Teste DM agregado por origem (HAC até h−1 + correção de "
            "Harvey–Leybourne–Newbold; p bilateral, α=0,05). Convenção: "
            "ΔMSE/DM positivos ⇒ modelo A pior.",
            label="tab:metodos_dm",
            note="Fontes: Diebold & Mariano (1995), "
                 "doi:10.1080/07350015.1995.10524599; Harvey, Leybourne "
                 "& Newbold (1997), doi:10.1016/S0169-2070(96)00719-4."),
        sub="Para cada origem do rolling forecast calcula-se a diferença "
            "de perda d_t entre dois modelos; H₀: E[d_t]=0 (mesma "
            "precisão). A variância usa estimador HAC (Newey–West) porque "
            "as janelas se sobrepõem, e a estatística recebe a correção de "
            "pequenas amostras de HLN, avaliada contra t(n−1). IC 95% via "
            "bootstrap estacionário (Politis–Romano 1994, "
            "doi:10.1080/01621459.1994.10476870).",
        sec_id="dm"))

    # ---------------- A2 Skill ----------------
    skill_rows = load_skill_agg()
    parts.append(C.section(
        "Skill score vs. persistência",
        C.figure("figures/skill_vs_persistencia.png",
                 "Skill por horizonte (1 − MSE_modelo/MSE_persistência); "
                 "sombra = IC 95% do bootstrap estacionário; linha "
                 "tracejada = persistência (skill 0).",
                 label="fig:metodos_skill")
        + C.table(["Modelo", "Skill médio", "Skill h6 (1 h)",
                   "Skill h36 (6 h)"],
                  skill_rows,
                  "Skill score contra a persistência: 0 = empatar com a "
                  "predição ingênua; negativo = pior que a ingênua.",
                  label="tab:metodos_skill",
                  note="Referência: Hyndman & Athanasopoulos, FPP3 "
                       "§5.8–5.9 (otexts.com/fpp3/accuracy.html)."),
        sub="No agregado quase tudo fica abaixo de 0 — a persistência é "
            "forte em horizonte curto. A história do artigo está no "
            "horizonte de 6 h, onde TCN/TCN-Bi/ARIMA cruzam para cima. "
            "Skill negativo reportado por honestidade (prática de "
            "bendes2026bio).",
        sec_id="skill"))

    # ---------------- A4 Estratos ----------------
    reg_rows = [(r["modelo"], r["regime"], f"{r['n']:,}".replace(",", "."),
                 fmt(r["MAE"]), fmt(r["RMSE"]), fmt(r["MBE"]))
                for r in d.get("strata", {}).get("regime", [])]
    band_rows = [(r["modelo"], str(r["banda"]), fmt(r["n"]),
                  fmt(r["MAE"]), fmt(r["RMSE"]), fmt(r["MBE"]))
                 for r in d.get("strata", {}).get("banda", [])]
    parts.append(C.section(
        "Erro por regime e por intensidade",
        C.figure("figures/fig_strata_regime.png",
                 "MAE e MBE por regime diurno (06–17h) × noturno "
                 "(18–05h): a troca de sinal do MBE revela viés "
                 "condicional ao regime.",
                 label="fig:metodos_strata_regime", width=1.0)
        + C.figure("figures/fig_strata_band.png",
                   "MBE por faixa de intensidade do vento real; a banda "
                   "0–3 m/s está vazia (sítio sem calmaria) e a ≥13 m/s "
                   "(n=36) mostra subestimação de ~4 m/s.",
                   label="fig:metodos_strata_band")
        + C.table(["Modelo", "Regime", "n", "MAE", "RMSE", "MBE"],
                  reg_rows,
                  "Erro estratificado por regime diurno/noturno.",
                  label="tab:metodos_strata_regime")
        + C.table(["Modelo", "Faixa", "n", "MAE", "RMSE", "MBE"],
                  band_rows,
                  "Erro estratificado por faixa de intensidade "
                  "(faixas operacionais).",
                  label="tab:metodos_strata_banda",
                  note="Prática: xianfeng2026data (faixas 0–3/3–13/13–25 "
                       "+ MBE); aslam2026physics (quartis de "
                       "volatilidade); wei2026secondary (eventos fortes)."),
        sub="MAE/RMSE globais escondem ONDE o modelo falha. MBE = "
            "média(prev − real): positivo = superestima. Extremos "
            "subestimados ~4 m/s é o erro clássico em séries turbulentas "
            "— e o mais caro para energia.",
        sec_id="estratos"))

    # ---------------- A5 Caudas ----------------
    tail_rows = [(r["modelo"], fmt(r["MedAE"]), fmt(r["P95_global"]),
                  fmt(r["MaxAE"]))
                 for r in d.get("tails_medians", [])]
    parts.append(C.section(
        "Métricas de cauda",
        C.figure("figures/fig_tails.png",
                 "Histograma do |erro| (TCN) com MedAE/P95/MaxAE marcados; "
                 "P95 do |erro| por horizonte para todos os métodos.",
                 label="fig:metodos_tails", width=1.0)
        + C.table(["Modelo", "MedAE", "P95", "MaxAE"], tail_rows,
                  "Percentis do erro absoluto: mediana (robusta), pior "
                  "caso típico (P95) e pior caso absoluto (MaxAE).",
                  label="tab:metodos_tails",
                  note="Prática: xiang2026seqbinder (P95/MaxAE por "
                       "horizonte); ehsan2000wind (MedAE)."),
        sub="MAE/MSE falam da média; despacho e segurança de grade "
            "precisam do pior caso. Dois modelos com mesmo MAE podem ter "
            "P95 muito diferentes.",
        sec_id="caudas"))

    # ---------------- A6 Custo ----------------
    cost_rows = []
    for r in d.get("cost", []):
        cost_rows.append((
            r["Modelo"],
            "—" if r["Parametros"] in ("-", None) else f"{r['Parametros']:,}".replace(",", "."),
            fmt(r["Tamanho (MB)"], 2), str(r["Epocas"]), str(r["Batch"]),
            fmt(r["Tempo treino (s)"], 1),
            fmt(r["Inferencia (ms/janela)"], 2), str(r["Loss"]),
            str(r["Seed"])))
    parts.append(C.section(
        "Custo computacional",
        C.figure("figures/fig_cost.png",
                 "Trade-off tempo de treino × MSE (origens comuns); área "
                 "∝ nº de parâmetros; linhas pontilhadas = baselines.",
                 label="fig:metodos_cost")
        + C.table(["Modelo", "Params", "MB", "Épocas", "Batch",
                   "Treino (s)", "Infer. (ms)", "Loss", "Seed"],
                  cost_rows,
                  "Custo computacional por método no mesmo hardware "
                  "(pedido da revisão SBSR).",
                  label="tab:metodos_cost",
                  note="Prática: liu2026robust (custo × ciclo de "
                       "despacho); jiang2026data (TT/PT)."),
        sub="Pontos à esquerda-embaixo = baratos e bons. A TCN-Bi é a "
            "mais cara (1,6M params, 443 s) e não compensa em acurácia.",
        sec_id="custo"))

    # ---------------- B3 KS ----------------
    ks_rows = [(r["par"], fmt(r["KS_D"]), fmt(r["p"], 4), r["mesma_dist"])
               for r in d.get("kspa", [])[:8]]
    parts.append(C.section(
        "KS sobre distribuições de erro",
        C.figure("figures/fig_ecdf_annot.png",
                 "ECDFs do |erro| com a distância máxima (estatística D) "
                 "anotada entre TCN e LSTM.",
                 label="fig:metodos_ecdf")
        + C.table(["Par", "KS D", "p", "Mesma dist.?"], ks_rows,
                  "Teste KS duas amostras sobre os erros absolutos — "
                  "compara distribuições inteiras, não só médias.",
                  label="tab:metodos_ks",
                  note="Implementação: scipy.stats.ks_2samp. Uso: "
                       "gomez2026predicting (ECDF + KSPA)."),
        sub="Com n=43 mil amostras todo p<0,05 é trivial — reporte o "
            "tamanho de efeito D e o gráfico, não o p isolado.",
        sec_id="ks"))

    # ---------------- B4 Fase ----------------
    ph_rows = [(r["modelo"], fmt(r["L_max (passos)"], 2),
                fmt(r["D_phase_mediano"], 1), fmt(r["D_phase_medio"], 1),
                fmt(r["DA"]), str(r["n_origins"]))
               for r in d.get("phase", [])]
    parts.append(C.section(
        "Métricas de fase",
        C.figure("figures/fig_phase_traj.png",
                 "Trajetórias reais (típica e difícil): observado × "
                 "previsto pela TCN, com o timing do pico marcado — "
                 "acertar o valor ≠ acertar o momento.",
                 label="fig:metodos_phase", width=1.0)
        + C.table(["Modelo", "L_max (passos)", "D_phase mediano",
                   "D_phase médio", "DA", "Origens"],
                  ph_rows,
                  "Defasagem (L_max), erro de timing do pico (D_phase) e "
                  "acurácia direcional (DA) por modelo.",
                  label="tab:metodos_phase",
                  note="Prática: liu2026robust (L_max/D_phase, "
                       "l.775–789); huang2024research (directional "
                       "accuracy)."),
        sub="Série com ciclo semidiurno premia quem acerta o relógio: um "
            "pico previsto 1,5 h depois do real tem MAE razoável e valor "
            "operacional baixo. DA ≈ 0,5 em todos: na escala de 10 min a "
            "informação está no padrão da janela, não no incremento "
            "passo a passo.",
        sec_id="fase"))

    # ---------------- B6 Loss A/B ----------------
    if d.get("loss_ab"):
        ab_rows = [(r["modelo"], str(r["n_origins"]),
                    fmt(r["dMSE (canonica - mse)"]), fmt(r["DM"], 2),
                    fmt(r["p"], 4), r["melhor"])
                   for r in d["loss_ab"]]
        parts.append(C.section(
            "A/B de função de perda",
            C.figure("figures/fig_loss_forest.png",
                     "ΔMSE (canônica − MSE puro) por modelo; vermelho = "
                     "diferença significativa (DM).",
                     label="fig:metodos_lossab")
            + C.table(["Modelo", "Origens", "ΔMSE", "DM", "p", "Melhor"],
                      ab_rows,
                      "Diebold–Mariano entre a loss ponderada por "
                      "horizonte (canônica) e MSE pura, mesma arquitetura "
                      "e seed.",
                      label="tab:metodos_lossab",
                      note="Prática de ablação: sheng2026shape "
                           "(WYM-1/2/3); aslam2026physics (5 perdas)."),
            sub="Só a TCN-Bi difere — a favor do MSE puro (p=0,028). A "
                "motivação da perda ponderada descrita no artigo fica "
                "sem sustentação estatística nesta rodada.",
            sec_id="lossab"))

    # ---------------- B8 Resíduos ----------------
    res_rows = [(r["modelo"], fmt(r["media"]), fmt(r["desvio"]),
                 fmt(r["assimetria"], 2), fmt(r["curtose"], 2),
                 fmt(r["jarque_bera_p"], 4))
                for r in d.get("residuals", [])]
    near_rows = [(r["modelo"], fmt(r["MAE_perto_break"]),
                  fmt(r["MAE_longe_break"]),
                  fmt(r["frac_amostras_perto"]))
                 for r in d.get("near_break", [])]
    brk = ", ".join(d.get("breakpoints", [])) or "nenhum"
    body = (C.figure("figures/fig_resid.png",
                     "Histogramas de resíduos com a Normal de referência: "
                     "caudas pesadas (leptocurtose) em todos os modelos.",
                     label="fig:metodos_resid", width=1.0)
            + C.figure("figures/fig_breaks.png",
                       "Série real no período avaliado com mudanças de "
                       "média (Page–Hinkley) e MAE perto × longe dos "
                       "breakpoints.",
                       label="fig:metodos_breaks", width=1.0)
            + C.table(["Modelo", "Média", "Desvio", "Assim.", "Curtose",
                       "JB p"], res_rows,
                      "Diagnóstico dos resíduos por modelo.",
                      label="tab:metodos_resid")
            + C.table(["Modelo", "MAE perto", "MAE longe", "Frac. perto"],
                      near_rows,
                      "Erro condicionado aos breakpoints detectados "
                      "(±6 h).",
                      label="tab:metodos_breaks",
                      note="Page (1954), Biometrika 41(1/2) 100–115 "
                           "(citado em cao2026probabilistic)."))
    parts.append(C.section(
        "Resíduos e mudanças estruturais",
        body,
        sub=f"Caudas mais pesadas que a Normal ⇒ IC via bootstrap, não "
            f"Gaussianidade. A hipótese “o erro se concentra nas "
            f"transições” foi REFUTADA (perto ≈ longe). Breakpoints: "
            f"{brk}.",
        sec_id="residuos"))

    return "".join(parts)


REPORT = {
    "title": "Análises metodológicas complementares",
    "description": "DM (HAC+HLN), skill vs. persistência, estratos por "
                   "regime/intensidade, caudas, custo, KS/ECDF, fase, A/B "
                   "de loss e resíduos — DL e baselines na mesma grade de "
                   "origens.",
    "regenerate": [".venv/bin/python", "tests/metodos_complementares.py",
                   "--models-dir", "pipeline/tmp/trained_8020",
                   "--loss-ab-dir", "pipeline/tmp/trained_8020_mseloss",
                   "--out-dir", "data/results/metodos_complementares"],
    "retrain": None,
    "source_dir": "data/results/metodos_complementares",
    "info": [
        ("Rodada", "trained_8020 (loss ponderada) × trained_8020_mseloss"),
        ("Base", "1196 origens × 36 horizontes (DL + baselines alinhados)"),
        ("Testes", "DM com HAC + HLN; bootstrap estacionário (2000, bloco 24)"),
        ("Fontes", "docs/revisions/sbr/guia_metricas.ai.md (DOIs verificados)"),
    ],
}

if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
