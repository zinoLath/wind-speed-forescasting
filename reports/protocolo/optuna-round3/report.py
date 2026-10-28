"""Relatório de protocolo — Optuna Round 3 (design + progresso).

Documento vivo da rodada 3 de hiperparâmetros: decisões tomadas com o
autor, especificação do objetivo (skill vs ARIMA com cache anti-vazamento),
espaço de busca realinhado (tamanho prioritário, defaults fixos para
invariantes), matemática das dilatações (lags sazonais 36/72), plano de
workers (2 kernels Kaggle) e seção de PROGRESSO ao vivo que lê os
snapshots do estudo em pipeline/tmp/optuna_round3/.

Executar a partir da raiz:
    .venv/bin/python reports/protocolo/optuna-round3/report.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from reports.src import components as C  # noqa: E402

STUDY_DIR = PROJECT_ROOT / "pipeline" / "tmp" / "optuna_round3"
FIG_DIR = Path(__file__).resolve().parent / "figures"

WRAPPERS = {"Seq2Seq_LSTM": "LSTM", "Seq2Seq_LSTM_Bidirectional": "LSTM-Bi",
            "Seq2Seq_TCN": "TCN", "Seq2Seq_TCN_Bidirectional": "TCN-Bi"}
N_TRIALS_TARGET = 120

plt.rcParams.update({
    "figure.dpi": 130, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.grid": True, "grid.alpha": 0.25,
})


def _progress_data():
    """Lê trials.csv/best_trial.json dos snapshots do estudo round3."""
    out = {}
    for name, label in WRAPPERS.items():
        d = STUDY_DIR / name
        csv = d / "trials.csv"
        if csv.is_file():
            df = pd.read_csv(csv)
            done = df[df["state"] == "COMPLETE"]
            out[label] = {"df": df, "done": len(done),
                          "best": float(done["value"].min())
                          if len(done) else None,
                          "best_trial": json.loads(
                              (d / "best_trial.json").read_text("utf-8"))
                          if (d / "best_trial.json").is_file() else None}
    return out


def _progress_figure(data):
    if not data:
        return C.note("Nenhum snapshot encontrado ainda em "
                      "<code>pipeline/tmp/optuna_round3/</code>. A seção "
                      "popula assim que os kernels começarem a rodar (ou "
                      "após <code>collect</code>).")
    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    rows = []
    for label, info in data.items():
        done = info["df"][info["df"]["state"] == "COMPLETE"] \
            .sort_values("number")
        if not len(done):
            continue
        running = done["value"].cummin()
        ax.plot(done["number"], running, lw=1.8, marker="o", ms=2.5,
                label=f"{label} ({len(done)} trials)")
        rows.append((label, str(len(done)),
                     f"{info['best']:.6f}" if info["best"] is not None
                     else "—"))
    ax.set_xlabel("trial")
    ax.set_ylabel("melhor objetivo acumulado (menor = melhor)")
    ax.set_title("Progresso do estudo round3 — running best por wrapper")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "progresso_round3.png")
    plt.close(fig)
    body = C.figure("figures/progresso_round3.png",
                    "Running best do objetivo (skill vs ARIMA, blocado) por "
                    "wrapper ao longo dos trials.",
                    label="fig:round3_progresso")
    if rows:
        body += C.table(["Wrapper", "Trials completos", "Melhor objetivo"],
                        rows, "Progresso acumulado por wrapper.",
                        label="tab:round3_progresso")
    return body


def build(ctx):  # noqa: ARG001
    data = _progress_data()

    parts = [C.page_header(
        "Optuna Round 3 — design e progresso",
        "Rodada de hiperparâmetros sob o protocolo 80/20 do artigo: "
        "objetivo skill vs ARIMA (cache anti-vazamento), espaços "
        "realinhados com defaults fixos para invariantes, 2 kernels "
        "Kaggle.",
        badges=("80/20", "skill vs ARIMA", "60 épocas", "TPE",
                "Kaggle ×2"))]

    # ------------------ decisões ------------------
    parts.append(C.section(
        "Decisões (com o autor)",
        "<div class='markdown'><ul>"
        "<li><strong>Split</strong>: protocolo do artigo — treino 80% / "
        "teste 20% intocado; validação do HPO = <strong>últimos 10% do "
        "bloco de 80%</strong> (treino efetivo ~72%), 4 blocos internos "
        "para robustez a regimes.</li>"
        "<li><strong>Objetivo</strong>: média em blocos de "
        "<code>1 − MSE_modelo/MSE_ARIMA</code> com peso de horizonte "
        "(<strong>skill vs ARIMA</strong>); hw-MSE e nº de parâmetros "
        "gravados como user_attrs.</li>"
        "<li><strong>Orçamento/trial</strong>: 60 épocas fixas "
        "(regime do protocolo final) + MedianPruner por época.</li>"
        "<li><strong>LR</strong>: ReduceLROnPlateau <strong>fixo</strong> "
        "(factor 0,5, patience 6) — fora do HPO.</li>"
        "<li><strong>Tamanho</strong>: units/filters 48–252 (passo 4) — "
        "inclui os bests do round 1 (64u, 48–80f) e dá teto maior; o "
        "passo 4 é a grade fina que cobre os valores legados e o 252.</li>"
        "<li><strong>Kernel/stacks</strong>: voltam ao HPO — interagem com "
        "as dilatações no campo receptivo (RF = 1+2(k−1)·stacks·Σd). "
        "Tamanho é só uma frente de regularização.</li>"
        "<li><strong>Loss por modelo</strong> (decisão da ablação de "
        "loss, out/2026): LSTM/&shy;LSTM-Bi → <strong>ITILDE-Q</strong> "
        "(shape-aware, melhor no report alinhado); TCN/&shy;TCN-Bi → "
        "<strong>MSE puro</strong> (ITILDE-Q não ajudou as TCN; MSE empatou "
        "com a hwmse). Mapa em <code>LOSS_BY_WRAPPER</code> "
        "(<code>scripts/experiments/lidar_7525_train.py</code>); <code>--legacy-hwmse</code> "
        "volta ao default antigo.</li>"
        "<li><strong>Dilatações</strong>: base estática; sequências "
        "fixas com lag sazonal 36 direto (seção própria abaixo); busca "
        "apenas a profundidade (prefixo).</li>"
        "<li><strong>Workers</strong>: só Kaggle — 2 kernels GPU, "
        "pareamento custo-equilibrado: <code>k1 = lstm + tcn_bi</code>, "
        "<code>k2 = lstm_bi + tcn</code>.</li>"
        "</ul></div>",
        sub="Perguntas em aberto respondidas em 14–15/10/2026; ver "
            "session-comentários.md para o raciocínio completo.",
        sec_id="decisoes"))

    # ------------------ objetivo ------------------
    parts.append(C.section(
        "Objetivo: skill vs ARIMA (com cache)",
        "<div class='markdown'>"
        "<p><code>skill_bloco = 1 − MSE_modelo,bloco / MSE_ARIMA,bloco</code>"
        "; objetivo = média dos 4 blocos com pesos de horizonte "
        "(0,7→1,3). Lê-se como <em>% de melhora sobre o ARIMA</em>; "
        "escala-livre e diretamente reportável no artigo.</p>"
        "<p><strong>Por que não hw-MSE cru:</strong> com referência fixa, "
        "dentro do bloco a ordenação é a mesma do MSE — o que muda é o "
        "<em>peso entre blocos</em>: cada bloco é normalizado pelo erro do "
        "ARIMA naquele bloco, então o estudo prioriza vencer o ARIMA "
        "onde ele é forte (curto prazo), a fraqueza documentada dos "
        "nossos modelos.</p>"
        "<p><strong>Cache anti-vazamento</strong>: ARIMA reajustado por "
        "AIC <strong>só no prefixo de treino</strong> (grade de ordens "
        "restrita ao treino), predições pré-computadas na MESMA grade de "
        "origens da validação — com o fix do drift de microssegundos "
        "(<code>round('min')</code>, o bug do mcs.json) — e distribuídas "
        "como CSV no dataset Kaggle. Persistência é trivial (último valor "
        "observado) e também entra no cache para user_attrs.</p>"
        "</div>",
        sub="Respaldo: FPP3 §5.9 (skill scores, otexts.com/fpp3/accuracy."
            "html); bendes2026bio (skill vs persistência com IC, honesto "
            "com negativos); jiang2026data (reduções % vs benchmark); "
            "significância pós-hoc via Diebold–Mariano (DM 1995 + HLN "
            "1997) já implementada.",
        sec_id="objetivo"))

    # ------------------ espaço de busca ------------------
    search_rows = [
        ("learning_rate", "todos", "log 1e-4 – 2e-2",
         "tamanho do passo do Adam/AdamW; log porque escala em ordens de "
         "grandeza (bests round1: 4e-3–1e-2)"),
        ("weight_decay", "todos", "Choice [0, 1e-5, 1e-4, 3e-4, 1e-3]",
         "regularização L2 do AdamW; 0 mantém Adam puro como categoria"),
        ("units (LSTM) / filters (TCN)", "todos", "48–252, passo 4",
         "largura do codificador/decodificador; cobre os bests legados "
         "(64u, 48–80f) e sobe o teto"),
        ("encoder/decoder dropout", "todos", "0–0,4, passo 0,05",
         "regularização; TCN tolera mais (best 0,4) que LSTM (best 0,05)"),
        ("kernel_size", "TCN", "{2,3}",
         "largura do filtro — junto com stacks e dilatações define o RF"),
        ("nb_stacks", "TCN", "{1,2}",
         "blocos empilhados — 2 dobra RF/custo; interage com as dilatações"),
        ("enc_dil_depth", "TCN", "{4,5,6,7}",
         "quantas camadas do array estático do CODIFICADOR são usadas "
         "(prefixo) — controla quanta história o encoder enxerga "
         "(~5 h / ~10,5 h / 12 h) e o custo; ver seção Dilatações"),
        ("dec_dil_depth", "TCN", "{3,4,5}",
         "idem, no DECODIFICADOR — quanto do horizonte de 6 h o bloco "
         "consegue olhar para trás ao gerar (2,5 h / ~5 h / 6 h)"),
    ]
    fixed_rows = [
        ("LR schedule", "ReduceLROnPlateau (0,5 / patience 6)",
         "decisão do autor — fora do HPO"),
        ("batch_size", "32", "invariante na prática; menos uma dimensão"),
        ("loss", "LSTM/LSTM-Bi: ITILDE-Q · TCN/TCN-Bi: MSE",
         "decisão da ablação de loss (report alinhado; ver "
         "lidar/8020-itildeq)"),
        ("janela / horizonte", "72 → 36 (10 min)", "invariante do dataset"),
        ("alvo / suavização", "ws100_wavelet (sym18 nível 2)", "idem"),
        ("normalização", "min-max ajustada só no treino", "anti-vazamento"),
        ("seed", "42", "reprodutibilidade"),
    ]
    parts.append(C.section(
        "Espaço de busca realinhado",
        C.table(["HP", "Wrappers", "Faixa", "Contexto de uso"],
                search_rows, "O que o TPE busca no round 3.",
                label="tab:round3_search")
        + C.table(["Parâmetro", "Default fixo", "Por quê"], fixed_rows,
                  "Defaults fixos — invariantes do dataset ou decididos.",
                  label="tab:round3_fixos",
                  note="Espaço enxuto (LSTM: 5 dims; TCN: 7) → 120 trials "
                       "por wrapper são confortáveis para o TPE "
                       "multivariado."),
        sub="Filosofia: buscar o que muda de verdade entre sítios e "
            "arquiteturas (tamanho, lr, regularização, profundidade "
            "temporal); fixar o que é invariante do dataset ou já decidido "
            "no protocolo.",
        sec_id="espaco"))

    # ------------------ dilatações ------------------
    dil_rows = [
        ("A: [1,3,6,12,18,24,36] (autor)", "7", "100", "201", "48/72",
         "18 já era 12+6 ⇒ alcance idêntico a B"),
        ("B: [1,3,6,12,24,36] (autor)", "6", "82", "165", "48/72",
         "mesmo alcance de A, uma camada mais barato"),
        ("C′: [1,2,4,8,16,32,36] (TRAVADA)", "7", "99", "199", "72/72",
         "base-2 + salto sazonal 36: cobertura total"),
        ("base-2 (padrão TCN)", "6", "63", "127", "63/72",
         "não alcança 72 nem amostra 36 direto"),
    ]
    depth_rows = [
        ("encoder", "4", "[1,2,4,8]", "31", "≈5 h de história"),
        ("encoder", "5", "[1,2,4,8,16]", "63", "≈10,5 h"),
        ("encoder", "6", "[1,2,4,8,16,32]", "127", "12 h (lags 1..63)"),
        ("encoder", "7", "[1,2,4,8,16,32,36]", "199",
         "12 h + lag 36 direto + 72 alcançável"),
        ("decoder", "3", "[1,2,4]", "15", "2,5 h do horizonte"),
        ("decoder", "4", "[1,2,4,8]", "31", "≈5 h"),
        ("decoder", "5", "[1,2,4,8,16]", "63", "6 h (tudo; lags 1..31)"),
    ]
    parts.append(C.section(
        "Dilatações: sequência estática e o HP de profundidade",
        "<div class='markdown'>"
        "<p><strong>Como uma dilatação funciona.</strong> Com kernel 2, "
        "cada camada de dilatação <code>d</code> conecta <em>diretamente</em> "
        "os passos <code>t</code> e <code>t−d</code> — então “capturar a "
        "correlação de 6 h” exige <code>d = 36</code> na lista. O campo "
        "receptivo acumulado é <code>RF = 1 + 2·(k−1)·stacks·Σd</code>. "
        "Limites estruturais: o codificador processa a janela de 72 passos "
        "(12 h) e o decodificador o horizonte de 36 (6 h); a dilatação "
        "máxima útil é <code>comprimento/2</code> (dilatação 72 numa "
        "entrada de 72 vira metade padding). O EDA do artigo justifica os "
        "alvos: ACF forte até 36 passos, reincidência em 72 (ciclo "
        "semidiurno).</p>"
        "<p><strong>O que é <code>dil_depth</code>.</strong> As sequências "
        "são <strong>estáticas</strong> (decisão do autor: base fixa): "
        "encoder <code>[1,2,4,8,16,36]</code>, decoder "
        "<code>[1,2,4,8,16]</code>. O HP buscado é apenas a "
        "<strong>profundidade</strong> — quantas camadas do array são "
        "efetivamente usadas (o prefixo). Ex.: "
        "<code>enc_dil_depth=5</code> ⇒ o encoder roda "
        "<code>[1,2,4,8,16]</code> e enxerga ≈10,5 h de história; "
        "<code>dec_dil_depth=3</code> ⇒ o decoder roda <code>[1,2,4]</code> "
        "e cobre só 2,5 h do horizonte ao gerar. Menos profundidade = "
        "menos parâmetros e menos história — um <em>trade-off</em> que o "
        "TPE decide, não uma garantia.</p>"
        "<p><strong>Importante (mudança vs código atual)</strong>: hoje o "
        "<code>tcn_hp.py</code> estende automaticamente a sequência até o "
        "RF cobrir a janela inteira — sob essa regra o prefixo menor seria "
        "esticado de volta ao array cheio e o HP de profundidade "
        "<em>não faria nada</em>. No round 3 a extensão automática é "
        "<strong>removida</strong>: o prefixo escolhido é exatamente o que "
        "roda, e enxergar menos história passa a ser uma configuração "
        "válida que o Optuna pode preferir (modelo mais barato).</p>"
        "<p><strong>Alcance dos lags.</strong> O lag alcançável entre duas "
        "posições de entrada é a soma de um subconjunto das dilatações "
        "(cada camada escolhe <code>t</code> ou <code>t−d</code>). A "
        "sequência do autor <code>[1,3,6,12,18,24,36]</code> tem "
        "<strong>o mesmo alcance</strong> de <code>[1,3,6,12,24,36]</code> "
        "(o <code>18</code> já era <code>12+6</code>; a camada extra só o "
        "torna uma conexão direta). “1 + múltiplos de 3” deixa "
        "inalcançável todo lag <code>≡ 2 (mod 3)</code> — inclui o lag 2 "
        "(20 min), de autocorrelação alta. Em compensação, as duas acertam "
        "os alvos do EDA: 36 (6 h) direto e 72 (12 h) por composição.</p>"
        "<p><strong>Decisão travada com o autor:</strong> <strong>C′ "
        "<code>[1,2,4,8,16,32,36]</code></strong> (base-2 + salto sazonal) "
        "cobre <strong>todos</strong> os lags 1..72, amostra o 36 direto, "
        "alcança o 72 e custa o mesmo que A (7 camadas). No decoder, a "
        "versão truncada ao máximo útil <code>[1,2,4,8,16]</code>. A "
        "profundidade (prefixo) é o único HP de dilatação.</p>"
        "</div>"
        + C.table(["Sequência", "Camadas", "Σd", "RF", "Lags diretos", "Observação"], dil_rows,
                  "Candidatas de sequência estática de dilatações "
                  "(encoder; decoder usa [1,2,4,8,16], RF 63 ≥ 36, máx útil 18).",
                  label="tab:round3_dilatacoes",
                  highlight_rows=[2],
                  note="Travada: encoder C′ [1,2,4,8,16,32,36] (cobertura "
                       "total 1..72, lag 36 direto, 72 alcançável); "
                       "decoder [1,2,4,8,16].")
        + C.table(["Bloco", "depth", "Prefixo usado", "RF",
                   "História / cobertura"],
                  depth_rows,
                  "O que cada valor de enc/dec_dil_depth significa para a "
                  "sequência travada C′ (encoder [1,2,4,8,16,32,36]; "
                  "decoder [1,2,4,8,16]); kernel 2, 1 stack.",
                  label="tab:round3_depth"),
        sub="Decisão do autor: base estática (não buscada); apenas a "
            "profundidade (prefixo) entra no HPO — com extensão automática "
            "de RF desligada para o HP ser significativo.",
        sec_id="dilatacoes"))

    # ------------------ workers / operação ------------------
    parts.append(C.section(
        "Workers e operação (2 × Kaggle)",
        "<div class='markdown'><ul>"
        "<li><strong>k1</strong>: lstm + tcn_bi · <strong>k2</strong>: "
        "lstm_bi + tcn (rápido+lento por kernel; custo mediano histórico "
        "2/5/3/7 min por trial).</li>"
        "<li>Dataset Kaggle carrega: pacote do repo + "
        "<code>arima_val_cache.csv</code> (predições ARIMA/persistência "
        "na grade da validação).</li>"
        "<li>Um sqlite por wrapper (<code>load_if_exists</code>) — kernel "
        "reiniciado retoma de onde parou; snapshot do progresso no output "
        "a cada 10 min.</li>"
        "<li>Acompanhamento: esta página (seção abaixo) + "
        "<code>python scripts/dist/colab_monitor.py --watch 60</code> para o "
        "estado dos kernels.</li>"
        "</ul></div>",
        sec_id="workers"))

    # ------------------ progresso ao vivo ------------------
    parts.append(C.section(
        "Progresso ao vivo",
        _progress_figure(data),
        sub="Lê os snapshots em <code>pipeline/tmp/optuna_round3/</code> "
            "(trazidos pelo collect/monitor). O botão Regenerar "
            "reconstrói a página com o estado mais recente do disco.",
        sec_id="progresso"))

    return "".join(parts)


REPORT = {
    "title": "Optuna Round 3 — design e progresso",
    "description": "Design da rodada 3 (skill vs ARIMA, 80/20, defaults "
                   "fixos, dilatações estáticas com lag 36) e progresso "
                   "ao vivo do estudo.",
    "regenerate": None,  # rebuild da página já re-lê os snapshots
    "retrain": None,
    "source_dir": "pipeline/tmp/optuna_round3",
    "info": [
        ("Objetivo", "skill vs ARIMA (4 blocos, peso de horizonte)"),
        ("Split", "80/20 do artigo; val = últimos 10% do bloco de treino"),
        ("Orçamento", "60 épocas fixas + MedianPruner; 120 trials/wrapper"),
        ("Workers", "Kaggle ×2 (k1: lstm+tcn_bi; k2: lstm_bi+tcn)"),
    ],
}

if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
