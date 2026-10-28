"""Relatório — ITILDE-Q nas LSTMs: estudo de seeds (80/20).

Para cada arquitetura (LSTM e LSTM-Bi), treina a **mesma** configuração com
várias seeds do TensorFlow e analisa a distribuição do erro, o ranking das
seeds e a estabilidade da melhor/pior escolha. Tudo é fixo exceto a seed:
HPs canônicos 80/20, 60 épocas, janela 72→36, alvo ``ws100_wavelet`` sym18
nível 2 e loss ITILDE-Q (α=0,3, β=0,5, γ=0,1).

Fontes:
    data/results/itildeq_seeds_8020/itildeq_seeds_8020.json + images/*.png
Gerador: scripts/experiments/lidar_8020_itildeq_seeds_train.py (GPU/Kaggle) +
         studies/itildeq_seeds_8020.py (análise, CPU).

    .venv/bin/python reports/lidar/8020-itildeq-seeds/report.py
"""

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src import components as C  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "results" / "itildeq_seeds_8020"
LOCAL_LOSSAB = (PROJECT_ROOT / "pipeline" / "tmp" /
                "trained_8020_lstm_lossab" / "summary.json")
META_JSON = DATA_DIR / "itildeq_seeds_8020.json"
IMG_DIR = DATA_DIR / "images"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIGS = (
    "fig_mse_seeds.png", "fig_seed_ranking.png", "fig_error_hist.png",
    "fig_error_box.png", "fig_ecdf.png", "fig_qq.png",
    "fig_mse_horizon.png", "fig_bias_heatmap.png",
    "fig_bootstrap_rank.png", "fig_rolling.png", "fig_arch_compare.png",
    "fig_pred_scatter.png",
)
WRAPPERS = ("lstm", "lstm_bi")


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


def pct(x, nd=1):
    if x is None:
        return "—"
    f = float(x)
    if f != f:
        return "—"
    return f"{100.0 * f:.{nd}f} %".replace(".", ",")


def p_txt(p):
    if p is None or p != p:
        return "—"
    return "< 0,001" if p < 0.001 else fmt(p, 4)


def copy_figures():
    if IMG_DIR.is_dir():
        FIG_DIR.mkdir(exist_ok=True)
        for name in FIGS:
            src = IMG_DIR / name
            if src.is_file():
                shutil.copy2(src, FIG_DIR / name)


def hp_row(wrapper, variants):
    a = variants.get(wrapper)
    if not a:
        return None
    hp = {}
    for run in a["metrics"].values():
        if run.get("hyperparameters"):
            hp = run["hyperparameters"]
            break
    return (a["label"],
            fmt(hp.get("lstm_units"), 0),
            f"{hp.get('learning_rate', float('nan')):.2e}".replace(".", ","),
            fmt(hp.get("weight_decay"), 4),
            fmt(hp.get("encoder_dropout_rate"), 2),
            fmt(hp.get("decoder_dropout_rate"), 2),
            fmt(hp.get("batch_size", 32), 0))


def per_horizon_value(run, horizon, key="mse"):
    for row in run["per_horizon"]:
        if int(row["horizon"]) == horizon:
            return row.get(key)
    return None


def build(ctx):  # noqa: ARG001 (contrato do builder)
    copy_figures()
    if not META_JSON.is_file():
        return C.page_header(
            "ITILDE-Q nas LSTMs — estudo de seeds (80/20)",
            "Análise ainda não executada.") + C.note(
            "Rode <code>scripts/experiments/lidar_8020_itildeq_seeds_train.py</code> e "
            "<code>studies/itildeq_seeds_8020.py</code> (ou o botão "
            "<b>Regenerar</b>).")

    d = json.loads(META_JSON.read_text(encoding="utf-8"))
    variants = d.get("variants", {})
    proto = d.get("protocol", {})
    if not variants:
        return C.page_header(
            "ITILDE-Q nas LSTMs — estudo de seeds (80/20)",
            "Sem variantes no JSON.") + C.note(
            "Rode <code>studies/itildeq_seeds_8020.py</code>.")

    available = [w for w in WRAPPERS if w in variants]
    n_seeds = max(len(variants[w]["seeds"]) for w in available)
    total_models = sum(len(variants[w]["seeds"]) for w in available)
    coeffs = proto.get("loss_coeffs", {})

    parts = [C.page_header(
        "ITILDE-Q nas LSTMs — estudo de seeds (80/20)",
        "A mesma rede, os mesmos hiperparâmetros e a mesma perda; só a seed "
        "muda. O objetivo é medir o quanto o resultado depende do sorteio "
        "inicial e como o erro se distribui entre a melhor e a pior seed.",
        badges=(f"{n_seeds} seeds", "LSTM / LSTM-Bi", "ITILDE-Q",
                "80/20 sem validação", "bootstrap + DM"))]

    parts.append(C.cards([
        (str(n_seeds), "seeds por arquitetura"),
        (str(len(available)), "arquiteturas"),
        (str(total_models), "modelos treinados"),
        ("60", "épocas fixas · 72→36"),
    ]))

    # --- desenho -----------------------------------------------------------
    hp_rows = [row for row in (hp_row(w, variants) for w in available)
               if row is not None]

    local_note = ""
    if LOCAL_LOSSAB.is_file() and "lstm" in variants and "lstm_bi" in variants:
        try:
            local = json.loads(LOCAL_LOSSAB.read_text(encoding="utf-8"))
            pairs = (("lstm", "lstm_itq"), ("lstm_bi", "lstm_bi_itq"))
            bits = []
            for wrapper, key in pairs:
                local_mse = local[key]["all"]["mse"]
                kaggle_mse = variants[wrapper]["seed_mse"].get("42")
                if kaggle_mse is not None:
                    bits.append(
                        f"{variants[wrapper]['label']}: local "
                        f"{fmt(local_mse, 4)} × Kaggle {fmt(kaggle_mse, 4)}")
            if bits:
                local_note = C.note(
                    "Checagem de reprodutibilidade com a <b>seed 42</b>: "
                    + "; ".join(bits)
                    + ". A diferença entre ambientes (GPU/cuDNN) é uma fonte "
                    "extra de variância que o estudo entre seeds não separa "
                    "— por isso o ranking vale para esta rodada do Kaggle.")
        except (KeyError, ValueError, OSError):
            local_note = ""

    parts.append(C.section(
        "Desenho do experimento",
        "<p>O estudo isola a <b>variabilidade de seed</b>: hiperparâmetros "
        "canônicos do protocolo 80/20, 60 épocas fixas (sem early stopping), "
        "split temporal 80/20 sem validação, janela de 72 passos (12 h) para "
        "prever 36 passos (6 h), alvo <code>ws100_wavelet</code> (sym18, "
        "nível 2) e métricas contra o <code>ws100</code> bruto. A perda é a "
        f"ITILDE-Q com α = {fmt(coeffs.get('alpha'), 2)}, β = "
        f"{fmt(coeffs.get('beta'), 2)} e γ = {fmt(coeffs.get('gamma'), 2)}. "
        "As seeds são "
        + ", ".join(str(s) for s in variants[available[0]]["seeds"])
        + ". Cada (arquitetura, seed) foi treinado em um processo limpo "
        "(<code>K.clear_session()</code> + <code>set_random_seed</code>) e "
        "as predições densas (todas as origens × 36 horizontes) e rolling "
        "(h=36) foram gravadas para a análise.</p>"
        + C.table(["Arquitetura", "unidades", "lr", "weight decay",
                   "dropout enc.", "dropout dec.", "batch"], hp_rows,
                  "Hiperparâmetros fixos por arquitetura (idênticos entre "
                  "as seeds).", label="tab:seeds_hp",
                  note="HPs canônicos de <code>pipeline/tmp/trained_8020</code>; "
                       "a única variável do estudo é a seed.")
        + C.note(
            "No Kaggle, <code>lstm</code> e <code>lstm_bi</code> rodam em "
            "<b>kernels separados</b> (uma GPU cada), com o mesmo dataset e "
            "o mesmo protocolo. O treino local é possível pelo mesmo script "
            "(sequencial).")
        + local_note,
        sub="HPs fixos, seed variável — o desenho isola o efeito do sorteio "
            "inicial.", sec_id="desenho"))

    # --- ranking -----------------------------------------------------------
    rank_tables = []
    for wrapper in available:
        a = variants[wrapper]
        rows = []
        for rank, seed in enumerate(a["ranking"], start=1):
            m = a["metrics"][str(seed)]
            es = m["error_stats_all"]
            boot = a["bootstrap"][str(seed)]
            tag = "melhor" if seed == a["best_seed"] else (
                "pior" if seed == a["worst_seed"] else "")
            rows.append((f"{seed} ({tag})" if tag else str(seed),
                         fmt(m["metrics_all"]["mse"], 4),
                         fmt(m["metrics_all"]["rmse"], 4),
                         fmt(m["metrics_all"]["mae"], 4),
                         fmt(m["metrics_all"]["r2"], 3),
                         fmt(es["bias"], 3),
                         fmt(es["p95_ae"], 3),
                         fmt(m["skill_rolling"], 3),
                         pct(boot["p_best"], 1),
                         pct(boot["p_worst"], 1)))
        rank_tables.append(C.table(
            ["Seed", "MSE", "RMSE", "MAE", "R²", "Viés", "P95 |e|",
             "Skill rolling", "P(melhor)", "P(pior)"], rows,
            f"{a['label']} — ranking das seeds por MSE (denso, todos os "
            "horizontes).", label=f"tab:seeds_rank_{wrapper}",
            highlight_rows=(0, len(rows) - 1),
            note="P(melhor)/P(pior) vêm do bootstrap em blocos pareado "
                 "entre seeds (2000 reamostragens de 24 h)."))

    best_worst_lines = []
    for wrapper in available:
        a = variants[wrapper]
        b, w = a["best_seed"], a["worst_seed"]
        mb = a["seed_mse"][str(b)]
        mw = a["seed_mse"][str(w)]
        best_worst_lines.append(
            f"<li><b>{a['label']}:</b> melhor seed <b>{b}</b> "
            f"(MSE {fmt(mb, 4)}); pior seed <b>{w}</b> "
            f"(MSE {fmt(mw, 4)}) — a pior é "
            f"<b>{mw / mb:.2f}×</b> o MSE da melhor.</li>")

    parts.append(C.section(
        "Ranking das seeds — melhor e pior",
        "<p>As tabelas abaixo ordenam as seeds por MSE denso. O destaque "
        "verde/vermelho marca a melhor e a pior; as colunas "
        "<b>P(melhor)</b> e <b>P(pior)</b> mostram a probabilidade de cada "
        "seed ocupar cada extremo no bootstrap em blocos — se essa "
        "probabilidade for baixa, a diferença entre seeds é compatível com "
        "ruído amostral.</p>"
        + "".join(rank_tables)
        + C.figure("figures/fig_mse_seeds.png",
                   "MSE por seed com intervalo de confiança 95 % (bootstrap "
                   "em blocos de 24 h). Verde = melhor, vermelho = pior; a "
                   "linha tracejada é a média entre seeds.",
                   label="fig:seeds_mse", width=1.0)
        + C.figure("figures/fig_seed_ranking.png",
                   "Ranking completo das seeds por arquitetura.",
                   label="fig:seeds_ranking", width=0.92),
        sub="<ul>" + "".join(best_worst_lines) + "</ul>",
        sub_after=True, sec_id="ranking"))

    # --- distribuição do erro ---------------------------------------------
    rows = []
    for wrapper in available:
        a = variants[wrapper]
        for kind, key in (("melhor", "best"), ("pior", "worst")):
            seed = a[f"{key}_seed"]
            run = a["metrics"][str(seed)]
            es = run["error_stats_all"]
            rows.append((f"{a['label']} — {kind} ({seed})", str(es["n"]),
                         fmt(es["bias"], 3), fmt(es["std"], 3),
                         fmt(es["mae"], 3), fmt(es["rmse"], 3),
                         fmt(es["median_ae"], 3), fmt(es["p95_ae"], 3),
                         fmt(es["p99_ae"], 3), fmt(es["max_ae"], 2),
                         fmt(es["skew"], 2), fmt(es["kurtosis"], 2),
                         pct(es["within_1"], 1), pct(es["within_2"], 1)))
        es = a["error_stats_pooled"]
        rows.append((f"{a['label']} — todas as seeds", str(es["n"]),
                     fmt(es["bias"], 3), fmt(es["std"], 3),
                     fmt(es["mae"], 3), fmt(es["rmse"], 3),
                     fmt(es["median_ae"], 3), fmt(es["p95_ae"], 3),
                     fmt(es["p99_ae"], 3), fmt(es["max_ae"], 2),
                     fmt(es["skew"], 2), fmt(es["kurtosis"], 2),
                     pct(es["within_1"], 1), pct(es["within_2"], 1)))

    parts.append(C.section(
        "Distribuição do erro",
        "<p>O erro é definido como <b>previsto − observado</b> no "
        "<code>ws100</code> bruto, em m/s. Um viés positivo indica "
        "superestimação média; desvio-padrão, quantis e caudas mostram se a "
        "distribuição é simétrica ou se uma seed troca acertos médios por "
        "erros extremos. A curtose (excesso, normal = 0) mede o peso das "
        "caudas; <b>≤1 m/s</b> e <b>≤2 m/s</b> são as frações de previsões "
        "dentro dessas tolerâncias.</p>"
        + C.table(["Variante", "n", "Viés", "Desvio", "MAE", "RMSE",
                   "Mediana |e|", "P95 |e|", "P99 |e|", "Máx |e|",
                   "Assim.", "Curtose", "≤1 m/s", "≤2 m/s"], rows,
                  "Estatísticas descritivas do erro — melhor seed, pior seed "
                  "e todas as seeds agrupadas (denso, todos os horizontes).",
                  label="tab:seeds_error_stats",
                  note="Assimetria e curtose sem correção de viés amostral; "
                       "unidades de erro em m/s.")
        + C.figure("figures/fig_error_hist.png",
                   "Densidade do erro: melhor seed (verde), pior seed "
                   "(vermelho) e todas as seeds agrupadas (cinza tracejado).",
                   label="fig:seeds_hist", width=1.0)
        + C.figure("figures/fig_error_box.png",
                   "Boxplot do erro por seed (sem outliers gráficos), na "
                   "ordem de MSE; verde = melhor, vermelho = pior.",
                   label="fig:seeds_box", width=1.0)
        + C.grid(
            C.figure("figures/fig_ecdf.png",
                     "ECDF do erro absoluto — probabilidade acumulada de "
                     "|erro| até cada limiar.", label="fig:seeds_ecdf",
                     width=0.5)
            + C.figure("figures/fig_qq.png",
                       "QQ-plot do erro contra a normal (2×2): melhor e pior "
                       "seed de cada arquitetura.", label="fig:seeds_qq",
                       width=0.5))
        + C.figure("figures/fig_pred_scatter.png",
                   "Previsto × observado (hexbin, escala log) para a melhor "
                   "e a pior seed de cada arquitetura; a diagonal tracejada "
                   "é a previsão perfeita.", label="fig:seeds_scatter",
                   width=1.0),
        sub="A comparação é sempre entre a melhor e a pior seed da mesma "
            "arquitetura, mais a distribuição agrupada de todas as seeds.",
        sec_id="distribuicao"))

    # --- por horizonte -----------------------------------------------------
    hz_rows = []
    for wrapper in available:
        a = variants[wrapper]
        for kind, key in (("melhor", "best"), ("pior", "worst")):
            seed = a[f"{key}_seed"]
            run = a["metrics"][str(seed)]
            hz_rows.append((
                f"{a['label']} — {kind} ({seed})",
                fmt(per_horizon_value(run, 6, "mse"), 4),
                fmt(per_horizon_value(run, 12, "mse"), 4),
                fmt(per_horizon_value(run, 24, "mse"), 4),
                fmt(per_horizon_value(run, 36, "mse"), 4),
                fmt(per_horizon_value(run, 6, "mae"), 3),
                fmt(per_horizon_value(run, 12, "mae"), 3),
                fmt(per_horizon_value(run, 24, "mae"), 3),
                fmt(per_horizon_value(run, 36, "mae"), 3),
            ))

    parts.append(C.section(
        "Erro por horizonte e viés",
        "<p>O MSE cresce com o horizonte; a pergunta é se a vantagem da "
        "melhor seed aparece cedo (calibração de curto prazo) ou se ela só "
        "se materializa nos passos longos. O heatmap de viés mostra, seed a "
        "seed, em que horizontes o modelo sistematicamente superestima "
        "(vermelho) ou subestima (azul).</p>"
        + C.table(["Variante", "MSE h6", "MSE h12", "MSE h24", "MSE h36",
                   "MAE h6", "MAE h12", "MAE h24", "MAE h36"], hz_rows,
                  "MSE e MAE nos horizontes 6/12/24/36 (1 h/2 h/4 h/6 h).",
                  label="tab:seeds_horizon")
        + C.figure("figures/fig_mse_horizon.png",
                   "MSE por horizonte — cinza: todas as seeds; verde/vermelho: "
                   "melhor/pior.", label="fig:seeds_horizon", width=1.0)
        + C.figure("figures/fig_bias_heatmap.png",
                   "Viés médio (previsto − observado) por seed e horizonte; "
                   "vermelho = superestima, azul = subestima.",
                   label="fig:seeds_bias", width=1.0),
        sub="O eixo y dos heatmaps segue o ranking de MSE; a melhor seed "
            "fica no topo.", sec_id="horizonte"))

    # --- variabilidade -----------------------------------------------------
    var_rows = []
    for wrapper in available:
        a = variants[wrapper]
        dm = a.get("dm_best_vs_worst") or {}
        var_rows.append((
            a["label"], fmt(a["seed_mse_mean"], 4),
            fmt(a["seed_mse_std"], 4), pct(a["seed_mse_cv"], 1),
            fmt(a["seed_mse_range"], 4), str(a["best_seed"]),
            str(a["worst_seed"]),
            fmt(a["seed_mse"][str(a["worst_seed"])]
                - a["seed_mse"][str(a["best_seed"])], 4),
            fmt(a["seed_rolling_mse_mean"], 4),
            fmt(a["seed_rolling_mse_std"], 4),
            fmt(dm.get("dm"), 2), p_txt(dm.get("p")),
        ))

    dm_rows = []
    for wrapper in available:
        a = variants[wrapper]
        for seed in a["seeds"]:
            if seed == a["dm_reference_seed"]:
                continue
            row = a["dm_vs_reference"][str(seed)]
            dm_rows.append((a["label"], str(seed),
                            fmt(row["delta_mse"], 4), fmt(row["dm"], 2),
                            p_txt(row["p"]), row["melhor"],
                            "sim" if row["sig"] else "não"))

    dm_table = ""
    if dm_rows:
        dm_table = C.table(
            ["Arquitetura", "Seed", "ΔMSE vs 42", "DM", "p", "Melhor",
             "Sig.?"], dm_rows,
            "Diebold–Mariano de cada seed contra a seed 42 "
            "(referência histórica do projeto); DM positivo = a seed "
            "tem perda maior que a 42.", label="tab:seeds_dm")

    arch_rows = []
    for wrapper in available:
        a = variants[wrapper]
        arch_rows.append((
            a["label"], fmt(a["seed_mse_mean"], 4), fmt(a["seed_mse_std"], 4),
            pct(a["seed_mse_cv"], 1),
            f"{a['best_seed']} / {a['worst_seed']}",
            fmt(a["seed_mae_mean"], 4), fmt(a["seed_r2_mean"], 3),
        ))

    parts.append(C.section(
        "Variabilidade entre seeds e estabilidade do ranking",
        "<p>Esta é a pergunta central do estudo: <b>quanto do resultado é "
        "sorte</b>? O desvio e o coeficiente de variação (CV) do MSE entre "
        "seeds quantificam a dispersão; a amplitude mostra o pior caso. O "
        "bootstrap pareado (mesmos blocos de origens para todas as seeds) "
        "dá o intervalo de confiança de cada MSE e a probabilidade de cada "
        "seed ser a melhor/pior. O teste de Diebold–Mariano pareado por "
        "origem verifica se a diferença entre a melhor e a pior seed "
        "sobrevive à autocorrelação das janelas sobrepostas.</p>"
        + C.table(["Arquitetura", "MSE médio", "Desvio", "CV", "Amplitude",
                   "Melhor", "Pior", "Δ(pior−melhor)", "MSE rolling médio",
                   "Desvio rolling", "DM pior×melhor", "p"], var_rows,
                  "Resumo da variabilidade entre seeds e DM da pior contra a "
                  "melhor.", label="tab:seeds_variability",
                  note="DM positivo = a pior seed tem perda maior que a "
                       "melhor. Como melhor/pior são escolhidas após ver o "
                       "teste, o p-valor é descritivo (otimista).")
        + C.figure("figures/fig_bootstrap_rank.png",
                   "Probabilidade de cada seed ser a melhor (verde) e a pior "
                   "(vermelho) no bootstrap em blocos.",
                   label="fig:seeds_bootstrap", width=1.0)
        + dm_table
        + C.figure("figures/fig_arch_compare.png",
                   "Comparação LSTM × LSTM-Bi: MSE por seed e dispersão.",
                   label="fig:seeds_arch", width=1.0)
        + C.table(["Arquitetura", "MSE médio", "Desvio", "CV",
                   "Melhor / pior", "MAE médio", "R² médio"], arch_rows,
                  "Comparação das arquiteturas sob a mesma perda e o mesmo "
                  "protocolo.", label="tab:seeds_arch"),
        sub="Use o CV para comparar a sensibilidade das arquiteturas à seed "
            "(CV alto = resultado mais dependente do sorteio inicial).",
        sec_id="variabilidade"))

    # --- rolling -----------------------------------------------------------
    parts.append(C.section(
        "Rolling autônomo (h=36)",
        "<p>No rolling, cada previsão de 6 h é feita a partir de história "
        "real (não realimentada), mas em todas as origens do teste — é a "
        "condição operacional e a que mais expõe diferenças de trajetória. "
        "A figura compara observado, melhor seed e pior seed.</p>"
        + C.figure("figures/fig_rolling.png",
                   "Rolling h=36 no teste: observado (preto), melhor seed "
                   "(verde) e pior seed (vermelho).",
                   label="fig:seeds_rolling", width=1.0),
        sec_id="rolling"))

    # --- leitura -----------------------------------------------------------
    bullets = []
    for wrapper in available:
        a = variants[wrapper]
        b, w = a["best_seed"], a["worst_seed"]
        mb = a["seed_mse"][str(b)]
        mw = a["seed_mse"][str(w)]
        dm = a.get("dm_best_vs_worst") or {}
        max_pbest = max(a["bootstrap"][str(s)]["p_best"] for s in a["seeds"])
        top_seed = max(a["seeds"],
                       key=lambda s: a["bootstrap"][str(s)]["p_best"])
        if dm:
            sig = ("significativa" if dm["p"] < 0.05
                   else "não significativa")
        else:
            sig = "não avaliada (só 1 seed)"
        ref_seed = a.get("dm_reference_seed")
        ref_rows = a.get("dm_vs_reference", {})
        n_sig = sum(1 for row in ref_rows.values() if row.get("sig"))
        if ref_rows:
            if n_sig == 0:
                ref_txt = (f"nenhuma das {len(ref_rows)} seeds difere "
                           "significativamente da referência")
            else:
                ref_txt = (f"{n_sig} de {len(ref_rows)} seeds diferem "
                           "significativamente da referência")
        else:
            ref_txt = "referência não avaliada"
        bullets.append(
            f"<li><b>{a['label']}:</b> melhor <b>{b}</b> (MSE {fmt(mb, 4)}, "
            f"R² {fmt(a['metrics'][str(b)]['metrics_all']['r2'], 3)}), pior "
            f"<b>{w}</b> (MSE {fmt(mw, 4)}, R² "
            f"{fmt(a['metrics'][str(w)]['metrics_all']['r2'], 3)}). A pior "
            f"seed tem <b>{mw / mb:.2f}×</b> o MSE da melhor; o CV entre "
            f"seeds é <b>{pct(a['seed_mse_cv'], 1)}</b>. O DM pior×melhor "
            f"(DM = {fmt(dm.get('dm'), 2)}, p = {p_txt(dm.get('p'))}) é "
            f"{sig} — lembrando que o par foi escolhido após o teste.</li>"
            f"<li>No bootstrap, a seed com maior P(melhor) é a <b>{top_seed}</b> "
            f"(P = {pct(max_pbest, 1)}); quando esse valor é baixo, várias "
            "seeds são indistinguíveis e escolher a \"campeã\" é "
            "sobreajustar o conjunto de teste.</li>"
            f"<li>Contra a seed de referência <b>{ref_seed}</b> "
            f"(DM pareado, α = 0,05): {ref_txt}.</li>")

    if "lstm" in variants and "lstm_bi" in variants:
        li, lb = variants["lstm"], variants["lstm_bi"]
        better = "LSTM" if li["seed_mse_mean"] < lb["seed_mse_mean"] else \
            "LSTM-Bi"
        more_stable = "LSTM" if li["seed_mse_cv"] < lb["seed_mse_cv"] else \
            "LSTM-Bi"
        bullets.append(
            f"<li><b>Arquiteturas:</b> em média a <b>{better}</b> tem MSE "
            f"menor ({fmt(min(li['seed_mse_mean'], lb['seed_mse_mean']), 4)} "
            f"vs {fmt(max(li['seed_mse_mean'], lb['seed_mse_mean']), 4)}), "
            f"mas a <b>{more_stable}</b> é mais estável entre seeds "
            f"(CV {pct(min(li['seed_mse_cv'], lb['seed_mse_cv']), 1)} vs "
            f"{pct(max(li['seed_mse_cv'], lb['seed_mse_cv']), 1)}). Uma seed "
            "boa de uma arquitetura pode empatar com a média da outra: a "
            "comparação de arquitetura deve usar a distribuição, não o "
            "melhor ponto.</li>")

    bullets.append(
        "<li><b>Distribuição do erro:</b> a melhor e a pior seed não diferem "
        "só na média — valem as caudas. Compare P95/P99 e a fração dentro de "
        "1 m/s na tabela de estatísticas: uma seed pode ter MSE parecido e "
        "ainda assim mais erros extremos, o que importa para operação.</li>")
    bullets.append(
        "<li><b>Recomendação prática:</b> reporte <b>média ± desvio entre "
        "seeds</b> (ou um ensemble de seeds) em vez de um único número, e "
        "não selecione a seed pelo teste. Se for necessário fixar uma seed "
        "por reprodutibilidade, prefira uma com bom desempenho <i>e</i> alta "
        "P(melhor), documentando o critério.</li>")

    parts.append(C.section(
        "Leitura",
        "<div class='markdown'><ul>" + "".join(bullets) + "</ul></div>",
        sec_id="leitura"))

    return "".join(parts)


REPORT = {
    "title": "ITILDE-Q nas LSTMs — estudo de seeds (80/20)",
    "description": "Várias seeds do TensorFlow com a mesma LSTM/LSTM-Bi e a "
                   "loss ITILDE-Q: distribuição do erro, ranking, melhor/pior "
                   "seed e estabilidade (bootstrap + Diebold–Mariano).",
    "regenerate": [".venv/bin/python", "studies/itildeq_seeds_8020.py",
                   "--models-dir", "pipeline/tmp/trained_8020_itildeq_seeds",
                   "--out-dir", "data/results/itildeq_seeds_8020"],
    "retrain": [".venv/bin/python",
                "scripts/experiments/lidar_8020_itildeq_seeds_train.py"],
    "source_dir": "data/results/itildeq_seeds_8020",
    "info": [
        ("Rodada", "trained_8020_itildeq_seeds"),
        ("Protocolo", "80/20 sem validação · 60 épocas · 72→36"),
        ("Variável", "apenas a seed (42–51)"),
        ("Loss", "ITILDE-Q (α=0,3 · β=0,5 · γ=0,1)"),
        ("Testes", "bootstrap em blocos + DM pareado"),
        ("Kaggle", "1 kernel/GPU por arquitetura"),
    ],
}


if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
