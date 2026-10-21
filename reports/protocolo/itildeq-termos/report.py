"""Relatório — perdas da LSTM: busca de coeficientes ITILDE-Q, hweight e
comparação com baselines.

Fontes:
- data/results/itildeq_terms_search/terms_search.json (grade OFAT no Kaggle)
- pipeline/tmp/itildeq_terms_search/combo_lstm_bi_*.json (MSE pura na Bi)
- data/results/lstm_lossab_8020/lossab_8020.json (A/B hweight × ITILDE-Q,
  DM pareado)
- data/results/report_8020_itildeq/evaluation.json (rodada canônica)
- data/results/baselines/arima_rf/metrics.json

    python reports/protocolo/itildeq-termos/report.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src import components as C  # noqa: E402


def load(path):
    p = PROJECT_ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def h36(entry):
    for rec in entry.get("per_horizon", []):
        if int(rec.get("horizon", -1)) == 36:
            return rec
    return None


def build(ctx):  # noqa: ARG001
    terms = load("data/results/itildeq_terms_search/terms_search.json") or []
    local_bi = [json.loads(p.read_text(encoding="utf-8")) for p in
                sorted((PROJECT_ROOT / "pipeline" / "tmp"
                        / "itildeq_terms_search").glob("combo_lstm_bi_*.json"))]
    lossab = load("data/results/lstm_lossab_8020/lossab_8020.json")
    ev = load("data/results/report_8020_itildeq/evaluation.json")
    base = load("data/results/baselines/arima_rf/metrics.json")

    best_iq = min((r for r in terms if (r["alpha"], r["gamma"]) != (0.0, 0.0)),
                  key=lambda r: r["all"]["mse"], default=None)
    mse_pura = next((r for r in terms if (r["alpha"], r["gamma"]) == (0.0, 0.0)),
                    None)
    default = next((r for r in terms if (r["alpha"], r["beta"], r["gamma"])
                    == (0.3, 0.5, 0.1)), None)

    parts = [C.page_header(
        "ITILDE-Q — busca de coeficientes e perdas da LSTM",
        "Busca curta dos coeficientes (α, β, γ) da perda ITILDE-Q no "
        "Kaggle, comparação das três perdas da LSTM (MSE pura, ITILDE-Q, "
        "hweight) e posicionamento contra as baselines. O artigo canônico "
        "usa ITILDE-Q nas LSTMs; esta página documenta a evidência.",
        badges=("9 combos", "DM pareado", "80/20 seed 42",
                "rodada canônica")),
        C.cards([
            (len(terms), "combos OFAT (LSTM)"),
            (f"+{best_iq['all']['r2']:.3f}" if best_iq else "—",
             "melhor ITILDE-Q (R² all)"),
            (f"+{default['all']['r2']:.3f}" if default else "—",
             "default 0,3/0,5/0,1"),
            (f"+{mse_pura['all']['r2']:.3f}" if mse_pura else "—",
             "MSE pura (α=0)"),
        ])]

    # ---- protocolo
    parts.append(C.section(
        "Como a busca foi feita",
        "<div class='markdown'>"
        "<p><strong>Grade OFAT</strong> (um fator por vez) ao redor do "
        "default 0,3/0,5/0,1: α ∈ {0; 0,1; 0,3; 0,5; 1,0} com β=0,5, γ=0,1; "
        "β ∈ {0,2; 0,8}; γ ∈ {0; 0,3}. Cada combo = <strong>um treino "
        "completo</strong> (LSTM, HPs canônicos, 60 épocas, seed 42, split "
        "80/20), métricas contra <code>ws100</code> bruto; 2 kernels Kaggle. "
        "<strong>Micro-grade</strong> (α ∈ {0,01; 0,05}, γ ∈ {0; 0,01}) nas "
        "duas LSTMs — em execução (kernels "
        "<code>itildeq-terms-micro-a/b</code>).</p>"
        "<p><strong>Ressalvas:</strong> 1 treino por combo (o estudo de "
        "seeds mostra ruído de ±0,07 de R² entre execuções da LSTM; "
        "±0,045 na LSTM-Bi); o único pareado formal disponível é o A/B "
        "hweight × ITILDE-Q do <code>lostab_8020</code>.</p></div>",
        sub="Perda: itilde_q = MSE + α·(β·shift_amplitude + (1−β)·fase + "
            "γ·amplificação) — Eqs. 5–7 de Lee et al. 2022 (TILDE-Q, "
            "arXiv:2210.15050), Eq. 15 de Dong Sheng et al. 2026.",
        sec_id="protocolo"))

    # ---- grade completa
    rows = [(f"α={r['alpha']:.2f} β={r['beta']:.2f} γ={r['gamma']:.2f}",
             f"{r['all']['r2']:.4f}", f"{r['all']['mse']:.4f}",
             f"{r['h36']['r2']:.4f}", f"{r['h36']['mse']:.4f}")
            for r in sorted(terms, key=lambda r: r["all"]["mse"])]
    highlight = [0] if rows else None
    parts.append(C.section(
        "Grade OFAT — LSTM (ordenado por MSE all)",
        C.table(["Combinação", "R² all", "MSE all", "R² h36", "MSE h36"],
                rows, "Combinações da ITILDE-Q (1 treino cada; seed 42).",
                label="tab:itildeq_terms", highlight_rows=highlight),
        sub="α menor = melhor; a MSE pura (α=0, γ=0) lidera e o default "
            "0,3/0,5/0,1 é o 8º de 9. α=1,0 é o pior.",
        sec_id="grade"))

    # ---- perdas por wrapper + baselines
    if lossab and ev and base:
        v = lossab["variants"]
        loss_rows = [
            ("LSTM — MSE pura", f"{mse_pura['all']['r2']:.4f}",
             f"{mse_pura['all']['mse']:.4f}", f"{mse_pura['h36']['r2']:.4f}")
            if mse_pura else ("LSTM — MSE pura", "—", "—", "—"),
            ("LSTM — ITILDE-Q (canônica)",
             f"{ev['Seq2Seq_LSTM']['all']['r2']:.4f}",
             f"{ev['Seq2Seq_LSTM']['all']['mse']:.4f}",
             f"{h36(ev['Seq2Seq_LSTM'])['r2']:.4f}"),
            ("LSTM — ITILDE-Q (lossab)", f"{v['lstm_itq']['all']['r2']:.4f}",
             f"{v['lstm_itq']['all']['mse']:.4f}", "—"),
            ("LSTM — hweight (lossab)", f"{v['lstm_hw']['all']['r2']:.4f}",
             f"{v['lstm_hw']['all']['mse']:.4f}", "—"),
            ("LSTM-Bi — ITILDE-Q (canônica)",
             f"{ev['Seq2Seq_LSTM_Bidirectional']['all']['r2']:.4f}",
             f"{ev['Seq2Seq_LSTM_Bidirectional']['all']['mse']:.4f}",
             f"{h36(ev['Seq2Seq_LSTM_Bidirectional'])['r2']:.4f}"),
            ("LSTM-Bi — hweight (lossab)",
             f"{v['lstm_bi_hw']['all']['r2']:.4f}",
             f"{v['lstm_bi_hw']['all']['mse']:.4f}", "—"),
        ] + ([("LSTM-Bi — MSE pura (local)",
               f"{local_bi[0]['all']['r2']:.4f}",
               f"{local_bi[0]['all']['mse']:.4f}",
               f"{local_bi[0]['h36']['r2']:.4f}")] if local_bi else [])
        parts.append(C.section(
            "Perdas por wrapper",
            C.table(["Variante", "R² all", "MSE all", "R² h36"], loss_rows,
                    "Comparação entre perdas na mesma variante (fontes "
                    "distintas: runs únicos).",
                    label="tab:losses_by_wrapper"),
            sub="LSTM uni: MSE pura > ITILDE-Q > hweight. LSTM-Bi: as três "
                "ficam dentro do ruído de seed (itilde-q 0,314 ± 0,045 em "
                "10 seeds; hweight 0,292; MSE pura 0,276) — perda "
                "indiferente.",
            sec_id="perdas"))

    # ---- DM
    if lossab:
        dm_rows = [(d["arq"], f"{d['delta']:+.3f}", f"{d['dm']:+.2f}",
                    f"{d['p']:.4f}", d["melhor"], "sim" if d["sig"] else "não")
                   for d in lossab["dm"]]
        parts.append(C.section(
            "Teste DM pareado — ITILDE-Q × hweight",
            C.table(["Arquitetura", "ΔMSE", "DM", "p", "Melhor", "Sig."],
                    dm_rows,
                    "A/B com mesmos HPs/seed (n=1203 origens).",
                    label="tab:loss_dm"),
            sub="Na LSTM unidirecional a ITILDE-Q vence com significância "
                "(p=0,0105); na bidirecional não há diferença (p=0,63).",
            sec_id="dm"))

    # ---- baselines
    if mse_pura and ev and base:
        comp = [
            ("TCN (canônica, MSE)", ev["Seq2Seq_TCN"]["all"],
             h36(ev["Seq2Seq_TCN"])),
            ("ARIMA", base["arima"]["all_horizons"], base["arima"]["rolling"]),
            ("LSTM MSE pura (busca)", mse_pura["all"], mse_pura["h36"]),
            ("Random Forest", base["rf"]["all_horizons"], base["rf"]["rolling"]),
            ("Persistência", base["persistence"]["all_horizons"],
             base["persistence"]["rolling"]),
            ("LSTM ITILDE-Q (canônica)", ev["Seq2Seq_LSTM"]["all"],
             h36(ev["Seq2Seq_LSTM"])),
        ]
        rows = [(n, f"{a['r2']:.4f}", f"{a['mse']:.4f}",
                 f"{h['r2']:.4f}", f"{h['mse']:.4f}")
                for n, a, h in comp if h]
        parts.append(C.section(
            "MSE pura × baselines",
            C.table(["Modelo", "R² all", "MSE all", "R² h36", "MSE h36"],
                    rows, "Rodada canônica + baselines na mesma grade de "
                    "origens.", label="tab:mse_vs_baselines"),
            sub="No agregado a LSTM MSE pura empata com o RF e fica abaixo "
                "do ARIMA; no h36 ela supera todas as baselines (0,328 × "
                "0,316 do ARIMA) — padrão do artigo: baselines no curto "
                "prazo, DL nos horizontes longos.",
            sec_id="baselines"))

    parts.append(C.section(
        "Leitura para o artigo",
        "<div class='markdown'><ul>"
        "<li><strong>Hierarquia da LSTM uni</strong>: MSE pura (0,42) &gt; "
        "ITILDE-Q (~0,32 em 3 runs) &gt; hweight (~0,27) — o default da "
        "ITILDE-Q ficou em 8º de 9 na grade;</li>"
        "<li><strong>hweight é a pior opção</strong> para a LSTM — e o "
        "único pareado significativo (ITILDE-Q &gt; hweight, p=0,0105) "
        "aponta na mesma direção;</li>"
        "<li><strong>LSTM-Bi é indiferente à perda</strong> (tudo dentro "
        "do ruído) — manter ITILDE-Q ali preserva a narrativa sem custo "
        "estatístico;</li>"
        "<li><strong>Decisões abertas</strong>: a micro-grade (α≤0,05) pode "
        "achar um ponto em que a ITILDE-Q ≥ MSE pura; e o pareado "
        "MSE×ITILDE-Q exige re-run com predições salvas.</li>"
        "</ul></div>", sec_id="leitura"))

    return "".join(parts)


REPORT = {
    "title": "ITILDE-Q — busca de coeficientes e perdas da LSTM",
    "description": "Busca OFAT dos coeficientes (α, β, γ) da ITILDE-Q no "
                   "Kaggle, A/B com DM pareado contra hweight e comparação "
                   "com baselines — evidência para a decisão de loss do "
                   "artigo.",
    "regenerate": None,
    "retrain": None,
    "source_dir": "data/results/itildeq_terms_search",
    "info": [
        ("Grade", "OFAT α/β/γ (micro em execução)"),
        ("Protocolo", "80/20, 60 épocas, seed 42, HPs canônicos"),
        ("Pareado", "A/B lossab (hweight × ITILDE-Q, n=1203)"),
        ("Coleta", "scripts/itildeq_terms_kaggle.py collect [--micro]"),
    ],
}

if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
