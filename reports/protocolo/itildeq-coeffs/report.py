"""Relatório — coeficientes ITILDE-Q via Optuna (protocolo round 5).

Populado automaticamente pelo timer (scripts/dist/cron_itildeq_coeffs_report.sh):
lê ``data/results/itildeq_coeffs_optuna/live_<wrapper>.json`` (progresso
exportado do sqlite) e, quando o estudo termina, o resultado final
``<wrapper>.json`` (teste + painel DTW).

    python reports/protocolo/itildeq-coeffs/report.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src import components as C  # noqa: E402

DATA = PROJECT_ROOT / "data" / "results" / "itildeq_coeffs_optuna"


def load(name):
    p = DATA / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def wrapper_section(wrapper, live, final):
    label = "LSTM" if wrapper == "lstm" else "LSTM-Bi"
    best = (live or {}).get("best") or {}
    parts = [C.section(
        f"{label} — estudo",
        C.cards([
            ((live or {}).get("n_complete", 0), "trials completos"),
            (f"{(best.get('val_block_mse') or 0):.4f}", "melhor val_block_mse"),
            (f"α={best.get('alpha', 0):.4f}", "melhor α"),
            (f"β={best.get('beta', 0):.3f} · γ={best.get('gamma', 0):.4f}",
             "melhor β · γ"),
        ]), sec_id=f"{wrapper}-cards")]

    trials = (live or {}).get("trials") or []
    if trials:
        rows = [(str(t["trial"]), f"{t['alpha']:.4f}", f"{t['beta']:.3f}",
                 f"{t['gamma']:.4f}", f"{t['val_block_mse']:.4f}",
                 f"{(t.get('elapsed_sec') or 0):.0f}s") for t in trials]
        parts.append(C.section(
            f"{label} — trials (ordenado pelo objetivo)",
            C.table(["trial", "α", "β", "γ", "val_block_mse", "tempo"],
                    rows, "Objetivo: MSE em 4 blocos de horizonte na "
                    "validação (split 72/8/20, HPs da round 5).",
                    label=f"tab:itildeq_coeffs_{wrapper}"),
            sec_id=f"{wrapper}-trials"))

    if final and final.get("test"):
        t = final["test"]
        parts.append(C.section(
            f"{label} — melhor no teste + comportamento",
            C.table(["R²", "MSE", "viés", "std_ratio", "HF", "ac1", "corr",
                     "DTW", "lag"],
                    [("teste", f"{t['r2']:.4f}", f"{t['mse']:.4f}",
                      f"{t['bias']:.4f}", f"{t['std_ratio']:.3f}",
                      f"{t['hf_ratio_pred']:.6f}", f"{t['ac1_pred']:.3f}",
                      f"{t['corr_mean']:.3f}", f"{t['dtw_mean']:.3f}",
                      f"{t['lag_mean_steps']:.0f}")],
                    "Reavaliação do melhor trial no teste (20% intocado) + "
                    "painel de comportamento das predições.",
                    label=f"tab:itildeq_coeffs_test_{wrapper}"),
            sub="DTW/comportamento conforme scripts/prediction_diagnostics.py.",
            sec_id=f"{wrapper}-test"))
    return "".join(parts)


def build(ctx):  # noqa: ARG001
    lives = {w: load(f"live_{w}.json") for w in ("lstm", "lstm_bi")}
    finals = {w: load(f"{w}.json") for w in ("lstm", "lstm_bi")}
    if not any(lives.values()) and not any(finals.values()):
        return C.page_header(
            "ITILDE-Q — coeficientes via Optuna",
            "Sem dados ainda.") + C.note(
            "Rode <code>python scripts/experiments/itildeq_coeffs_optuna.py</code>; o "
            "timer popula esta página a cada 10 min.")

    status = "concluído" if all(finals.values()) else "em andamento"
    parts = [C.page_header(
        "ITILDE-Q — coeficientes via Optuna (protocolo round 5)",
        "Busca de (α, β, γ) da perda ITILDE-Q sob o protocolo da Optuna "
        "round 5: split 72/8/20, HPs promovidos da round 5, 60 épocas, "
        "seed 42; objetivo = MSE em 4 blocos na validação. Página "
        "populada automaticamente pelo timer.",
        badges=("72/8/20", "val block MSE k=4", "HPs round 5", status)),
        C.cards([
            (sum((l or {}).get("n_complete", 0) for l in lives.values()),
             "trials completos"),
            ("2", "wrappers (lstm, lstm_bi)"),
            (status, "status"),
            ("10 min", "atualização (timer)"),
        ])]

    parts.append(C.section(
        "Protocolo",
        "<div class='markdown'><p><strong>Split 72/8/20</strong>: teste de "
        "20% intocado; validação de 8% usada como objetivo (mesma família "
        "do <code>val_block_mse(k=4)</code> da round 5). "
        "<strong>HPs</strong>: vencedores promovidos da round 5 "
        "(<code>pipeline/tmp/optuna/</code>). <strong>Espaço</strong>: "
        "α ∈ [1e-3, 0,5] (log), β ∈ [0,2; 0,8], γ ∈ [1e-3, 0,5] (log). "
        "<strong>Ressalva</strong>: 1 seed por trial (42); o melhor é "
        "reavaliado no teste com o painel de diagnóstico.</p></div>",
        sub="Perda: MSE + α·(β·shift_amplitude + (1−β)·fase + γ·amplificação) "
            "— Lee et al. 2022 (TILDE-Q) / Dong Sheng 2026 (Eq. 15).",
        sec_id="protocolo"))

    for wrapper in ("lstm", "lstm_bi"):
        parts.append(wrapper_section(wrapper, lives[wrapper], finals[wrapper]))
    return "".join(parts)


REPORT = {
    "title": "ITILDE-Q — coeficientes via Optuna (round 5)",
    "description": "Busca Optuna de (α, β, γ) da ITILDE-Q sob o protocolo "
                   "da round 5 (72/8/20, objetivo val em blocos); populada "
                   "automaticamente pelo timer a cada 10 min.",
    "regenerate": [".venv/bin/python", "scripts/experiments/itildeq_coeffs_export.py"],
    "retrain": None,
    "source_dir": "data/results/itildeq_coeffs_optuna",
    "info": [
        ("Protocolo", "72/8/20, 60 épocas, seed 42"),
        ("Objetivo", "val_block_mse(k=4)"),
        ("HPs", "promovidos da Optuna round 5"),
        ("Atualização", "timer a cada 10 min (systemd user)"),
    ],
}

if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
