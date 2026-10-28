"""Tabela de métricas no h = 36 (6 h) — modelos DL + baselines, reval SBSR.

DL: predições por origem do banco (run 7, grade já com trim 72). Baselines
(ARIMA/RF/persistência): predições long armazenadas
(data/results/baselines/arima_rf/predictions_long.csv, grade 80/20 que
CONTÉM a janela de teste de 15% da revalidação) — filtradas ao conjunto
de origens comum (h36). Saída: tabela_h36.json.

    python reports/lidar/reval_sbsr/tabela_h36.py
"""

import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src import common  # noqa: E402

OUT_JSON = (PROJECT_ROOT / "data" / "results" / "report_reval_sbsr"
            / "tabela_h36.json")
BASELINE_PREDS = (PROJECT_ROOT / "data" / "results" / "baselines"
                  / "arima_rf_reval" / "predictions_long.csv")
RUN_ID = 7
H = 36


def main():
    con = sqlite3.connect(PROJECT_ROOT / "data" / "predictions.db")
    dl = pd.read_sql_query(
        """SELECT m.name, p.origin, p.timestamp, p.horizon,
                  p.actual, p.predicted
           FROM run_model_prediction p JOIN run_model m ON m.id = p.model_id
           WHERE m.run_id = ? AND p.horizon = ?""",
        con, params=(RUN_ID, H))
    dl["origin"] = pd.to_datetime(dl["origin"]).dt.round("min")
    origins = set(dl["origin"].unique())

    base = pd.read_csv(BASELINE_PREDS, parse_dates=["origin", "timestamp"])
    base["origin"] = base["origin"].dt.round("min")
    base = base[(base["horizon"] == H) & (base["origin"].isin(origins))]

    rows = []
    persistence_mse = None
    for name in ("Seq2Seq_LSTM", "Seq2Seq_TCN_Bidirectional", "Seq2Seq_TCN",
                 "Seq2Seq_LSTM_Bidirectional"):
        g = dl[dl["name"] == name]
        m = common.compute_metrics(g["actual"], g["predicted"])
        rows.append((name.replace("Seq2Seq_", ""), len(g), m))
    for key, label in (("arima", "ARIMA"), ("rf", "Random Forest"),
                       ("persistence", "Persistência")):
        m = common.compute_metrics(base["actual"], base[key])
        if key == "persistence":
            persistence_mse = m["mse"]
        rows.append((label, len(base), m))

    out = {"horizon": H, "n_origins": int(len(base)),
           "rows": [{"modelo": label, "n": n, **m,
                     "skill_persistence": (1 - m["mse"] / persistence_mse
                                           if persistence_mse else None)}
                    for label, n, m in rows]}
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"origens comuns no h{H}: {len(base)}")
    for r in out["rows"]:
        sk = r["skill_persistence"]
        print(f"{r['modelo']:<18} MSE {r['mse']:.4f}  RMSE {r['rmse']:.4f}  "
              f"MAE {r['mae']:.4f}  R2 {r['r2']:.4f}  skill {sk:+.3f}")
    print(f"-> {OUT_JSON}")


if __name__ == "__main__":
    main()
