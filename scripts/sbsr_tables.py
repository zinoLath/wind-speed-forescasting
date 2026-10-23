"""Gera as tabelas LaTeX do artigo SBSR (tab:error_all / tab:error_h36)
a partir do evaluation.json de uma rodada + métricas dos baselines.

As tabelas já saem com caption e \\label configurados (formato steam, sbsr
ou generic, via reports/src/latexfmt.py), ordenadas por R² decrescente e
com separador decimal no padrão do template (vírgula para sbsr/steam).

Fontes:
- <round-dir>/evaluation.json  -> modelos seq2seq ("all" = agregado; h=36
  de "per_horizon", recomputado dos pesos quando ausente e persistido de
  volta — mesma convenção do relatorio_resultados).
- data/results/baselines/arima_rf/metrics.json -> ARIMA/RF/persistência
  ("all_horizons" e "rolling" = h=36).

Exemplos:
    python scripts/sbsr_tables.py                       # imprime sbsr
    python scripts/sbsr_tables.py --format steam
    python scripts/sbsr_tables.py --out reports/lidar/8020/tabelas_sbsr.tex
    python scripts/sbsr_tables.py --no-baselines --decimal point
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src import latexfmt
from src import common

ROUND_DIR = PROJECT_ROOT / "data" / "results" / "report_8020"
BASELINES_JSON = PROJECT_ROOT / "data" / "results" / "baselines" / "arima_rf" / "metrics.json"

COLUMNS = ("Método", "R²", "MSE", "MAE", "RMSE")
DEFAULTS = {
    "all": {"label": "tab:error_all",
            "caption": "Métricas de erro observadas."},
    "h36": {"label": "tab:error_h36",
            "caption": "Métricas de erro no último horizonte de predição."},
}
NAME_OVERRIDES = {
    "ARIMA (baseline)": "ARIMA",
    "Random Forest (baseline)": "Random Forest",
    "Persistência": "Persistência",
}

BASELINE_KEYS = ("arima", "rf", "persistence")


def display_name(name: str) -> str:
    return NAME_OVERRIDES.get(name, name.replace(" Bidirectional", " - Bi"))


def fmt_num(x, nd: int, comma: bool) -> str:
    s = f"{x:.{nd}f}"
    return s.replace(".", ",") if comma else s


def load_rr():
    path = PROJECT_ROOT / "tests" / "relatorio_resultados.py"
    spec = importlib.util.spec_from_file_location("relatorio_resultados", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["relatorio_resultados"] = mod
    spec.loader.exec_module(mod)
    return mod


def h36_from_eval(eval_entry: dict):
    for rec in eval_entry.get("per_horizon", []):
        if int(rec["horizon"]) == 36:
            return rec
    return None


def h36_recompute(eval_json: dict, models_dir: Path, persist: bool) -> dict:
    rr = load_rr()
    tf = common.setup_tensorflow(True)
    print(f"TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")
    dataset = common.load_dataset(rr.DATA_FILE, keep_raw=("ws40", "ws100", "v40"))
    observed = common.observed_timestamps(rr.DATA_FILE)
    models = rr.load_metadata(models_dir)
    out = {}
    for entry in models:
        name = entry["meta"]["name"]
        if name not in eval_json:
            continue
        meta = entry["meta"]
        ratios = meta.get("dataset", {})
        train_ratio = ratios.get("train_ratio", 0.8)
        val_ratio = ratios.get("val_ratio", 0.0)
        train_df, val_df, _ = common.split_dataset(dataset, train_ratio, val_ratio)
        val_end = len(train_df) + len(val_df)
        val_for_prepare = val_df if len(val_df) else train_df.tail(400)
        print(f"[{meta['wrapper_key']}] recomputando per-horizonte de "
              f"{entry['dir'].name} ...")
        wrapper = rr.build_wrapper(meta, train_df, val_for_prepare,
                                   entry["dir"] / "model.keras")
        eval_df = dataset.iloc[max(0, val_end - wrapper.input_steps):]
        all_df = common.predict_all_horizons(wrapper, eval_df)
        all_df["actual_raw"] = dataset["ws100"].reindex(all_df["timestamp"]).to_numpy()
        if observed is not None:
            all_df = all_df[all_df["timestamp"].isin(observed)]
        per_h = common.horizon_metrics(all_df.assign(actual=all_df["actual_raw"]))
        per_h_den = common.horizon_metrics(all_df.assign(actual=all_df["actual"]))
        out[name] = per_h[per_h["horizon"] == 36].iloc[0].to_dict()
        if persist:
            eval_json[name]["per_horizon"] = per_h.to_dict("records")
            eval_json[name]["per_horizon_den"] = per_h_den.to_dict("records")
    return out


def build_tables(args) -> str:
    round_dir = Path(args.round_dir)
    eval_path = round_dir / "evaluation.json"
    if not eval_path.is_file():
        sys.exit(f"evaluation.json não encontrado: {eval_path}")
    eval_json = json.loads(eval_path.read_text(encoding="utf-8"))

    comma = (args.decimal == "comma") if args.decimal != "auto" \
        else args.format in ("sbsr", "steam")

    all_rows, h36_rows = [], []

    for name, entry in eval_json.items():
        if "all" not in entry:
            continue
        agg = entry["all"]
        all_rows.append((display_name(entry.get("short", name)), agg))
        h = h36_from_eval(entry)
        if h is None and not args.skip_recompute:
            h = h36_recompute({name: entry}, models_dir_for(round_dir), args.persist).get(name)
            if h is not None and args.persist:
                common.write_json(eval_path, eval_json)
                print(f"[{name}] per_horizon persistido em {eval_path}")
        if h is None:
            sys.exit(f"sem métricas h=36 para {name}; rode sem --skip-recompute "
                     "(precisa de TF + pesos em --models-dir)")
        h36_rows.append((display_name(entry.get("short", name)), h))

    if not args.no_baselines:
        base = None
        if BASELINES_JSON.is_file():
            base = json.loads(BASELINES_JSON.read_text(encoding="utf-8"))
        if getattr(args, "trim_boundary_origins", 0):
            base = trimmed_baseline_metrics(base, args.trim_boundary_origins)
        if base:
            for key in BASELINE_KEYS:
                blk = base.get(key)
                if not blk:
                    continue
                label = {"arima": "ARIMA", "rf": "Random Forest",
                         "persistence": "Persistência"}[key]
                all_rows.append((label, blk["all_horizons"]))
                h36_rows.append((label, blk["rolling"]))
        else:
            print(f"aviso: baselines ausentes ({BASELINES_JSON}); use --no-baselines "
                  "para silenciar", file=sys.stderr)

    def to_cells(rows):
        return [(nome, fmt_num(m["r2"], args.nd, comma),
                 fmt_num(m["mse"], args.nd, comma),
                 fmt_num(m["mae"], args.nd, comma),
                 fmt_num(m["rmse"], args.nd, comma))
                for nome, m in sorted(rows, key=lambda r: r[1]["r2"], reverse=True)]

    body_all = latexfmt.tabular_body(COLUMNS, to_cells(all_rows), args.format,
                                     align="c" * len(COLUMNS))
    body_h36 = latexfmt.tabular_body(COLUMNS, to_cells(h36_rows), args.format,
                                     align="c" * len(COLUMNS))
    table_all = latexfmt.render_table(
        args.format, caption=args.caption_all, label=args.label_all,
        body=body_all, placement=args.placement)
    table_h36 = latexfmt.render_table(
        args.format, caption=args.caption_h36, label=args.label_h36,
        body=body_h36, placement=args.placement)

    header = (f"% Gerado por scripts/sbsr_tables.py em "
              f"{datetime.date.today().isoformat()} — rodada: "
              f"{round_dir.resolve().relative_to(PROJECT_ROOT)}/evaluation.json"
              + ("" if args.no_baselines else " + baselines (arima_rf/metrics.json")
              + (f"; -{args.trim_boundary_origins} origens de fronteira)"
                 if getattr(args, "trim_boundary_origins", 0) else
                 (")" if not args.no_baselines else "")))
    return "\n\n".join([header, table_all, table_h36]) + "\n"


def trimmed_baseline_metrics(base: dict | None, n: int) -> dict:
    """Baselines recomputados na grade sem as primeiras *n* origens.

    Espelha o corte de ``tests/relatorio_resultados.py``: descarta as
    primeiras *n* origens do ``predictions_long.csv`` (janela de entrada
    cruza a fronteira treino/teste) e recalcula as métricas para todos os
    métodos ficarem na mesma grade.
    """
    csv_path = BASELINES_JSON.parent / "predictions_long.csv"
    if not csv_path.is_file():
        sys.exit(f"predictions_long.csv ausente ({csv_path}); impossível "
                 "aplicar --trim-boundary-origins")
    df = pd.read_csv(csv_path, parse_dates=["origin", "timestamp"])
    for col in ("origin", "timestamp"):
        df[col] = df[col].dt.round("min")
    origins = sorted(df["origin"].unique())
    if len(origins) <= n:
        sys.exit(f"grade tem {len(origins)} origens <= n={n}")
    cut = origins[n - 1]
    df = df[df["origin"] > cut]
    print(f"[trim-boundary] baselines: {len(origins)} -> "
          f"{df['origin'].nunique()} origens (descartadas as primeiras "
          f"{n}, ate {cut})")
    out = json.loads(json.dumps(base)) if base else {}
    for key in BASELINE_KEYS:
        if key not in df.columns:
            continue
        m_all = common.compute_metrics(df["actual"], df[key])
        h_max = int(df["horizon"].max())
        roll = df[df["horizon"] == h_max]
        m_roll = common.compute_metrics(roll["actual"], roll[key])
        blk = out.get(key, {})
        blk["all_horizons"] = m_all
        blk["rolling"] = m_roll
        out[key] = blk
    return out


def models_dir_for(round_dir: Path) -> Path:
    suffix = round_dir.name.replace("report_", "")
    cand = PROJECT_ROOT / "pipeline" / "tmp" / f"trained_{suffix}"
    return cand if cand.is_dir() else PROJECT_ROOT / "pipeline" / "tmp" / "trained_8020"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round-dir", default=str(ROUND_DIR),
                    help="pasta da rodada com evaluation.json (default: report_8020)")
    ap.add_argument("--format", choices=("sbsr", "steam", "generic"), default="sbsr")
    ap.add_argument("--decimal", choices=("comma", "point", "auto"), default="auto",
                    help="vírgula é o padrão dos templates sbsr/steam")
    ap.add_argument("--nd", type=int, default=3, help="casas decimais (default: 3)")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--trim-boundary-origins", type=int, default=0,
                    help="descarta as primeiras N origens do teste e "
                         "recomputa os baselines na grade cortada "
                         "(espelha tests/relatorio_resultados.py)")
    ap.add_argument("--placement", default="h")
    ap.add_argument("--label-all", default=DEFAULTS["all"]["label"])
    ap.add_argument("--caption-all", default=DEFAULTS["all"]["caption"])
    ap.add_argument("--label-h36", default=DEFAULTS["h36"]["label"])
    ap.add_argument("--caption-h36", default=DEFAULTS["h36"]["caption"])
    ap.add_argument("--skip-recompute", action="store_true",
                    help="usa apenas per_horizon já persistido no evaluation.json")
    ap.add_argument("--persist", action="store_true", default=True,
                    help="persiste per_horizon recomputado no evaluation.json")
    ap.add_argument("--no-persist", dest="persist", action="store_false")
    ap.add_argument("--out", default=None, help="arquivo .tex de saída (default: stdout)")
    args = ap.parse_args()

    tables = build_tables(args)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(tables, encoding="utf-8")
        print(f"escrito: {out}")
    else:
        print(tables)


if __name__ == "__main__":
    main()
