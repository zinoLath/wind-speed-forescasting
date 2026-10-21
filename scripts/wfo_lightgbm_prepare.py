"""Preparacao do experimento walk-forward com imputacao LightGBM.

1. Concatena o periodo LiDAR (data/dataset.csv, set-nov/2021, restrito as
   alturas 40-140 m) com o SODAR (data/wind_data.csv, nov/2021-out/2022) numa
   unica serie de 10 minutos — a "serie de imputacao do ano inteiro".
2. Imputa com LightGBM (IterativeImputer), guardando flag booleana POR CELULA
   (nao apenas por linha) para a analise de erro condicionada a imputacao.
3. Valida a imputacao com injecao de gaps artificiais (erro por coluna e por
   comprimento de gap) usando o protocolo de src/impute/base.validate().
4. Audita a qualidade: maiores vazios, trechos imputados constantes (artefato
   conhecido de imputadores iterativos) e cobertura mensal.

Saidas em data/wfo_lightgbm/: series_imputed.csv, imputed_flags.csv,
series_meta.json e imputation_validation/ (detailed/summary/plots).

Usage:
    python scripts/wfo_lightgbm_prepare.py [--skip-imputation]
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src import common
from src.impute import base
from src.impute.lightgbm import impute_dataframe
import wfo_lightgbm_common as proto

IMPUTE_PARAMS = {"n_estimators": 100, "max_iter": 10, "random_state": 42, "train_sample": 3000}
VALIDATION_PARAMS = {"n_samples": 30, "gap_lengths": (1, 2, 3, 5, 10, 20, 36), "seed": 42}


def load_lidar(path):
    """LiDAR bruto (com NaNs!) restrito as colunas canonicas 40-140 m.

    Os timestamps do LiDAR carregam microssegundos de jitter (ex. .000004);
    arredondar para a grade de 10 minutos antes do reindex — sem isso o
    reindex descartaria ~89% das linhas.
    """
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["id"], format="mixed").dt.round("10min")
    df = common.canonicalize_columns(df)
    df = df[["timestamp", *proto.SERIES_COLUMNS]]
    df = df.set_index("timestamp")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.groupby(level=0).mean().sort_index()
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="10min")
    return df.reindex(full_index)


def load_sodar(path):
    """SODAR bruto (com NaNs) via o loader padrao do pacote de imputacao."""
    df = base.load_wind_data(path)
    return df[[c for c in proto.SERIES_COLUMNS if c in df.columns]]


def largest_gaps(index, mask, top=10):
    """Maiores sequencias continuas de linhas True (mask booleano alinhado)."""
    gaps = []
    run_start = None
    for i, flag in enumerate(mask):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            gaps.append((run_start, i - run_start))
            run_start = None
    if run_start is not None:
        gaps.append((run_start, len(mask) - run_start))
    gaps.sort(key=lambda g: -g[1])
    return [
        {
            "start": str(index[s]),
            "end": str(index[min(s + length, len(index) - 1)]),
            "rows": int(length),
            "hours": round(length / 6.0, 2),
        }
        for s, length in gaps[:top]
    ]


def longest_constant_runs(series, mask, top=5, min_run=36):
    """Maiores trechos com valor identico dentro das linhas imputadas de *series*.

    Detecta o artefato documentado em docs/wfo_imputed_detailed_study.md
    (imputacao constante por milhares de linhas).
    """
    values = series.to_numpy(dtype=float)
    flags = mask.to_numpy(dtype=bool)
    runs = []
    run_start = None
    run_value = None
    for i in range(len(values)):
        if not flags[i]:
            run_start, run_value = None, None
            continue
        value = values[i]
        if run_start is not None and value == run_value:
            continue
        if run_start is not None and i - run_start >= min_run:
            runs.append((run_start, i - run_start, run_value))
        run_start, run_value = i, value
    if run_start is not None and len(values) - run_start >= min_run:
        runs.append((run_start, len(values) - run_start, run_value))
    runs.sort(key=lambda r: -r[1])
    return [
        {
            "start": str(series.index[s]),
            "rows": int(length),
            "value": float(value),
        }
        for s, length, value in runs[:top]
    ]


def audit(raw, imputed, flags, source):
    """Estatisticas de cobertura/qualidade da serie combinada."""
    row_flag = flags.any(axis=1)
    monthly = pd.DataFrame({
        "imputed_frac": row_flag.groupby(raw.index.to_period("M")).mean(),
        "rows": row_flag.groupby(raw.index.to_period("M")).size(),
    })
    constant_runs = {}
    for col in proto.SERIES_COLUMNS:
        runs = longest_constant_runs(imputed[col], flags[col])
        if runs:
            constant_runs[col] = runs
    meta = {
        "period_start": str(raw.index.min()),
        "period_end": str(raw.index.max()),
        "rows": int(len(raw)),
        "columns": list(raw.columns),
        "source_counts": source.value_counts().rename(index=str).to_dict(),
        "missing_cells_total": int(flags.to_numpy().sum()),
        "missing_cells_frac": float(flags.to_numpy().mean()),
        "rows_with_imputation": int(row_flag.sum()),
        "monthly_coverage": {
            str(period): {
                "imputed_frac": round(float(row["imputed_frac"]), 4),
                "rows": int(row["rows"]),
            }
            for period, row in monthly.iterrows()
        },
        "imputed_frac_by_column": {
            col: round(float(frac), 4) for col, frac in flags.mean().items()
        },
        "largest_all_nan_gaps": largest_gaps(raw.index, raw.isna().all(axis=1).to_numpy()),
        "largest_missing_runs": {
            col: largest_gaps(raw.index, flags[col].to_numpy(), top=3)
            for col in ("ws40", "ws100", "ws140")
        },
        "constant_runs_imputed": constant_runs,
    }
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-imputation", action="store_true",
                        help="usa series_imputed.csv existente (so validacao/auditoria)")
    args = parser.parse_args()

    proto.DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Carregando LiDAR (data/dataset.csv)...")
    lidar = load_lidar(common.resolve("data/dataset.csv"))
    print(f"  {lidar.shape} | {lidar.index.min()} -> {lidar.index.max()}")
    print("Carregando SODAR (data/wind_data.csv)...")
    sodar = load_sodar(common.resolve("data/wind_data.csv"))
    print(f"  {sodar.shape} | {sodar.index.min()} -> {sodar.index.max()}")

    full_index = pd.date_range(lidar.index.min(), sodar.index.max(), freq="10min")
    lidar = lidar.reindex(full_index)
    sodar = sodar.reindex(full_index)
    raw = sodar.combine_first(lidar)[list(proto.SERIES_COLUMNS)]

    source = pd.Series("gap", index=full_index, name="source")
    source[lidar.notna().any(axis=1)] = "lidar"
    source[sodar.notna().any(axis=1)] = "sodar"
    print(f"Serie combinada: {raw.shape} | fontes: {source.value_counts().to_dict()}")
    print(f"Celulas ausentes antes da imputacao: {int(raw.isna().sum().sum())}")

    flags = raw.isna()
    flags.to_csv(proto.FLAGS_CSV, index_label="timestamp")

    if args.skip_imputation and proto.SERIES_CSV.is_file():
        imputed = pd.read_csv(proto.SERIES_CSV, index_col="timestamp", parse_dates=True)
        print("Imputacao pulada (--skip-imputation); series_imputed.csv existente.")
    else:
        print(f"Imputando com LightGBM {IMPUTE_PARAMS}...")
        imputed = impute_dataframe(raw, **IMPUTE_PARAMS)
        imputed.to_csv(proto.SERIES_CSV, index_label="timestamp")
        resid = int(imputed.isna().sum().sum())
        print(f"Imputacao concluida | NaNs remanescentes: {resid}")

    print("Auditoria da serie...")
    meta = audit(raw, imputed, flags, source)
    meta["imputation_params"] = IMPUTE_PARAMS
    common.write_json(proto.META_JSON, meta)
    worst_constants = sorted(
        (
            (col, run["rows"])
            for col, runs in meta["constant_runs_imputed"].items()
            for run in runs
        ),
        key=lambda item: -item[1],
    )[:5]
    for col, rows in worst_constants:
        print(f"  AVISO: {col} tem trecho imputado constante de {rows} linhas")
    print(f"  maior vazio total: {meta['largest_all_nan_gaps'][0] if meta['largest_all_nan_gaps'] else 'n/a'}")

    if not proto.VALIDATION_DIR.is_dir() or not (proto.VALIDATION_DIR / "summary.csv").is_file():
        print("Validando imputacao com gaps artificiais (por coluna)...")
        detailed, summary, overall, settings = base.validate(
            impute_dataframe,
            "lightgbm",
            raw,
            imputer_kwargs=dict(IMPUTE_PARAMS),
            **VALIDATION_PARAMS,
        )
        base.save_report(proto.VALIDATION_DIR, settings, detailed, summary, overall)
        base.plot_metrics(summary, proto.VALIDATION_DIR)
        print(
            f"Validacao: MAE={overall['mae']:.4f} RMSE={overall['rmse']:.4f} "
            f"(n={overall['samples']})"
        )
    else:
        print("Validacao existente em imputation_validation/ (apague para refazer).")

    print("Plano de janelas walk-forward:")
    windows = proto.plan_windows(raw.index)
    for window in windows:
        print(
            f"  teste {window['window']}: treino {window['train_start']:%Y-%m-%d} -> "
            f"{window['train_end'] - pd.Timedelta(minutes=10):%Y-%m-%d} "
            f"({window['train_rows']} linhas, cobertura {window['train_coverage']:.0%}) | "
            f"teste {window['test_rows']} linhas"
        )
    common.write_json(
        proto.DATA_DIR / "windows_plan.json",
        [{**w, "train_start": str(w["train_start"]), "train_end": str(w["train_end"]),
          "test_start": str(w["test_start"]), "test_end": str(w["test_end"])}
         for w in windows],
    )
    print(f"Pronto. Artefatos em {proto.DATA_DIR}")


if __name__ == "__main__":
    main()
