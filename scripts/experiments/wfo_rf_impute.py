"""Imputacao Random Forest da mesma serie combinada (LiDAR+SODAR) do WFO.

Gera data/wfo_lightgbm/series_imputed_rf.csv com o MESMO protocolo do
LightGBM (mesmo raw, mesmos parametros do pipeline: 100 estimators, 10
iteracoes, seed 42, train_sample 3000) para comparacao visual lado a lado
nos mesmos gaps.

Usage:
    python scripts/experiments/wfo_rf_impute.py
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pandas as pd

from src import common
from src.impute.rf import impute_dataframe
import wfo_lightgbm_common as proto
import wfo_lightgbm_prepare as prepare

PARAMS = {"n_estimators": 100, "max_iter": 10, "random_state": 42, "train_sample": 3000}
OUT = proto.DATA_DIR / "series_imputed_rf.csv"


def main():
    print("Reconstruindo a serie combinada (LiDAR 40-140 m + SODAR)...")
    lidar = prepare.load_lidar(common.resolve("data/dataset.csv"))
    sodar = prepare.load_sodar(common.resolve("data/wind_data.csv"))
    full_index = pd.date_range(lidar.index.min(), sodar.index.max(), freq="10min")
    raw = sodar.reindex(full_index).combine_first(
        lidar.reindex(full_index)
    )[list(proto.SERIES_COLUMNS)]
    print(f"  {raw.shape} | celulas ausentes: {int(raw.isna().sum().sum())}")

    if OUT.is_file():
        print(f"Imputacao RF ja existe: {OUT}")
        imputed = pd.read_csv(OUT, index_col="timestamp", parse_dates=True)
    else:
        print(f"Imputando com Random Forest {PARAMS} (n_jobs=-1)...")
        started = time.perf_counter()
        imputed = impute_dataframe(raw, **PARAMS)
        elapsed = time.perf_counter() - started
        assert imputed.notna().all().all(), "NaNs remanescentes"
        imputed.to_csv(OUT, index_label="timestamp")
        print(f"Concluido em {elapsed / 60:.1f} min -> {OUT}")

    # diferenca global vs LightGBM no ws100 imputado
    flags = pd.read_csv(proto.FLAGS_CSV, index_col="timestamp", parse_dates=True, dtype=bool)
    lgbm = pd.read_csv(proto.SERIES_CSV, index_col="timestamp", parse_dates=True)
    delta = (imputed.loc[flags["ws100"], "ws100"]
             - lgbm.loc[flags["ws100"], "ws100"]).abs()
    print(f"|LGBM - RF| no ws100 imputado: media {delta.mean():.3f} m/s, "
          f"p95 {delta.quantile(0.95):.3f} m/s")


if __name__ == "__main__":
    main()
