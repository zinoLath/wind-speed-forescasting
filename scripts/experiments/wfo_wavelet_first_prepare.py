"""Experimento: denoising ANTES da imputacao (ordem invertida).

Protocolo atual: LightGBM imputa a serie crua; a wavelet (sym18, nivel 2)
aplica-se DEPOIS, por janela, dentro do treino. Aqui: (1) preenche-se os gaps
com andaime linear apenas para a wavelet ser computavel; (2) denoise do ws100;
(3) os gaps sao REABERTOS em NaN; (4) o LightGBM imputa no dominio denoised.

Saida: data/wfo_lightgbm_wavelet_first/series_imputed.csv — identica a
canonical, mas com a coluna extra ws100_wavelet (alvo pre-imputado). O runner
pula o denoise quando a coluna ja existe (prepare_data).

Usage:
    python scripts/experiments/wfo_wavelet_first_prepare.py
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
from src.impute.lightgbm import impute_dataframe
from src.utils import wavelet_denoising
import wfo_lightgbm_common as proto
import wfo_lightgbm_prepare as prepare

PARAMS = {"n_estimators": 100, "max_iter": 10, "random_state": 42, "train_sample": 3000}
OUT_DIR = common.resolve("data/wfo_lightgbm_wavelet_first")
OUT = OUT_DIR / "series_imputed.csv"


def main():
    print("Reconstruindo a serie combinada crua (com NaNs)...")
    lidar = prepare.load_lidar(common.resolve("data/dataset.csv"))
    sodar = prepare.load_sodar(common.resolve("data/wind_data.csv"))
    full_index = pd.date_range(lidar.index.min(), sodar.index.max(), freq="10min")
    raw = sodar.reindex(full_index).combine_first(
        lidar.reindex(full_index)
    )[list(proto.SERIES_COLUMNS)]
    flags_ws = raw["ws100"].isna()
    print(f"  {raw.shape} | gaps ws100: {int(flags_ws.sum())} celulas "
          f"({flags_ws.mean():.1%})")

    print("Denoising ANTES da imputacao (andaime linear nos gaps)...")
    started = time.perf_counter()
    scaffold = raw.interpolate(limit_direction="both")
    ws_denoised = pd.Series(
        wavelet_denoising(scaffold["ws100"].to_numpy(), level=2), index=raw.index
    )
    ws_denoised[flags_ws] = pd.NA  # reabre os gaps no dominio denoised
    print(f"  denoise em {time.perf_counter() - started:.1f}s")

    frame = raw.copy()
    frame["ws100"] = ws_denoised
    print(f"Imputando no dominio denoised {PARAMS}...")
    imputed = impute_dataframe(frame, **PARAMS)
    assert imputed.notna().all().all()

    # serie final: ws100/demais colunas do dominio CRU (identicas a canonical)
    # + ws100_wavelet do dominio denoised-imputado (alvo)
    final = pd.read_csv(proto.SERIES_CSV, index_col="timestamp", parse_dates=True)
    final["ws100_wavelet"] = imputed["ws100"].reindex(final.index)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT, index_label="timestamp")
    (OUT_DIR / "imputed_flags.csv").write_text(
        proto.FLAGS_CSV.read_text()
    )

    delta = (final["ws100_wavelet"] - final["ws100"]).abs()
    print(f"\nSerie: {OUT}")
    print(f"|wavelet_first - cru| no ws100: media {delta.mean():.3f} m/s, "
          f"p95 {delta.quantile(0.95):.3f} m/s")


if __name__ == "__main__":
    main()
