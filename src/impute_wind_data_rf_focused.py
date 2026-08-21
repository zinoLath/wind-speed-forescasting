"""Generate focused Random Forest-imputed wind data using cached Optuna parameters."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.imputation_model_utils import generate_imputed_csv


if __name__ == "__main__":
    generate_imputed_csv(
        model="random_forest",
        description="Impute focused wind columns with the Optuna-tuned Random Forest.",
        default_output=PROJECT_ROOT / "data" / "wind_data_imputed_rf_focused.csv",
    )
