"""KNN imputation (per-row KNNImputer)."""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from src.impute import base


def impute_dataframe(df, n_neighbors=5, weights="distance"):
    """Fill missing values per row with KNNImputer (sin/cos direction handling)."""
    direction_cols = [c for c in df.columns if base.is_direction_column(c)]
    transformed = base.direction_to_components(df, direction_cols)
    numeric_cols = transformed.select_dtypes(include=[np.number]).columns.tolist()

    values = KNNImputer(n_neighbors=n_neighbors, weights=weights).fit_transform(
        transformed[numeric_cols]
    )
    imputed = pd.DataFrame(values, index=transformed.index, columns=numeric_cols)
    for col in transformed.columns:
        if col not in imputed.columns:
            imputed[col] = transformed[col]

    imputed = base.components_to_direction(imputed, direction_cols)
    return base.fill_remaining_nan(imputed)
