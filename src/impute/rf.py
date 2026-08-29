"""Random Forest imputation (per-row IterativeImputer)."""

from sklearn.ensemble import RandomForestRegressor

from src.impute import base


def impute_dataframe(
    df,
    n_estimators=100,
    max_iter=10,
    random_state=42,
    train_sample=3000,
):
    """Fill missing values per row with IterativeImputer + RandomForestRegressor."""
    estimator = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    imputed = base.impute_with_estimator(
        df,
        estimator,
        max_iter=max_iter,
        random_state=random_state,
        train_sample=train_sample,
    )
    return base.fill_remaining_nan(imputed)