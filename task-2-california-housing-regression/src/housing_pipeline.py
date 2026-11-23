"""California housing regression pipeline (OLS, Ridge, Lasso).

Loads the housing data, engineers ratio and interaction features, removes
redundant (multicollinear) columns, and compares OLS, Ridge and Lasso
regression on RMSE, MAE and R2.

Usage (from the task folder):
    python src/housing_pipeline.py --data data/housing.csv
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TARGET = "median_house_value"
CATEGORICAL_FEATURES = ["ocean_proximity"]
RAW_NUMERIC = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
REQUIRED_COLUMNS = RAW_NUMERIC + CATEGORICAL_FEATURES + [TARGET]

# Raw counts are almost perfectly correlated with each other (VIF far above 10);
# the per-household ratios created below carry the useful information instead.
REDUNDANT_COUNT_COLUMNS = ["total_rooms", "total_bedrooms", "population", "households"]
ENGINEERED_FEATURES = [
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
    "income_x_age",
    "income_x_rooms",
]
NUMERIC_FEATURES = [
    col for col in RAW_NUMERIC if col not in REDUNDANT_COUNT_COLUMNS
] + ENGINEERED_FEATURES

DEFAULT_RIDGE_ALPHAS = np.logspace(-3, 4, 29)
DEFAULT_LASSO_ALPHAS = np.logspace(-6, -1, 11)


@dataclass(frozen=True)
class FeatureStats:
    """Training-set means used to center the interaction features."""

    income_mean: float
    age_mean: float
    rooms_per_household_mean: float


def load_data(path: Path | str) -> pd.DataFrame:
    """Load the housing CSV, validate the schema and coerce numeric types.

    Rows without a target value are dropped; missing feature values are kept
    and imputed later inside the preprocessing pipeline.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if required columns are missing or no valid rows remain.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[REQUIRED_COLUMNS].copy()
    numeric_columns = RAW_NUMERIC + [TARGET]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    n_no_target = int(df[TARGET].isna().sum())
    if n_no_target:
        logger.warning("Dropping %d row(s) without a target value", n_no_target)
        df = df.dropna(subset=[TARGET])
    if df.empty:
        raise ValueError("No valid rows left after cleaning the data")

    logger.info(
        "Loaded %d rows from %s (%d missing feature values)",
        len(df),
        path,
        int(df[RAW_NUMERIC].isna().sum().sum()),
    )
    return df.reset_index(drop=True)


def compute_feature_stats(train: pd.DataFrame) -> FeatureStats:
    """Compute centering statistics from the training data only."""
    rooms_per_household = train["total_rooms"] / train["households"].replace(0, np.nan)
    return FeatureStats(
        income_mean=float(train["median_income"].mean()),
        age_mean=float(train["housing_median_age"].mean()),
        rooms_per_household_mean=float(rooms_per_household.mean()),
    )


def add_features(data: pd.DataFrame, stats: FeatureStats) -> pd.DataFrame:
    """Add ratio and (centered) interaction features.

    Zero denominators are treated as missing instead of producing infinities.
    """
    data = data.copy()
    households = data["households"].replace(0, np.nan)
    rooms = data["total_rooms"].replace(0, np.nan)

    data["rooms_per_household"] = data["total_rooms"] / households
    data["bedrooms_per_room"] = data["total_bedrooms"] / rooms
    data["population_per_household"] = data["population"] / households

    income_centered = data["median_income"] - stats.income_mean
    data["income_x_age"] = income_centered * (data["housing_median_age"] - stats.age_mean)
    data["income_x_rooms"] = income_centered * (
        data["rooms_per_household"] - stats.rooms_per_household_mean
    )
    return data


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split first, then engineer features with training-set statistics only."""
    features = df.drop(columns=TARGET)
    target = df[TARGET]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    stats = compute_feature_stats(X_train_raw)
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train = add_features(X_train_raw, stats)[columns]
    X_test = add_features(X_test_raw, stats)[columns]
    return X_train, X_test, y_train, y_test


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Median imputation + scaling for numbers, one-hot encoding for categories."""
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def _wrap(estimator: object, preprocessor: ColumnTransformer) -> TransformedTargetRegressor:
    """Pipeline whose target is standardized internally (predictions stay in USD)."""
    pipeline = Pipeline([("prep", preprocessor), ("model", estimator)])
    return TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())


def fit_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    ridge_alphas: np.ndarray | list[float] | None = None,
    lasso_alphas: np.ndarray | list[float] | None = None,
    cv: int = 5,
) -> dict[str, TransformedTargetRegressor]:
    """Fit OLS and tune Ridge and Lasso with cross-validated grid search."""
    ridge_alphas = DEFAULT_RIDGE_ALPHAS if ridge_alphas is None else ridge_alphas
    lasso_alphas = DEFAULT_LASSO_ALPHAS if lasso_alphas is None else lasso_alphas

    fitted: dict[str, TransformedTargetRegressor] = {}
    fitted["OLS"] = _wrap(LinearRegression(), build_preprocessor()).fit(X_train, y_train)
    logger.info("Trained OLS")

    searches = {
        "Ridge (L2)": (Ridge(), ridge_alphas),
        "Lasso (L1)": (Lasso(max_iter=10000), lasso_alphas),
    }
    for name, (estimator, alphas) in searches.items():
        search = GridSearchCV(
            _wrap(estimator, build_preprocessor()),
            {"regressor__model__alpha": list(alphas)},
            cv=cv,
            scoring="neg_root_mean_squared_error",
        )
        with warnings.catch_warnings():
            # Lasso with very small alphas (never selected) can warn about convergence.
            warnings.simplefilter("ignore", ConvergenceWarning)
            search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        logger.info(
            "Trained %s: best alpha = %.4g, CV RMSE = %.0f",
            name,
            search.best_params_["regressor__model__alpha"],
            -search.best_score_,
        )
    return fitted


def evaluate_model(
    model: TransformedTargetRegressor, X: pd.DataFrame, y: pd.Series
) -> dict[str, float]:
    """RMSE, MAE (both in target units) and R2 of one fitted model."""
    predictions = model.predict(X)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, predictions))),
        "MAE": float(mean_absolute_error(y, predictions)),
        "R2": float(r2_score(y, predictions)),
    }


def evaluate_models(
    models: dict[str, TransformedTargetRegressor], X: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    """Metrics table with one row per model."""
    rows = {name: evaluate_model(model, X, y) for name, model in models.items()}
    return pd.DataFrame(rows).T.rename_axis("Model")


def compute_vif(frame: pd.DataFrame) -> pd.Series:
    """Variance Inflation Factor of every column (regressed on all the others).

    Values above 10 signal serious multicollinearity.
    """
    values: dict[str, float] = {}
    for column in frame.columns:
        others = frame.drop(columns=column)
        r_squared = LinearRegression().fit(others, frame[column]).score(others, frame[column])
        values[column] = np.inf if r_squared >= 1 else 1.0 / (1.0 - r_squared)
    return pd.Series(values).sort_values(ascending=False)


def run_pipeline(data_path: Path | str, cv: int = 5) -> pd.DataFrame:
    """Run the full pipeline and return the test-set metrics of all models."""
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = prepare_data(df)
    models = fit_models(X_train, y_train, cv=cv)
    return evaluate_models(models, X_test, y_test)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run the housing regression pipeline.")
    parser.add_argument("--data", type=Path, default=Path("data/housing.csv"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = run_pipeline(args.data)
    logger.info("Test-set results:\n%s", results.round(4).to_string())


if __name__ == "__main__":
    main()
