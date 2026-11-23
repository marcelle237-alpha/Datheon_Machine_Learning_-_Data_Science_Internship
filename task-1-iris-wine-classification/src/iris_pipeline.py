"""Iris classification pipeline.

Loads the Iris starter file, splits it (80/20, stratified), scales the
features without data leakage, trains Logistic Regression and a Decision
Tree, and benchmarks both with accuracy, precision, recall and F1-score.

Usage (from the task folder):
    python src/iris_pipeline.py --data data/iris.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
LABEL_COLUMN = "species"


def load_data(path: Path | str) -> pd.DataFrame:
    """Load the Iris CSV, validate its schema and clean invalid rows.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if required columns are missing or no valid rows remain.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    required = FEATURE_COLUMNS + [LABEL_COLUMN]
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[required].copy()
    # Coerce feature columns to numbers; unparsable values become NaN.
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    n_invalid = int(df.isna().any(axis=1).sum())
    if n_invalid:
        logger.warning("Dropping %d row(s) with missing or invalid values", n_invalid)
        df = df.dropna()
    if df.empty:
        raise ValueError("No valid rows left after cleaning the data")

    logger.info("Loaded %d rows from %s", len(df), path)
    return df.reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split (every class keeps its proportion)."""
    features = df[FEATURE_COLUMNS]
    labels = df[LABEL_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Standardize features. The scaler is fitted on the training set only."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_models(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> dict[str, ClassifierMixin]:
    """Train Logistic Regression and a depth-limited Decision Tree."""
    models: dict[str, ClassifierMixin] = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=random_state),
    }
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        logger.info("Trained %s", name)
    return models


def evaluate_models(
    models: dict[str, ClassifierMixin],
    X_test_scaled: np.ndarray,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Accuracy and macro-averaged precision, recall and F1 for each model."""
    rows: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        rows[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (macro)": precision_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "Recall (macro)": recall_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "F1 (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
        }
    return pd.DataFrame(rows).T.rename_axis("Model")


def run_pipeline(data_path: Path | str) -> pd.DataFrame:
    """Run the full pipeline and return the test-set benchmark table."""
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
    models = train_models(X_train_scaled, y_train)
    return evaluate_models(models, X_test_scaled, y_test)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run the Iris classification pipeline.")
    parser.add_argument("--data", type=Path, default=Path("data/iris.csv"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = run_pipeline(args.data)
    logger.info("Test-set results:\n%s", results.round(4).to_string())


if __name__ == "__main__":
    main()
