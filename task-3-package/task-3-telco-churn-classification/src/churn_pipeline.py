"""Telco customer churn pipeline (Random Forest and XGBoost).

Loads the Telco Customer Churn data, preprocesses it (numeric conversion,
imputation, one-hot encoding), handles the class imbalance with class weights
or SMOTE (applied inside the training folds only), trains Random Forest and
XGBoost classifiers and evaluates them with ROC-AUC and classification metrics.

Usage (from the task folder):
    python src/churn_pipeline.py --data data/telco_churn.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
ID_COLUMN = "customerID"
TARGET = "Churn"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_FEATURES = ["SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = FEATURES + [TARGET]


def load_data(path: Path | str) -> pd.DataFrame:
    """Load the churn CSV, validate it and prepare the target.

    - ``TotalCharges`` contains blank strings for brand-new customers; they are
      converted to numbers and left as NaN (imputed later in the pipeline).
    - ``Churn`` ("Yes"/"No") is encoded as 1/0.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if required columns are missing, the target contains
            unexpected values or no rows remain.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[REQUIRED_COLUMNS].copy()

    numeric = NUMERIC_FEATURES + BINARY_FEATURES
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")

    target_text = df[TARGET].astype("string").str.strip().str.lower()
    mapped = target_text.map({"yes": 1, "no": 0})
    invalid = int((mapped.isna() & df[TARGET].notna()).sum())
    if invalid:
        raise ValueError(f"'{TARGET}' contains {invalid} value(s) other than Yes/No")
    keep = mapped.notna()
    if int((~keep).sum()):
        logger.warning("Dropping %d row(s) without a target value", int((~keep).sum()))
    df = df.loc[keep].copy()
    df[TARGET] = mapped.loc[keep].astype(int)
    if df.empty:
        raise ValueError("No valid rows left after cleaning the data")

    logger.info(
        "Loaded %d rows from %s (churn rate %.1f%%, %d missing TotalCharges)",
        len(df),
        path,
        100 * df[TARGET].mean(),
        int(df["TotalCharges"].isna().sum()),
    )
    return df.reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split so both parts keep the churn rate."""
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET],
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    """Median imputation for numbers, one-hot encoding for categories."""
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def compute_scale_pos_weight(y: pd.Series) -> float:
    """XGBoost class weight: number of negatives divided by number of positives."""
    positives = int((y == 1).sum())
    if positives == 0:
        raise ValueError("The target contains no positive (churn) examples")
    return float((y == 0).sum() / positives)


def build_models(
    y_train: pd.Series,
    n_estimators: int = 200,
    random_state: int = RANDOM_STATE,
) -> dict[str, Pipeline | ImbPipeline]:
    """Six models: Random Forest and XGBoost with three imbalance strategies.

    SMOTE lives inside an imbalanced-learn pipeline, so synthetic samples are
    created from the training part of each cross-validation fold only and never
    leak into validation or test data.
    """
    weight = compute_scale_pos_weight(y_train)

    def forest(**kwargs: object) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
            **kwargs,
        )

    def xgboost(**kwargs: object) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
            **kwargs,
        )

    def plain(model: object) -> Pipeline:
        return Pipeline([("prep", build_preprocessor()), ("model", model)])

    def with_smote(model: object) -> ImbPipeline:
        return ImbPipeline(
            [
                ("prep", build_preprocessor()),
                ("smote", SMOTE(random_state=random_state)),
                ("model", model),
            ]
        )

    return {
        "Random Forest (no balancing)": plain(forest()),
        "Random Forest (class weights)": plain(forest(class_weight="balanced")),
        "Random Forest (SMOTE)": with_smote(forest()),
        "XGBoost (no balancing)": plain(xgboost()),
        "XGBoost (scale_pos_weight)": plain(xgboost(scale_pos_weight=weight)),
        "XGBoost (SMOTE)": with_smote(xgboost()),
    }


def cross_validate_models(
    models: dict[str, Pipeline | ImbPipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
) -> pd.DataFrame:
    """Stratified k-fold ROC-AUC (mean and standard deviation) per model."""
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    rows: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="roc_auc")
        rows[name] = {"CV ROC-AUC (mean)": scores.mean(), "CV ROC-AUC (std)": scores.std()}
        logger.info("%s: CV ROC-AUC = %.4f +/- %.4f", name, scores.mean(), scores.std())
    return pd.DataFrame(rows).T.rename_axis("Model")


def evaluate_models(
    models: dict[str, Pipeline | ImbPipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """ROC-AUC, average precision and class-1 (churn) metrics on the test set."""
    rows: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        rows[name] = {
            "ROC-AUC": roc_auc_score(y_test, probabilities),
            "Avg. precision": average_precision_score(y_test, probabilities),
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision (churn)": precision_score(y_test, predictions, zero_division=0),
            "Recall (churn)": recall_score(y_test, predictions, zero_division=0),
            "F1 (churn)": f1_score(y_test, predictions, zero_division=0),
        }
    return pd.DataFrame(rows).T.rename_axis("Model")


def top_feature_importances(model: Pipeline | ImbPipeline, n: int = 10) -> pd.Series:
    """The ``n`` most important features of a fitted tree-based pipeline."""
    names = model.named_steps["prep"].get_feature_names_out()
    names = [name.split("__", 1)[1] for name in names]
    importances = pd.Series(model.named_steps["model"].feature_importances_, index=names)
    return importances.sort_values(ascending=False).head(n)


def run_pipeline(
    data_path: Path | str, cv: int = 5, n_estimators: int = 200
) -> pd.DataFrame:
    """Run the whole pipeline and return the test-set benchmark table."""
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)
    models = build_models(y_train, n_estimators=n_estimators)
    cross_validate_models(models, X_train, y_train, cv=cv)
    for name, model in models.items():
        model.fit(X_train, y_train)
        logger.info("Trained %s", name)
    return evaluate_models(models, X_test, y_test)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run the churn classification pipeline.")
    parser.add_argument("--data", type=Path, default=Path("data/telco_churn.csv"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = run_pipeline(args.data)
    logger.info("Test-set results:\n%s", results.round(4).to_string())


if __name__ == "__main__":
    main()
