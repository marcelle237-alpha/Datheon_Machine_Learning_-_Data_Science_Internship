"""Tests for the Iris classification pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

import iris_pipeline as ip


@pytest.fixture
def iris_df() -> pd.DataFrame:
    """The Iris data in the same format as the starter file."""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    df = df.drop(columns="target")
    df.columns = ip.FEATURE_COLUMNS + [ip.LABEL_COLUMN]
    return df


@pytest.fixture
def iris_csv(tmp_path, iris_df):
    path = tmp_path / "iris.csv"
    iris_df.to_csv(path, index=False)
    return path


# ---------- loading and validation ----------

def test_load_data_returns_expected_shape(iris_csv):
    df = ip.load_data(iris_csv)
    assert df.shape == (150, 5)
    assert list(df.columns) == ip.FEATURE_COLUMNS + [ip.LABEL_COLUMN]


def test_load_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ip.load_data(tmp_path / "does_not_exist.csv")


def test_load_data_missing_columns_raises(tmp_path, iris_df):
    path = tmp_path / "broken.csv"
    iris_df.drop(columns="petal_width").to_csv(path, index=False)
    with pytest.raises(ValueError, match="petal_width"):
        ip.load_data(path)


def test_load_data_drops_rows_with_missing_values(tmp_path, iris_df):
    iris_df.loc[0, "sepal_length"] = np.nan
    iris_df.loc[1, "species"] = np.nan
    path = tmp_path / "missing.csv"
    iris_df.to_csv(path, index=False)
    assert len(ip.load_data(path)) == 148


def test_load_data_coerces_invalid_numbers(tmp_path, iris_df):
    iris_df["sepal_width"] = iris_df["sepal_width"].astype(object)
    iris_df.loc[5, "sepal_width"] = "not a number"
    path = tmp_path / "text.csv"
    iris_df.to_csv(path, index=False)
    df = ip.load_data(path)
    assert len(df) == 149
    assert pd.api.types.is_float_dtype(df["sepal_width"])


def test_load_data_all_rows_invalid_raises(tmp_path):
    path = tmp_path / "empty.csv"
    data = {col: ["x"] for col in ip.FEATURE_COLUMNS}
    data[ip.LABEL_COLUMN] = ["setosa"]
    pd.DataFrame(data).to_csv(path, index=False)
    with pytest.raises(ValueError, match="No valid rows"):
        ip.load_data(path)


# ---------- splitting and scaling ----------

def test_split_is_80_20_and_stratified(iris_df):
    X_train, X_test, y_train, y_test = ip.split_data(iris_df)
    assert len(X_train) == 120
    assert len(X_test) == 30
    assert y_test.value_counts().to_dict() == {
        "setosa": 10,
        "versicolor": 10,
        "virginica": 10,
    }


def test_scaler_is_fitted_on_training_data_only():
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.normal(0, 1, (200, 2)), columns=["a", "b"])
    X_test = pd.DataFrame(rng.normal(5, 1, (50, 2)), columns=["a", "b"])

    X_train_scaled, X_test_scaled, _ = ip.scale_features(X_train, X_test)

    # the training data is standardized ...
    assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-9)
    # ... and the test data is NOT re-centered (that would be data leakage)
    assert (X_test_scaled.mean(axis=0) > 3).all()


# ---------- training and evaluation ----------

def test_train_models_returns_both_models(iris_df):
    X_train, X_test, y_train, _ = ip.split_data(iris_df)
    X_train_scaled, X_test_scaled, _ = ip.scale_features(X_train, X_test)
    models = ip.train_models(X_train_scaled, y_train)
    assert set(models) == {"Logistic Regression", "Decision Tree"}
    assert all(m.predict(X_test_scaled).shape == (30,) for m in models.values())


def test_evaluate_models_returns_valid_metrics(iris_csv):
    results = ip.run_pipeline(iris_csv)
    assert list(results.columns) == [
        "Accuracy",
        "Precision (macro)",
        "Recall (macro)",
        "F1 (macro)",
    ]
    assert ((results >= 0) & (results <= 1)).all().all()
    # Iris is easy: both models should clearly beat chance
    assert (results["Accuracy"] > 0.9).all()


def test_pipeline_is_reproducible(iris_csv):
    first = ip.run_pipeline(iris_csv)
    second = ip.run_pipeline(iris_csv)
    pd.testing.assert_frame_equal(first, second)
