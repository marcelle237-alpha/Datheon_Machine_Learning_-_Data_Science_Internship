"""Tests for the California housing regression pipeline."""

import numpy as np
import pandas as pd
import pytest

import housing_pipeline as hp


@pytest.fixture
def housing_df() -> pd.DataFrame:
    """Small synthetic dataset with the same schema as the real data."""
    rng = np.random.default_rng(0)
    n = 400
    households = rng.integers(50, 600, n).astype(float)
    total_rooms = households * rng.uniform(3, 8, n)
    income = rng.uniform(1, 10, n)
    age = rng.integers(1, 52, n).astype(float)
    ocean = rng.choice(["INLAND", "<1H OCEAN", "NEAR BAY", "NEAR OCEAN"], n)
    value = 40000 * income + 800 * age + 5000 * (ocean == "NEAR BAY") + 50000
    value = value + rng.normal(0, 10000, n)

    df = pd.DataFrame(
        {
            "longitude": rng.uniform(-124, -114, n),
            "latitude": rng.uniform(32, 42, n),
            "housing_median_age": age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_rooms * rng.uniform(0.15, 0.3, n),
            "population": households * rng.uniform(2, 4, n),
            "households": households,
            "median_income": income,
            "median_house_value": value,
            "ocean_proximity": ocean,
        }
    )
    df.loc[df.index[:10], "total_bedrooms"] = np.nan
    return df


@pytest.fixture
def housing_csv(tmp_path, housing_df):
    path = tmp_path / "housing.csv"
    housing_df.to_csv(path, index=False)
    return path


# ---------- loading and validation ----------

def test_load_data_keeps_missing_features_for_imputation(housing_csv):
    df = hp.load_data(housing_csv)
    assert len(df) == 400
    assert df["total_bedrooms"].isna().sum() == 10


def test_load_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        hp.load_data(tmp_path / "missing.csv")


def test_load_data_missing_columns_raises(tmp_path, housing_df):
    path = tmp_path / "broken.csv"
    housing_df.drop(columns="median_income").to_csv(path, index=False)
    with pytest.raises(ValueError, match="median_income"):
        hp.load_data(path)


def test_load_data_drops_rows_without_target(tmp_path, housing_df):
    housing_df.loc[0:4, "median_house_value"] = np.nan
    path = tmp_path / "no_target.csv"
    housing_df.to_csv(path, index=False)
    assert len(hp.load_data(path)) == 395


def test_load_data_coerces_text_in_numeric_columns(tmp_path, housing_df):
    housing_df["median_income"] = housing_df["median_income"].astype(object)
    housing_df.loc[3, "median_income"] = "n/a"
    path = tmp_path / "text.csv"
    housing_df.to_csv(path, index=False)
    df = hp.load_data(path)
    assert pd.api.types.is_float_dtype(df["median_income"])
    assert df["median_income"].isna().sum() == 1


# ---------- feature engineering ----------

def test_add_features_computes_correct_ratios(housing_df):
    stats = hp.compute_feature_stats(housing_df)
    result = hp.add_features(housing_df, stats)
    row = housing_df.iloc[20]
    assert result.loc[20, "rooms_per_household"] == pytest.approx(
        row["total_rooms"] / row["households"]
    )
    assert result.loc[20, "population_per_household"] == pytest.approx(
        row["population"] / row["households"]
    )
    assert set(hp.ENGINEERED_FEATURES) <= set(result.columns)


def test_add_features_handles_zero_households(housing_df):
    housing_df.loc[0, "households"] = 0
    stats = hp.compute_feature_stats(housing_df)
    result = hp.add_features(housing_df, stats)
    assert np.isnan(result.loc[0, "rooms_per_household"])
    assert not np.isinf(result[hp.ENGINEERED_FEATURES].to_numpy(dtype=float)).any()


def test_feature_stats_come_from_the_given_data_only(housing_df):
    train = housing_df.iloc[:300]
    stats = hp.compute_feature_stats(train)
    assert stats.income_mean == pytest.approx(train["median_income"].mean())
    assert stats.income_mean != pytest.approx(housing_df["median_income"].mean())


def test_prepare_data_splits_80_20_and_selects_features(housing_df):
    X_train, X_test, y_train, y_test = hp.prepare_data(housing_df)
    assert len(X_train) == 320
    assert len(X_test) == 80
    assert list(X_train.columns) == hp.NUMERIC_FEATURES + hp.CATEGORICAL_FEATURES
    for dropped in hp.REDUNDANT_COUNT_COLUMNS:
        assert dropped not in X_train.columns


# ---------- preprocessing ----------

def test_preprocessor_imputes_missing_values_and_encodes_categories(housing_df):
    X_train, _, _, _ = hp.prepare_data(housing_df)
    assert X_train.isna().any().any()  # bedrooms_per_room still has NaN before imputation

    matrix = hp.build_preprocessor().fit_transform(X_train)
    n_categories = housing_df["ocean_proximity"].nunique()
    assert not np.isnan(matrix).any()
    assert matrix.shape == (len(X_train), len(hp.NUMERIC_FEATURES) + n_categories)


# ---------- multicollinearity ----------

def test_vif_flags_collinear_columns():
    rng = np.random.default_rng(1)
    a = rng.normal(size=500)
    frame = pd.DataFrame(
        {
            "a": a,
            "a_twice": 2 * a + rng.normal(scale=0.01, size=500),
            "independent": rng.normal(size=500),
        }
    )
    vif = hp.compute_vif(frame)
    assert vif["a"] > 100
    assert vif["a_twice"] > 100
    assert vif["independent"] < 2


# ---------- models ----------

def test_fit_and_evaluate_models(housing_df):
    X_train, X_test, y_train, y_test = hp.prepare_data(housing_df)
    models = hp.fit_models(
        X_train, y_train, ridge_alphas=[0.1, 1.0], lasso_alphas=[1e-4, 1e-3], cv=3
    )
    assert set(models) == {"OLS", "Ridge (L2)", "Lasso (L1)"}

    results = hp.evaluate_models(models, X_test, y_test)
    assert list(results.columns) == ["RMSE", "MAE", "R2"]
    assert (results["R2"] > 0.8).all()
    assert (results["RMSE"] > 0).all()


def test_predictions_are_in_original_units(housing_df):
    X_train, X_test, y_train, _ = hp.prepare_data(housing_df)
    models = hp.fit_models(X_train, y_train, ridge_alphas=[1.0], lasso_alphas=[1e-3], cv=3)
    predictions = models["OLS"].predict(X_test)
    # the target is standardized internally, predictions must still be in USD
    assert predictions.mean() == pytest.approx(y_train.mean(), rel=0.2)


def test_run_pipeline_end_to_end(housing_csv):
    results = hp.run_pipeline(housing_csv, cv=3)
    assert set(results.index) == {"OLS", "Ridge (L2)", "Lasso (L1)"}
    assert (results["R2"] > 0.8).all()
