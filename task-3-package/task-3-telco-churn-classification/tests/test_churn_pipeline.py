"""Tests for the Telco churn pipeline."""

import numpy as np
import pandas as pd
import pytest

import churn_pipeline as cp


@pytest.fixture
def churn_df() -> pd.DataFrame:
    """Synthetic data with the same schema; churn depends on contract and tenure."""
    rng = np.random.default_rng(0)
    n = 600
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.6, 0.2, 0.2])
    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(20, 110, n).round(2)

    risk = 1.2 * (contract == "Month-to-month") - 0.03 * tenure - 1.0
    churn = rng.random(n) < 1 / (1 + np.exp(-risk))

    yes_no = ["Yes", "No"]
    data = {
        "customerID": [f"ID-{i}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(yes_no, n),
        "Dependents": rng.choice(yes_no, n),
        "tenure": tenure,
        "PhoneService": rng.choice(yes_no, n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(yes_no, n),
        "OnlineBackup": rng.choice(yes_no, n),
        "DeviceProtection": rng.choice(yes_no, n),
        "TechSupport": rng.choice(yes_no, n),
        "StreamingTV": rng.choice(yes_no, n),
        "StreamingMovies": rng.choice(yes_no, n),
        "Contract": contract,
        "PaperlessBilling": rng.choice(yes_no, n),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check"], n),
        "MonthlyCharges": monthly,
        "TotalCharges": (monthly * tenure).round(2).astype(str),
        "Churn": np.where(churn, "Yes", "No"),
    }
    df = pd.DataFrame(data)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "  # blank strings like the real file
    return df


@pytest.fixture
def churn_csv(tmp_path, churn_df):
    path = tmp_path / "churn.csv"
    churn_df.to_csv(path, index=False)
    return path


# ---------- loading and validation ----------

def test_load_data_converts_total_charges_and_target(churn_csv, churn_df):
    df = cp.load_data(churn_csv)
    assert pd.api.types.is_float_dtype(df["TotalCharges"])
    expected_blanks = int((churn_df["tenure"] == 0).sum())
    assert df["TotalCharges"].isna().sum() == expected_blanks
    assert set(df[cp.TARGET].unique()) <= {0, 1}
    assert cp.ID_COLUMN not in df.columns


def test_load_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cp.load_data(tmp_path / "missing.csv")


def test_load_data_missing_columns_raises(tmp_path, churn_df):
    path = tmp_path / "broken.csv"
    churn_df.drop(columns="Contract").to_csv(path, index=False)
    with pytest.raises(ValueError, match="Contract"):
        cp.load_data(path)


def test_load_data_rejects_unexpected_target_values(tmp_path, churn_df):
    churn_df.loc[0, "Churn"] = "Maybe"
    path = tmp_path / "bad_target.csv"
    churn_df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Yes/No"):
        cp.load_data(path)


def test_load_data_drops_rows_without_target(tmp_path, churn_df):
    churn_df.loc[:2, "Churn"] = np.nan
    path = tmp_path / "no_target.csv"
    churn_df.to_csv(path, index=False)
    assert len(cp.load_data(path)) == len(churn_df) - 3


# ---------- splitting and preprocessing ----------

def test_split_is_stratified(churn_csv):
    df = cp.load_data(churn_csv)
    _, _, y_train, y_test = cp.split_data(df)
    assert len(y_test) == pytest.approx(0.2 * len(df), abs=1)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.02)


def test_preprocessor_imputes_and_ignores_unseen_categories(churn_csv):
    df = cp.load_data(churn_csv)
    X_train, X_test, _, _ = cp.split_data(df)
    assert X_train["TotalCharges"].isna().any()

    preprocessor = cp.build_preprocessor().fit(X_train)
    X_new = X_test.copy()
    X_new["PaymentMethod"] = "Crypto"  # category never seen during training
    matrix = preprocessor.transform(X_new)
    assert not np.isnan(matrix).any()
    assert matrix.shape[0] == len(X_new)


def test_scale_pos_weight_is_negatives_over_positives():
    y = pd.Series([0] * 80 + [1] * 20)
    assert cp.compute_scale_pos_weight(y) == pytest.approx(4.0)


def test_scale_pos_weight_requires_positive_examples():
    with pytest.raises(ValueError):
        cp.compute_scale_pos_weight(pd.Series([0, 0, 0]))


# ---------- models ----------

def test_build_models_returns_six_strategies(churn_csv):
    df = cp.load_data(churn_csv)
    _, _, y_train, _ = cp.split_data(df)
    models = cp.build_models(y_train, n_estimators=10)
    assert len(models) == 6
    smote_models = [name for name in models if "SMOTE" in name]
    assert len(smote_models) == 2
    for name in smote_models:
        assert "smote" in models[name].named_steps


def test_smote_pipelines_do_not_resample_at_prediction_time(churn_csv):
    df = cp.load_data(churn_csv)
    X_train, X_test, y_train, _ = cp.split_data(df)
    model = cp.build_models(y_train, n_estimators=10)["Random Forest (SMOTE)"]
    model.fit(X_train, y_train)
    assert model.predict_proba(X_test).shape == (len(X_test), 2)


def test_evaluate_models_returns_valid_metrics(churn_csv):
    df = cp.load_data(churn_csv)
    X_train, X_test, y_train, y_test = cp.split_data(df)
    models = cp.build_models(y_train, n_estimators=30)
    for model in models.values():
        model.fit(X_train, y_train)

    results = cp.evaluate_models(models, X_test, y_test)
    assert len(results) == 6
    assert ((results >= 0) & (results <= 1)).all().all()
    assert (results["ROC-AUC"] > 0.65).all()


def test_class_weights_increase_recall_on_imbalanced_data(churn_csv):
    df = cp.load_data(churn_csv)
    X_train, X_test, y_train, y_test = cp.split_data(df)
    models = cp.build_models(y_train, n_estimators=30)
    for name in ("Random Forest (no balancing)", "Random Forest (class weights)"):
        models[name].fit(X_train, y_train)
    results = cp.evaluate_models(
        {k: models[k] for k in ("Random Forest (no balancing)", "Random Forest (class weights)")},
        X_test,
        y_test,
    )
    assert (
        results.loc["Random Forest (class weights)", "Recall (churn)"]
        >= results.loc["Random Forest (no balancing)", "Recall (churn)"]
    )


def test_top_feature_importances_are_sorted(churn_csv):
    df = cp.load_data(churn_csv)
    X_train, _, y_train, _ = cp.split_data(df)
    model = cp.build_models(y_train, n_estimators=30)["Random Forest (class weights)"]
    model.fit(X_train, y_train)

    top = cp.top_feature_importances(model, n=10)
    assert len(top) == 10
    assert top.is_monotonic_decreasing
    assert (top >= 0).all()


def test_cross_validate_models_reports_mean_and_std(churn_csv):
    df = cp.load_data(churn_csv)
    X_train, _, y_train, _ = cp.split_data(df)
    models = {"RF": cp.build_models(y_train, n_estimators=10)["Random Forest (class weights)"]}
    table = cp.cross_validate_models(models, X_train, y_train, cv=3)
    assert list(table.columns) == ["CV ROC-AUC (mean)", "CV ROC-AUC (std)"]
    assert 0.5 < table.loc["RF", "CV ROC-AUC (mean)"] <= 1.0
