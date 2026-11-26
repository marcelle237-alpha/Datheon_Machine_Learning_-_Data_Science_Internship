"""Tests for the sentiment classification pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

import sentiment_pipeline as sp

POSITIVE_WORDS = ["great", "excellent", "wonderful", "amazing", "enjoyable", "brilliant"]
NEGATIVE_WORDS = ["terrible", "awful", "boring", "dreadful", "pointless", "dull"]
FILLER_WORDS = ["movie", "film", "story", "acting", "plot", "scenes"]


@pytest.fixture
def reviews_df() -> pd.DataFrame:
    """Synthetic reviews: positive ones use positive words, negative ones negative words."""
    rng = np.random.default_rng(0)
    rows = []
    for label, words in (("pos", POSITIVE_WORDS), ("neg", NEGATIVE_WORDS)):
        for i in range(150):
            chosen = list(rng.choice(words, 3)) + list(rng.choice(FILLER_WORDS, 3))
            rng.shuffle(chosen)
            rows.append((f"The {' and the '.join(chosen)} number {i}.", label))
    return pd.DataFrame(rows, columns=["review", "sentiment"])


@pytest.fixture
def reviews_csv(tmp_path, reviews_df):
    path = tmp_path / "reviews.csv"
    reviews_df.to_csv(path, index=False)
    return path


@pytest.fixture
def fitted_model(reviews_csv):
    df = sp.load_data(reviews_csv)
    X_train, _, y_train, _ = sp.split_data(df)
    return sp.build_pipeline("logistic_regression").fit(X_train, y_train)


# ---------- text preprocessing ----------

def test_preprocess_lowercases_and_strips_html_and_punctuation():
    cleaned = sp.preprocess_text("<br />GREAT   Movie!!! 10/10, wow...", lemmatize=False)
    assert cleaned == "great movie wow"


def test_preprocess_removes_stopwords():
    tokens = sp.preprocess_text("the plot of the film is about a man", lemmatize=False).split()
    assert "the" not in tokens and "is" not in tokens
    assert "plot" in tokens and "film" in tokens


def test_preprocess_keeps_negations_by_default():
    assert "not" in sp.preprocess_text("This is not good").split()
    assert "not" in sp.preprocess_text("I don't like it").split()
    assert "not" in sp.preprocess_text("I can't stand it").split() or (
        "cannot" in sp.preprocess_text("I can't stand it").split()
    )


def test_preprocess_can_drop_negations():
    assert "not" not in sp.preprocess_text("This is not good", keep_negations=False).split()


def test_preprocess_lemmatizes_when_wordnet_is_available():
    if not sp.get_resources().has_lemmatizer:
        pytest.skip("WordNet not available")
    assert sp.preprocess_text("movies") == "movie"
    assert sp.preprocess_text("actors were running") == "actor run"


def test_preprocess_handles_empty_and_non_string_input():
    assert sp.preprocess_text("") == ""
    assert sp.preprocess_text("!!! ???") == ""
    assert sp.preprocess_text(None) == "none"  # str(None) is processed, no crash


def test_text_cleaner_transforms_all_texts_and_can_be_cloned():
    cleaner = sp.TextCleaner()
    result = cleaner.fit_transform(["Great film!", "Terrible film..."])
    assert len(result) == 2
    assert clone(cleaner).get_params() == cleaner.get_params()


def test_text_cleaner_without_preprocessing_passes_text_through():
    cleaner = sp.TextCleaner(preprocess=False)
    assert cleaner.transform(["The Film!"]) == ["The Film!"]


# ---------- loading and validation ----------

def test_load_data_encodes_labels(reviews_csv):
    df = sp.load_data(reviews_csv)
    assert set(df["sentiment"].unique()) == {0, 1}
    assert df["sentiment"].mean() == pytest.approx(0.5)


def test_load_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        sp.load_data(tmp_path / "missing.csv")


def test_load_data_missing_columns_raises(tmp_path, reviews_df):
    path = tmp_path / "broken.csv"
    reviews_df.rename(columns={"review": "text"}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="review"):
        sp.load_data(path)


def test_load_data_rejects_unknown_labels(tmp_path, reviews_df):
    reviews_df.loc[0, "sentiment"] = "meh"
    path = tmp_path / "labels.csv"
    reviews_df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unknown label"):
        sp.load_data(path)


def test_load_data_removes_empty_and_duplicate_reviews(tmp_path, reviews_df):
    extra = pd.DataFrame(
        {"review": ["", "   ", reviews_df.loc[0, "review"]], "sentiment": ["pos", "neg", "pos"]}
    )
    path = tmp_path / "dirty.csv"
    pd.concat([reviews_df, extra]).to_csv(path, index=False)
    assert len(sp.load_data(path)) == len(reviews_df)


def test_split_is_stratified(reviews_csv):
    df = sp.load_data(reviews_csv)
    X_train, X_test, y_train, y_test = sp.split_data(df)
    assert len(X_test) == pytest.approx(0.2 * len(df), abs=1)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.02)


# ---------- models ----------

def test_build_pipeline_rejects_unknown_classifier():
    with pytest.raises(ValueError, match="Unknown classifier"):
        sp.build_pipeline("svm")


def test_build_models_and_evaluate(reviews_csv):
    df = sp.load_data(reviews_csv)
    X_train, X_test, y_train, y_test = sp.split_data(df)
    models = sp.build_models()
    assert set(models) == {"Multinomial Naive Bayes", "Logistic Regression"}
    for model in models.values():
        model.fit(X_train, y_train)

    results = sp.evaluate_models(models, X_test, y_test)
    assert list(results.columns) == ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    assert ((results >= 0) & (results <= 1)).all().all()
    assert (results["Accuracy"] > 0.9).all()


def test_tfidf_respects_max_features(reviews_csv):
    df = sp.load_data(reviews_csv)
    X_train, _, y_train, _ = sp.split_data(df)
    model = sp.build_pipeline(max_features=7).fit(X_train, y_train)
    assert len(model.named_steps["tfidf"].get_feature_names_out()) == 7


def test_cross_validate_accuracy_returns_mean_and_std(reviews_csv):
    df = sp.load_data(reviews_csv)
    mean, std = sp.cross_validate_accuracy(
        sp.build_pipeline(), df["review"], df["sentiment"], cv=3
    )
    assert 0.9 < mean <= 1.0
    assert std >= 0


# ---------- predict_sentiment ----------

def test_predict_sentiment_on_clear_phrases(fitted_model):
    positive = sp.predict_sentiment("what a wonderful and excellent movie", fitted_model)
    negative = sp.predict_sentiment("an awful, boring and dreadful film", fitted_model)
    assert positive.sentiment == "positive" and positive.confidence > 0.5
    assert negative.sentiment == "negative" and negative.confidence > 0.5
    assert positive.positive_probability > negative.positive_probability


def test_predict_sentiment_rejects_empty_input(fitted_model):
    for bad in ["", "   ", None, 42]:
        with pytest.raises(ValueError):
            sp.predict_sentiment(bad, fitted_model)


def test_predict_sentiment_reports_unknown_for_unseen_vocabulary(fitted_model):
    result = sp.predict_sentiment("zxqv wvut plmk", fitted_model)
    assert result.sentiment == "unknown"
    assert result.confidence == 0.0


# ---------- external file (starter-file style) ----------

@pytest.fixture
def external_csv(tmp_path):
    path = tmp_path / "external.csv"
    pd.DataFrame(
        {
            "review_id": [1, 2, 3, 4],
            "review_text": [
                "wonderful and excellent",
                "awful and dreadful",
                "just average",
                "xqzv wvut",
            ],
            "sentiment": ["positive", "negative", "neutral", "positive"],
        }
    ).to_csv(path, index=False)
    return path


def test_evaluate_external_file_skips_neutral_and_counts_unknown_as_wrong(
    fitted_model, external_csv
):
    table, accuracy = sp.evaluate_external_file(fitted_model, external_csv)
    assert len(table) == 3  # the neutral row is skipped
    assert list(table.columns) == ["text", "actual", "predicted", "confidence", "correct"]
    assert table.loc[0, "predicted"] == "positive"
    assert table.loc[1, "predicted"] == "negative"
    assert table.loc[2, "predicted"] == "unknown" and not table.loc[2, "correct"]
    assert accuracy == pytest.approx(2 / 3)


def test_evaluate_external_file_validates_input(fitted_model, tmp_path, external_csv):
    with pytest.raises(FileNotFoundError):
        sp.evaluate_external_file(fitted_model, tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="review_text"):
        sp.evaluate_external_file(fitted_model, external_csv, text_column="review_text_x")

    only_neutral = tmp_path / "neutral.csv"
    pd.DataFrame({"review_text": ["fine"], "sentiment": ["neutral"]}).to_csv(
        only_neutral, index=False
    )
    with pytest.raises(ValueError, match="No positive or negative"):
        sp.evaluate_external_file(fitted_model, only_neutral)


# ---------- interpretation and end-to-end ----------

def test_top_words_for_both_classifiers(reviews_csv):
    # words are lemmatized by the pipeline ("amazing" -> "amaze"), so compare cleaned forms
    positive_vocab = {sp.preprocess_text(w) for w in POSITIVE_WORDS}
    negative_vocab = {sp.preprocess_text(w) for w in NEGATIVE_WORDS}
    df = sp.load_data(reviews_csv)
    X_train, _, y_train, _ = sp.split_data(df)
    for classifier in ("logistic_regression", "naive_bayes"):
        model = sp.build_pipeline(classifier).fit(X_train, y_train)
        table = sp.top_words(model, n=5)
        assert len(table) == 5
        assert table["positive"].iloc[0] in positive_vocab
        assert table["negative"].iloc[0] in negative_vocab
        assert table["positive_score"].is_monotonic_decreasing
        assert table["negative_score"].is_monotonic_increasing


def test_run_pipeline_end_to_end(reviews_csv):
    results = sp.run_pipeline(reviews_csv, cv=3)
    assert set(results.index) == {"Multinomial Naive Bayes", "Logistic Regression"}
    assert (results["Accuracy"] > 0.9).all()
