"""Sentiment classification of movie reviews (TF-IDF + Naive Bayes / Logistic Regression).

Cleans raw review text (lowercasing, HTML and punctuation removal, stopword
removal, lemmatization), turns it into TF-IDF features, trains a Multinomial
Naive Bayes and a Logistic Regression classifier and offers a
``predict_sentiment`` helper for custom phrases.

Usage (from the task folder):
    python src/sentiment_pipeline.py --data data/movie_reviews.csv
    python src/sentiment_pipeline.py --data data/movie_reviews.csv --predict "A great film"
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEXT_COLUMN = "review"
LABEL_COLUMN = "sentiment"
LABEL_MAP = {"pos": 1, "positive": 1, "1": 1, "neg": 0, "negative": 0, "0": 0}

# Negations must survive stopword removal: "not good" and "good" are opposites.
NEGATIONS = frozenset(
    {"no", "nor", "not", "never", "none", "nothing", "neither", "nowhere", "cannot"}
)


@dataclass(frozen=True)
class NlpResources:
    """Stopwords and a lemmatizer (with a safe fallback if NLTK data is missing)."""

    stop_words: frozenset[str]
    lemmatize: Callable[[str], str]
    has_lemmatizer: bool


@lru_cache(maxsize=1)
def get_resources() -> NlpResources:
    """Load NLTK stopwords and the WordNet lemmatizer, downloading them if needed.

    Falls back to scikit-learn's stopword list and no lemmatization when the
    NLTK data cannot be downloaded (for example without internet access).
    """
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        for resource in ("stopwords", "wordnet", "omw-1.4"):
            nltk.download(resource, quiet=True)
        stop_words = frozenset(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("test")  # fails fast if WordNet is unavailable

        @lru_cache(maxsize=None)
        def lemmatize(word: str) -> str:
            return lemmatizer.lemmatize(lemmatizer.lemmatize(word, "v"), "n")

        return NlpResources(stop_words, lemmatize, True)
    except Exception as error:  # noqa: BLE001 - any failure triggers the fallback
        logger.warning("NLTK resources unavailable (%s); using fallback", error)
        return NlpResources(frozenset(ENGLISH_STOP_WORDS), lambda word: word, False)


def preprocess_text(
    text: str,
    keep_negations: bool = True,
    lemmatize: bool = True,
) -> str:
    """Clean one review: lowercase, strip HTML/punctuation, drop stopwords, lemmatize."""
    resources = get_resources()
    text = re.sub(r"<[^>]+>", " ", str(text).lower())  # HTML tags
    text = text.replace("can't", "cannot").replace("won't", "will not")
    text = re.sub(r"n't\b", " not", text)  # "don't" -> "do not"
    tokens = re.findall(r"[a-z]+", text)  # drops digits and punctuation

    stop_words = resources.stop_words
    if keep_negations:
        stop_words = stop_words - NEGATIONS
    tokens = [t for t in tokens if len(t) > 1 and t not in stop_words]
    if lemmatize:
        tokens = [resources.lemmatize(t) for t in tokens]
    return " ".join(tokens)


class TextCleaner(BaseEstimator, TransformerMixin):
    """scikit-learn transformer that applies :func:`preprocess_text` to raw texts.

    With ``preprocess=False`` the text is passed through unchanged (baseline).
    """

    def __init__(
        self, preprocess: bool = True, keep_negations: bool = True, lemmatize: bool = True
    ) -> None:
        self.preprocess = preprocess
        self.keep_negations = keep_negations
        self.lemmatize = lemmatize

    def fit(self, X: object, y: object = None) -> TextCleaner:
        return self

    def transform(self, X: object) -> list[str]:
        if not self.preprocess:
            return [str(text) for text in X]
        return [
            preprocess_text(text, self.keep_negations, self.lemmatize) for text in X
        ]


def load_data(path: Path | str) -> pd.DataFrame:
    """Load the reviews CSV, validate it and encode the label (1 = positive).

    Empty reviews and exact duplicates are removed.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if columns are missing, labels are unknown or no rows remain.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    missing_columns = [c for c in (TEXT_COLUMN, LABEL_COLUMN) if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()
    labels = df[LABEL_COLUMN].astype("string").str.strip().str.lower().map(LABEL_MAP)
    unknown = int((labels.isna() & df[LABEL_COLUMN].notna()).sum())
    if unknown:
        raise ValueError(f"'{LABEL_COLUMN}' contains {unknown} unknown label value(s)")
    df[LABEL_COLUMN] = labels

    before = len(df)
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    df = df[df[TEXT_COLUMN].astype(str).str.strip() != ""]
    df = df.drop_duplicates(subset=TEXT_COLUMN)
    if len(df) < before:
        logger.warning("Removed %d empty or duplicate review(s)", before - len(df))
    if df.empty:
        raise ValueError("No valid rows left after cleaning the data")

    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    logger.info("Loaded %d reviews from %s (%.1f%% positive)", len(df), path,
                100 * df[LABEL_COLUMN].mean())
    return df.reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Stratified train/test split of the raw review texts."""
    X_train, X_test, y_train, y_test = train_test_split(
        df[TEXT_COLUMN],
        df[LABEL_COLUMN],
        test_size=test_size,
        random_state=random_state,
        stratify=df[LABEL_COLUMN],
    )
    return X_train, X_test, y_train, y_test


def build_pipeline(
    classifier: str = "logistic_regression",
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 1),
    cleaner: TextCleaner | None = None,
) -> Pipeline:
    """Raw text -> cleaning -> TF-IDF -> classifier, as a single pipeline."""
    models = {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    }
    if classifier not in models:
        raise ValueError(f"Unknown classifier '{classifier}'; choose from {sorted(models)}")
    return Pipeline(
        [
            ("clean", cleaner or TextCleaner()),
            ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
            ("model", models[classifier]),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    """The two classifiers of the task, each as a full text pipeline."""
    return {
        "Multinomial Naive Bayes": build_pipeline("naive_bayes"),
        "Logistic Regression": build_pipeline("logistic_regression"),
    }


def evaluate_models(
    models: dict[str, Pipeline], X_test: pd.Series, y_test: pd.Series
) -> pd.DataFrame:
    """Accuracy, precision, recall, F1 and ROC-AUC (positive class) per model."""
    rows: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        rows[name] = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1": f1_score(y_test, predictions, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
        }
    return pd.DataFrame(rows).T.rename_axis("Model")


def cross_validate_accuracy(
    pipeline: Pipeline, X: pd.Series, y: pd.Series, cv: int = 5
) -> tuple[float, float]:
    """Mean and standard deviation of the cross-validated accuracy."""
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


@dataclass(frozen=True)
class Prediction:
    """Result of :func:`predict_sentiment`."""

    text: str
    sentiment: str  # "positive", "negative" or "unknown"
    confidence: float  # probability of the predicted class (0.0 if unknown)
    positive_probability: float


def predict_sentiment(text: str, model: Pipeline) -> Prediction:
    """Classify a custom phrase with a fitted text pipeline.

    If none of the words are known to the model's vocabulary, the model would
    only return its class prior, so the result is reported as ``"unknown"``.

    Raises:
        ValueError: if ``text`` is not a non-empty string.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please provide a non-empty text")

    features = model[:-1].transform([text])
    if features.nnz == 0:
        return Prediction(text, "unknown", 0.0, 0.5)

    positive = float(model.predict_proba([text])[0, 1])
    if positive >= 0.5:
        return Prediction(text, "positive", positive, positive)
    return Prediction(text, "negative", 1 - positive, positive)


def evaluate_external_file(
    model: Pipeline,
    path: Path | str,
    text_column: str = "review_text",
    label_column: str = "sentiment",
) -> tuple[pd.DataFrame, float]:
    """Test a fitted model on another labelled file (for example product reviews).

    Only rows labelled positive or negative are used; other labels such as
    "neutral" are skipped because the model is binary. A prediction of
    "unknown" counts as wrong.

    Returns:
        A table with one row per review and the accuracy over these rows.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if columns are missing or no positive/negative rows exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    missing_columns = [c for c in (text_column, label_column) if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    labels = df[label_column].astype("string").str.strip().str.lower().map(LABEL_MAP)
    skipped = int(labels.isna().sum())
    if skipped:
        logger.info("Skipping %d row(s) that are neither positive nor negative", skipped)
    df = df.loc[labels.notna() & df[text_column].notna()].copy()
    labels = labels.loc[df.index].astype(int)
    if df.empty:
        raise ValueError("No positive or negative reviews found in the file")

    predictions = [predict_sentiment(str(text), model) for text in df[text_column]]
    table = pd.DataFrame(
        {
            "text": df[text_column].to_numpy(),
            "actual": ["positive" if label == 1 else "negative" for label in labels],
            "predicted": [p.sentiment for p in predictions],
            "confidence": [round(p.confidence, 2) for p in predictions],
        }
    )
    table["correct"] = table["actual"] == table["predicted"]
    return table, float(table["correct"].mean())


def top_words(model: Pipeline, n: int = 10) -> pd.DataFrame:
    """The ``n`` words that push a review most strongly towards each sentiment."""
    words = model.named_steps["tfidf"].get_feature_names_out()
    classifier = model.named_steps["model"]
    if hasattr(classifier, "coef_"):
        scores = classifier.coef_[0]
    else:  # Naive Bayes: log-probability difference between the classes
        scores = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
    ranked = pd.Series(scores, index=words).sort_values()
    return pd.DataFrame(
        {
            "positive": ranked.index[::-1][:n],
            "positive_score": ranked.values[::-1][:n],
            "negative": ranked.index[:n],
            "negative_score": ranked.values[:n],
        }
    )


def run_pipeline(data_path: Path | str, cv: int = 5) -> pd.DataFrame:
    """Run the whole pipeline and return the test-set benchmark table."""
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)
    models = build_models()
    for name, model in models.items():
        mean, std = cross_validate_accuracy(model, X_train, y_train, cv=cv)
        logger.info("%s: CV accuracy = %.4f +/- %.4f", name, mean, std)
        model.fit(X_train, y_train)
    return evaluate_models(models, X_test, y_test)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run the sentiment classification pipeline.")
    parser.add_argument("--data", type=Path, default=Path("data/movie_reviews.csv"))
    parser.add_argument("--predict", help="classify a custom phrase after training")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load_data(args.data)
    X_train, X_test, y_train, y_test = split_data(df)
    models = build_models()
    for model in models.values():
        model.fit(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    logger.info("Test-set results:\n%s", results.round(4).to_string())

    if args.predict:
        result = predict_sentiment(args.predict, models["Logistic Regression"])
        logger.info("%r -> %s (confidence %.2f)", result.text, result.sentiment, result.confidence)


if __name__ == "__main__":
    main()
