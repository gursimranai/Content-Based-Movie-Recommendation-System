from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from movie_recommender.config import (
    MODELS_DIR,
    MOVIE_FEATURES_PATH,
    MOVIE_METADATA_PATH,
    PROCESSED_DATA_PATH,
    TFIDF_MATRIX_PATH,
    TFIDF_VECTORIZER_PATH,
)


def load_features(data_path: Path) -> pd.DataFrame:
    """Load and validate the processed movie feature dataset."""

    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {data_path}\n"
            "Run Notebook 03 and Notebook 04 first."
        )

    movies = pd.read_csv(data_path)

    required_columns = {"id", "title", "tags"}
    missing_columns = required_columns.difference(movies.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    movies = movies.dropna(subset=["id", "title", "tags"]).copy()

    movies["title"] = movies["title"].astype(str).str.strip()
    movies["tags"] = movies["tags"].astype(str).str.strip()

    movies = movies[movies["title"].ne("") & movies["tags"].ne("")].reset_index(
        drop=True
    )

    if movies.empty:
        raise ValueError("No valid movie records found in the dataset.")

    if movies["id"].duplicated().any():
        raise ValueError("Dataset contains duplicate movie IDs.")

    if movies["title"].duplicated().any():
        print(
            "Warning: Duplicate movie titles exist. "
            "The recommender selects the first matching title."
        )

    print(f"Loaded {len(movies):,} valid movies.")

    return movies


def save_artifact(artifact: object, destination: Path) -> None:
    """Save an artifact safely using a temporary file."""

    temporary_path = destination.with_suffix(destination.suffix + ".tmp")

    try:
        joblib.dump(artifact, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def train_model(movies: pd.DataFrame) -> None:
    """Train the TF-IDF representation and save model artifacts."""

    print("\nStarting TF-IDF vectorization...")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
    )

    tfidf_matrix = vectorizer.fit_transform(movies["tags"])

    if tfidf_matrix.shape[1] == 0:
        raise ValueError("TF-IDF produced an empty feature matrix.")

    movie_metadata = movies[["id", "title"]].copy()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = {
        TFIDF_VECTORIZER_PATH: vectorizer,
        TFIDF_MATRIX_PATH: tfidf_matrix,
        MOVIE_METADATA_PATH: movie_metadata,
        MOVIE_FEATURES_PATH: movies.copy(),
    }

    print("\nSaving model artifacts...")

    for path, artifact in artifacts.items():
        save_artifact(artifact, path)
        print(f"Saved: {path.relative_to(MODELS_DIR.parent)}")

    print("\nTraining completed successfully.")
    print(f"Movies processed: {len(movies):,}")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"Non-zero values: {tfidf_matrix.nnz:,}")
    print(f"Matrix storage: {tfidf_matrix.data.nbytes / (1024**2):.2f} MB")


def main() -> None:
    print("=" * 55)
    print("CONTENT-BASED MOVIE RECOMMENDATION SYSTEM")
    print("MODEL TRAINING")
    print("=" * 55)

    movies = load_features(PROCESSED_DATA_PATH)
    train_model(movies)


if __name__ == "__main__":
    main()
