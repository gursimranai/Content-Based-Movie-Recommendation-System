from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


class MovieRecommender:
    """Content-based movie recommendation engine."""

    def __init__(
        self,
        tfidf_matrix: csr_matrix,
        movies: pd.DataFrame,
    ) -> None:
        required_columns = {"id", "title"}

        if not required_columns.issubset(movies.columns):
            raise ValueError("Movie metadata must contain 'id' and 'title' columns.")

        if tfidf_matrix.shape[0] != len(movies):
            raise ValueError(
                "TF-IDF matrix and movie metadata must have equal row counts."
            )

        if movies["id"].duplicated().any():
            raise ValueError("Movie IDs must be unique.")

        self.tfidf_matrix = tfidf_matrix.tocsr()
        self.movies = movies.reset_index(drop=True).copy()

        self._title_lookup = self.movies["title"].astype(str).str.casefold()

    @classmethod
    def from_artifacts(
        cls,
        tfidf_matrix_path: str | Path,
        movie_metadata_path: str | Path,
    ) -> MovieRecommender:
        """Load a recommender from saved model artifacts."""

        tfidf_matrix = joblib.load(tfidf_matrix_path)
        movies = joblib.load(movie_metadata_path)

        return cls(
            tfidf_matrix=tfidf_matrix,
            movies=movies,
        )

    def recommend(
        self,
        movie_title: str,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """Recommend movies similar to the supplied movie title."""

        if not isinstance(movie_title, str) or not movie_title.strip():
            raise ValueError("movie_title must be a non-empty string.")

        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
            raise ValueError("top_n must be a positive integer.")

        matches = self.movies.index[
            self._title_lookup == movie_title.strip().casefold()
        ].tolist()

        if not matches:
            raise ValueError(f"Movie not found: {movie_title}")

        movie_index = matches[0]

        scores = cosine_similarity(
            self.tfidf_matrix[movie_index],
            self.tfidf_matrix,
        ).ravel()

        scores[movie_index] = -1.0

        result_count = min(top_n, len(self.movies) - 1)

        if result_count <= 0:
            return pd.DataFrame(columns=["id", "title", "similarity_score"])

        top_indices = np.argsort(scores)[::-1][:result_count]

        recommendations = self.movies.iloc[top_indices][["id", "title"]].copy()

        recommendations["similarity_score"] = scores[top_indices]

        return recommendations.reset_index(drop=True)
