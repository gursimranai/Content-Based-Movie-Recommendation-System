from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


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

        if movies.empty:
            raise ValueError("Movie metadata cannot be empty.")

        if not isinstance(tfidf_matrix, csr_matrix):
            tfidf_matrix = csr_matrix(tfidf_matrix)

        if tfidf_matrix.ndim != 2:
            raise ValueError("TF-IDF matrix must be two-dimensional.")

        if tfidf_matrix.shape[0] != len(movies):
            raise ValueError(
                "TF-IDF matrix and movie metadata must have equal row counts."
            )

        if tfidf_matrix.shape[1] == 0:
            raise ValueError("TF-IDF matrix must contain features.")

        if not np.isfinite(tfidf_matrix.data).all():
            raise ValueError("TF-IDF matrix must contain finite values.")

        movies = movies.reset_index(drop=True).copy()

        if movies["id"].isna().any():
            raise ValueError("Movie IDs cannot contain missing values.")

        if movies["title"].isna().any():
            raise ValueError("Movie titles cannot contain missing values.")

        if movies["id"].duplicated().any():
            raise ValueError("Movie IDs must be unique.")

        if (
            not movies["id"]
            .map(
                lambda value: (
                    isinstance(value, (int, np.integer))
                    and not isinstance(value, (bool, np.bool_))
                )
            )
            .all()
        ):
            raise ValueError("Movie IDs must contain integers.")

        movies["id"] = movies["id"].astype("int64")
        movies["title"] = movies["title"].astype(str).str.strip()

        if movies["title"].eq("").any():
            raise ValueError("Movie titles cannot be empty.")

        self.tfidf_matrix = tfidf_matrix
        self.movies = movies

        self._title_lookup = self.movies["title"].str.casefold()

        self._id_lookup = {
            int(movie_id): index for index, movie_id in enumerate(self.movies["id"])
        }

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

    @staticmethod
    def _validate_top_n(top_n: int) -> None:
        if (
            not isinstance(top_n, (int, np.integer))
            or isinstance(top_n, (bool, np.bool_))
            or top_n < 1
        ):
            raise ValueError("top_n must be a positive integer.")

    def _recommend_from_index(
        self,
        movie_index: int,
        top_n: int,
    ) -> pd.DataFrame:
        """Generate recommendations for a movie index."""

        scores = cosine_similarity(
            self.tfidf_matrix[movie_index],
            self.tfidf_matrix,
        ).ravel()

        scores[movie_index] = -1.0

        result_count = min(top_n, len(self.movies) - 1)

        if result_count <= 0:
            return pd.DataFrame(columns=["id", "title", "similarity_score"])

        top_indices = np.argpartition(
            scores,
            -result_count,
        )[-result_count:]

        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        recommendations = self.movies.iloc[top_indices][["id", "title"]].copy()

        recommendations["similarity_score"] = scores[top_indices]

        return recommendations.reset_index(drop=True)

    def recommend_by_id(
        self,
        movie_id: int,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """Recommend movies using a unique TMDB movie ID."""

        self._validate_top_n(top_n)

        if not isinstance(movie_id, (int, np.integer)) or isinstance(
            movie_id, (bool, np.bool_)
        ):
            raise ValueError("movie_id must be an integer.")

        movie_index = self._id_lookup.get(int(movie_id))

        if movie_index is None:
            raise ValueError(f"Movie ID not found: {movie_id}")

        return self._recommend_from_index(movie_index, top_n)

    def recommend(
        self,
        movie_title: str,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """Recommend movies using an exact, case-insensitive title."""

        self._validate_top_n(top_n)

        if not isinstance(movie_title, str) or not movie_title.strip():
            raise ValueError("movie_title must be a non-empty string.")

        normalized_title = movie_title.strip().casefold()

        matches = self.movies.index[self._title_lookup == normalized_title].tolist()

        if not matches:
            raise ValueError(f"Movie not found: {movie_title}")

        if len(matches) > 1:
            raise ValueError(
                f"Multiple movies found with title '{movie_title}'. "
                "Use recommend_by_id() to select the correct movie."
            )

        return self._recommend_from_index(matches[0], top_n)
