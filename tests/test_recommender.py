import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from movie_recommender import MovieRecommender


@pytest.fixture
def recommender() -> MovieRecommender:
    movies = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "title": [
                "Action Movie",
                "Action Adventure",
                "Romantic Story",
                "Space Adventure",
            ],
        }
    )

    tfidf_matrix = csr_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.0],
            ]
        )
    )

    return MovieRecommender(
        tfidf_matrix=tfidf_matrix,
        movies=movies,
    )


def test_recommend_returns_requested_number(recommender):
    result = recommender.recommend("Action Movie", top_n=2)

    assert len(result) == 2


def test_recommend_excludes_original_movie(recommender):
    result = recommender.recommend("Action Movie", top_n=3)

    assert 1 not in result["id"].values


def test_recommend_returns_expected_columns(recommender):
    result = recommender.recommend("Action Movie")

    assert list(result.columns) == [
        "id",
        "title",
        "similarity_score",
    ]


def test_recommend_is_case_insensitive(recommender):
    result = recommender.recommend("action movie")

    assert not result.empty


def test_recommend_rejects_unknown_movie(recommender):
    with pytest.raises(ValueError, match="Movie not found"):
        recommender.recommend("Unknown Movie")


@pytest.mark.parametrize("top_n", [0, -1, 1.5, True])
def test_recommend_rejects_invalid_top_n(recommender, top_n):
    with pytest.raises(ValueError, match="top_n"):
        recommender.recommend("Action Movie", top_n=top_n)


@pytest.mark.parametrize("movie_title", ["", " ", None, 123])
def test_recommend_rejects_invalid_movie_title(
    recommender,
    movie_title,
):
    with pytest.raises(ValueError, match="movie_title"):
        recommender.recommend(movie_title)
