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


def test_recommend_by_id_returns_recommendations(recommender):
    movie_id = recommender.movies.iloc[0]["id"]

    result = recommender.recommend_by_id(
        int(movie_id),
        top_n=3,
    )

    assert len(result) == 3
    assert movie_id not in result["id"].tolist()
    assert list(result.columns) == [
        "id",
        "title",
        "similarity_score",
    ]


def test_recommend_by_id_rejects_unknown_id(recommender):
    with pytest.raises(ValueError, match="Movie ID not found"):
        recommender.recommend_by_id(999999)


def test_recommend_by_id_rejects_invalid_id(recommender):
    for movie_id in ["19995", None, 1.5, True]:
        with pytest.raises(ValueError, match="movie_id must be an integer"):
            recommender.recommend_by_id(movie_id)


def test_recommend_rejects_duplicate_titles():
    movies = pd.DataFrame(
        {
            "id": [101, 102, 103],
            "title": ["King Kong", "King Kong", "Avatar"],
        }
    )

    matrix = csr_matrix(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )

    model = MovieRecommender(matrix, movies)

    with pytest.raises(ValueError, match="Multiple movies found"):
        model.recommend("King Kong")


def test_recommend_by_id_handles_duplicate_titles():
    movies = pd.DataFrame(
        {
            "id": [101, 102, 103],
            "title": ["King Kong", "King Kong", "Avatar"],
        }
    )

    matrix = csr_matrix(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )

    model = MovieRecommender(matrix, movies)

    result = model.recommend_by_id(102, top_n=1)

    assert len(result) == 1
    assert result.iloc[0]["id"] == 103
