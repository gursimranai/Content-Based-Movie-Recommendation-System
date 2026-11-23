import pytest
import requests

from app.services import tmdb


class MockResponse:
    """Mock HTTP response for TMDB API tests."""

    def __init__(self, data: dict):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


def test_get_movie_details_success(monkeypatch):
    """Test successful retrieval of movie details."""

    expected_data = {
        "id": 19995,
        "title": "Avatar",
        "poster_path": "/avatar.jpg",
    }

    def mock_get(url, params, timeout):
        assert url.endswith("/movie/19995")
        assert params == {"api_key": "test_api_key"}
        assert timeout > 0

        return MockResponse(expected_data)

    monkeypatch.setattr(tmdb.SESSION, "get", mock_get)

    result = tmdb.get_movie_details(
        movie_id=19995,
        api_key="test_api_key",
    )

    assert result == expected_data
    assert result["title"] == "Avatar"


def test_get_movie_details_http_error(monkeypatch):
    """Test propagation of HTTP errors."""

    def mock_get(url, params, timeout):
        response = requests.Response()
        response.status_code = 404

        raise requests.HTTPError(
            "Movie not found",
            response=response,
        )

    monkeypatch.setattr(tmdb.SESSION, "get", mock_get)

    with pytest.raises(requests.HTTPError):
        tmdb.get_movie_details(
            movie_id=19995,
            api_key="test_api_key",
        )


def test_get_movie_details_connection_error(monkeypatch):
    """Test handling of connection failures."""

    def mock_get(url, params, timeout):
        raise requests.ConnectionError("Connection failed")

    monkeypatch.setattr(tmdb.SESSION, "get", mock_get)

    with pytest.raises(requests.ConnectionError):
        tmdb.get_movie_details(
            movie_id=19995,
            api_key="test_api_key",
        )


@pytest.mark.parametrize(
    "movie_id",
    [0, -1, 1.5, True, "19995", None],
)
def test_get_movie_details_invalid_movie_id(movie_id):
    """Test rejection of invalid movie IDs."""

    with pytest.raises(ValueError):
        tmdb.get_movie_details(
            movie_id=movie_id,
            api_key="test_api_key",
        )


def test_get_movie_details_missing_api_key():
    """Test rejection of an empty API key."""

    with pytest.raises(ValueError):
        tmdb.get_movie_details(
            movie_id=19995,
            api_key="",
        )


def test_build_poster_url():
    """Test construction of a valid TMDB poster URL."""

    result = tmdb.build_poster_url("/avatar.jpg")

    assert result == ("https://image.tmdb.org/t/p/w500/avatar.jpg")


def test_build_poster_url_custom_size():
    """Test poster URL construction with a custom size."""

    result = tmdb.build_poster_url(
        "/avatar.jpg",
        size="original",
    )

    assert result == ("https://image.tmdb.org/t/p/original/avatar.jpg")


@pytest.mark.parametrize(
    "poster_path",
    [None, ""],
)
def test_build_poster_url_missing_path(poster_path):
    """Test handling of missing poster paths."""

    assert tmdb.build_poster_url(poster_path) is None
