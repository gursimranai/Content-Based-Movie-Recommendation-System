import requests

TMDB_API_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/"

SESSION = requests.Session()


def get_movie_details(movie_id: int, api_key: str) -> dict:
    """Retrieve movie details from TMDB."""

    if not isinstance(movie_id, int) or isinstance(movie_id, bool):
        raise ValueError("movie_id must be an integer.")

    if movie_id <= 0:
        raise ValueError("movie_id must be positive.")

    if not api_key:
        raise ValueError("TMDB API key is required.")

    response = SESSION.get(
        f"{TMDB_API_BASE_URL}/movie/{movie_id}",
        params={"api_key": api_key},
        timeout=15,
    )

    response.raise_for_status()

    return response.json()


def build_poster_url(
    poster_path: str | None,
    size: str = "w500",
) -> str | None:
    """Construct a TMDB image URL."""

    if not poster_path:
        return None

    return f"{TMDB_IMAGE_BASE_URL}{size}{poster_path}"
