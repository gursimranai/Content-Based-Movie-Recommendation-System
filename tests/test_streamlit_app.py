from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from movie_recommender import MovieRecommender

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app" / "streamlit_app.py"


class FakeRecommender:
    """A predictable recommendation model for UI tests."""

    movies = pd.DataFrame(
        {
            "id": [19995, 679, 348],
            "title": ["Avatar", "Aliens", "Alien"],
        }
    )

    def recommend_by_id(self, movie_id, top_n=5):
        """Return predictable recommendations using a movie ID."""

        recommendations = {
            19995: [
                {
                    "id": 679,
                    "title": "Aliens",
                    "similarity_score": 0.42,
                },
                {
                    "id": 348,
                    "title": "Alien",
                    "similarity_score": 0.36,
                },
            ],
            679: [
                {
                    "id": 19995,
                    "title": "Avatar",
                    "similarity_score": 0.42,
                },
                {
                    "id": 348,
                    "title": "Alien",
                    "similarity_score": 0.36,
                },
            ],
            348: [
                {
                    "id": 679,
                    "title": "Aliens",
                    "similarity_score": 0.42,
                },
                {
                    "id": 19995,
                    "title": "Avatar",
                    "similarity_score": 0.36,
                },
            ],
        }

        return pd.DataFrame(
            recommendations.get(movie_id, [])[:top_n],
            columns=["id", "title", "similarity_score"],
        )


@pytest.fixture
def fake_app_dependencies(monkeypatch):
    """Replace model loading and external API dependencies."""

    monkeypatch.setattr(
        MovieRecommender,
        "from_artifacts",
        classmethod(lambda cls, **kwargs: FakeRecommender()),
    )

    monkeypatch.setenv("TMDB_API_KEY", "")

    st.cache_resource.clear()
    st.cache_data.clear()

    yield

    st.cache_resource.clear()
    st.cache_data.clear()


def create_app():
    """Create a Streamlit application test instance."""

    assert APP_PATH.is_file(), f"Application file not found: {APP_PATH}"

    return AppTest.from_file(APP_PATH)


def test_app_loads_without_errors(fake_app_dependencies):
    """The application should render without exceptions."""

    app = create_app().run()

    assert not app.exception
    assert len(app.selectbox) == 2

    assert app.button[0].label == "Get Recommendations"


def test_recommendation_workflow(fake_app_dependencies):
    """Selecting Avatar should display its recommendations."""

    app = create_app().run()

    app.selectbox[0].set_value("Avatar").run()
    app.button[0].click().run()

    assert not app.exception

    assert app.session_state["selected_movie_id"] == 19995
    assert app.session_state["top_n"] == 5

    assert any("Recommended movies" in element.value for element in app.markdown)

    assert any("Aliens" in element.value for element in app.markdown)
    assert any("Alien" in element.value for element in app.markdown)


def test_recommendation_count_selection(fake_app_dependencies):
    """The selected recommendation count should be stored."""

    app = create_app().run()

    app.selectbox[0].set_value("Avatar")
    app.selectbox[1].set_value(10)

    app.button[0].click().run()

    assert not app.exception
    assert app.session_state["selected_movie_id"] == 19995
    assert app.session_state["top_n"] == 10


def test_app_handles_missing_tmdb_key(fake_app_dependencies):
    """The app should work without TMDB credentials."""

    app = create_app().run()

    app.selectbox[0].set_value("Avatar")
    app.button[0].click().run()

    assert not app.exception

    assert any("TMDB details are unavailable" in element.value for element in app.info)
