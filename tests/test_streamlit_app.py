from pathlib import Path

import pandas as pd
import streamlit as st
import pytest
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

    def recommend(self, movie_title, top_n=5):
        if movie_title.casefold() != "avatar":
            raise ValueError("Unknown movie")

        recommendations = pd.DataFrame(
            {
                "id": [679, 348],
                "title": ["Aliens", "Alien"],
                "similarity_score": [0.42, 0.36],
            }
        )

        return recommendations.head(top_n)


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

    assert app.session_state["selected_movie"] == "Avatar"

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
    assert app.session_state["selected_movie"] == "Avatar"
    assert app.session_state["top_n"] == 10


def test_app_handles_missing_tmdb_key(fake_app_dependencies):
    """The app should work without TMDB credentials."""

    app = create_app().run()

    app.selectbox[0].set_value("Avatar")
    app.button[0].click().run()

    assert not app.exception

    assert any("TMDB details are unavailable" in element.value for element in app.info)
