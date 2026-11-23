import logging
import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

from app.components.movie_card import render_movie_card
from app.components.search import render_movie_search
from app.services.tmdb import build_poster_url, get_movie_details

from movie_recommender import MovieRecommender
from movie_recommender.config import (
    MOVIE_METADATA_PATH,
    TFIDF_MATRIX_PATH,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSS_PATH = PROJECT_ROOT / "app" / "styles" / "main.css"

load_dotenv(PROJECT_ROOT / ".env")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")


# --------------------------------------------------
# TMDB service
# --------------------------------------------------


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_tmdb_details(movie_id: int) -> dict | None:
    """Fetch and cache movie details from TMDB."""

    if not TMDB_API_KEY:
        logger.warning(
            "TMDB API key is not configured. Skipping details for movie ID %s.",
            movie_id,
        )
        return None

    try:
        return get_movie_details(
            movie_id=movie_id,
            api_key=TMDB_API_KEY,
        )

    except requests.RequestException:
        logger.exception(
            "TMDB request failed for movie ID %s.",
            movie_id,
        )
        return None

    except (ValueError, TypeError):
        logger.exception(
            "Invalid TMDB request for movie ID %s.",
            movie_id,
        )
        return None


# --------------------------------------------------
# Application utilities
# --------------------------------------------------


def load_css() -> None:
    """Load the custom application stylesheet."""

    if not CSS_PATH.exists():
        logger.warning("CSS stylesheet not found: %s", CSS_PATH)
        return

    try:
        css = CSS_PATH.read_text(encoding="utf-8")

    except OSError:
        logger.exception("Failed to read application stylesheet.")
        return

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_recommender() -> MovieRecommender:
    """Load the recommendation engine once per application session."""

    logger.info("Loading recommendation model artifacts.")

    recommender = MovieRecommender.from_artifacts(
        tfidf_matrix_path=TFIDF_MATRIX_PATH,
        movie_metadata_path=MOVIE_METADATA_PATH,
    )

    logger.info("Recommendation model loaded successfully.")

    return recommender


def display_selected_movie(
    movie_title: str,
    movie_id: int,
) -> None:
    """Display details for the selected movie."""

    details = fetch_tmdb_details(movie_id)

    st.markdown("---")
    st.markdown("### Selected movie")

    if not details:
        st.subheader(movie_title)
        st.info("TMDB details are unavailable. Recommendations can still be generated.")
        return

    poster_url = build_poster_url(details.get("poster_path"))

    col1, col2 = st.columns([1, 3])

    with col1:
        if poster_url:
            st.image(
                poster_url,
                use_container_width=True,
            )
        else:
            st.info("Poster unavailable.")

    with col2:
        st.subheader(details.get("title") or movie_title)

        st.caption(f"Release date: {details.get('release_date') or 'Unknown'}")

        st.write(details.get("overview") or "No overview available.")


def display_recommendations(
    recommender: MovieRecommender,
    movie_title: str,
    recommendation_count: int,
) -> None:
    """Generate and display movie recommendations."""

    try:
        recommendations = recommender.recommend(
            movie_title,
            top_n=recommendation_count,
        )

    except ValueError as error:
        logger.warning(
            "Could not generate recommendations for %s: %s",
            movie_title,
            error,
        )
        st.error(str(error))
        return

    except Exception:
        logger.exception(
            "Unexpected recommendation failure for movie %s.",
            movie_title,
        )
        st.error(
            "An unexpected error occurred while generating "
            "recommendations. Check the application logs."
        )
        return

    st.markdown(
        '<div class="section-heading">Recommended movies</div>',
        unsafe_allow_html=True,
    )

    if recommendations.empty:
        st.info("No recommendations found.")
        return

    movies_per_row = 5

    for start in range(
        0,
        len(recommendations),
        movies_per_row,
    ):
        batch = recommendations.iloc[start : start + movies_per_row]

        columns = st.columns(movies_per_row)

        for column, (_, movie) in zip(
            columns,
            batch.iterrows(),
        ):
            movie_id = int(movie["id"])
            details = fetch_tmdb_details(movie_id)

            poster_url = (
                build_poster_url(details.get("poster_path")) if details else None
            )

            release_date = details.get("release_date") if details else None

            release_year = (
                release_date[:4] if release_date and len(release_date) >= 4 else None
            )

            overview = details.get("overview") if details else None

            with column:
                render_movie_card(
                    title=str(movie["title"]),
                    movie_id=movie_id,
                    similarity_score=float(movie["similarity_score"]),
                    poster_url=poster_url,
                    overview=overview,
                    release_year=release_year,
                )


def display_model_information() -> None:
    """Display information about the recommendation model."""

    st.markdown("---")

    st.markdown(
        """
        <div class="model-panel">
            <div class="section-heading">Model information</div>
            <p class="model-label">Recommendation approach</p>
            <p class="model-value">Content-Based Filtering</p>
            <p class="model-label">Text representation</p>
            <p class="model-value">TF-IDF Vectorization</p>
            <p class="model-label">Similarity metric</p>
            <p class="model-value">Cosine Similarity</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Built with Python, Pandas, Scikit-learn and Streamlit.")

    st.markdown("---")

    st.caption(
        "This product uses the TMDB API but is not endorsed or certified by TMDB."
    )


# --------------------------------------------------
# Main application
# --------------------------------------------------


def main() -> None:
    """Run the Streamlit application."""

    st.set_page_config(
        page_title="Movie Recommendation System",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    load_css()

    st.markdown(
        '<div class="main-heading">Movie Recommendation System</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Discover movies using content-based machine learning."
        "</div>",
        unsafe_allow_html=True,
    )

    # Load recommendation engine.
    try:
        recommender = load_recommender()

    except FileNotFoundError:
        logger.exception("Recommendation model artifacts are missing.")
        st.error(
            "Model files are missing. Please generate the model "
            "artifacts by running Notebook 05."
        )
        st.stop()

    except Exception:
        logger.exception("Application failed to initialize the recommendation engine.")
        st.error(
            "The recommendation engine could not be initialized. "
            "Check the application logs for details."
        )
        st.stop()

    movie_titles = (
        recommender.movies["title"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    st.markdown(
        '<div class="section-heading">Find similar movies</div>',
        unsafe_allow_html=True,
    )

    selected_movie = render_movie_search(movie_titles)

    col1, col2 = st.columns([1, 4])

    with col1:
        top_n = st.selectbox(
            "Number of recommendations",
            options=[5, 10, 15, 20],
            index=0,
        )

    with col2:
        st.markdown(
            "<div style='padding-top:27px;color:#9295a4'>"
            "Choose a movie and discover similar titles."
            "</div>",
            unsafe_allow_html=True,
        )

    if st.button(
        "Get Recommendations",
        use_container_width=True,
    ):
        if not selected_movie:
            st.warning("Please select a movie first.")

        else:
            st.session_state["selected_movie"] = selected_movie
            st.session_state["top_n"] = top_n

            logger.info(
                "User requested recommendations for %s (top_n=%s).",
                selected_movie,
                top_n,
            )

    # Display results after the user requests recommendations.
    if "selected_movie" in st.session_state:
        movie_title = st.session_state["selected_movie"]
        recommendation_count = st.session_state["top_n"]

        selected_rows = recommender.movies[
            recommender.movies["title"].astype(str).str.strip().str.casefold()
            == movie_title.strip().casefold()
        ]

        if selected_rows.empty:
            st.error(f"Movie not found: {movie_title}")
            st.stop()

        selected = selected_rows.iloc[0]
        selected_id = int(selected["id"])

        display_selected_movie(
            movie_title=movie_title,
            movie_id=selected_id,
        )

        display_recommendations(
            recommender=recommender,
            movie_title=movie_title,
            recommendation_count=recommendation_count,
        )

    display_model_information()


if __name__ == "__main__":
    main()
