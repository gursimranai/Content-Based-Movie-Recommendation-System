import streamlit as st


def render_movie_search(movie_titles: list[str]) -> str | None:
    """Render movie search and return the selected movie title."""

    selected_movie = st.selectbox(
        "Search for a movie",
        options=movie_titles,
        index=None,
        placeholder="Type a movie title...",
    )

    return selected_movie
