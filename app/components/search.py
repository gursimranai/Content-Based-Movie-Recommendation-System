import pandas as pd
import streamlit as st


def render_movie_search(movies: pd.DataFrame) -> int | None:
    """Render a movie selector and return the selected TMDB movie ID."""

    required_columns = {"id", "title"}

    if not required_columns.issubset(movies.columns):
        raise ValueError("Movie data must contain 'id' and 'title' columns.")

    if movies.empty:
        st.warning("No movies are available.")
        return None

    movie_options = movies[["id", "title"]].copy()

    if movie_options["id"].duplicated().any():
        raise ValueError("Movie IDs must be unique.")

    title_counts = movie_options["title"].value_counts()

    movie_options["label"] = movie_options.apply(
        lambda row: (
            f"{row['title']} (TMDB ID: {row['id']})"
            if title_counts[row["title"]] > 1
            else row["title"]
        ),
        axis=1,
    )

    movie_lookup = dict(zip(movie_options["id"], movie_options["label"]))

    selected_movie_id = st.selectbox(
        "Search for a movie",
        options=movie_options["id"].tolist(),
        format_func=lambda movie_id: movie_lookup[movie_id],
        index=None,
        placeholder="Type a movie title...",
        key="movie_search_id",
    )

    return selected_movie_id
