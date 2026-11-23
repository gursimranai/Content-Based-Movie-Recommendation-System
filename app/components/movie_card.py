import streamlit as st


def render_movie_card(
    title: str,
    movie_id: int,
    similarity_score: float,
    poster_url: str | None = None,
    overview: str | None = None,
    release_year: str | None = None,
) -> None:
    """Render an individual movie recommendation card."""

    with st.container(border=True):
        if poster_url:
            st.image(
                poster_url,
                use_container_width=True,
            )
        else:
            st.markdown(
                """
                <div style="
                    height: 220px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: #202230;
                    border-radius: 8px;
                    font-size: 48px;
                ">🎬</div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(f"**{title}**")

        if release_year:
            st.caption(f"Release year: {release_year}")

        st.caption(f"TMDB ID: {movie_id}")

        st.markdown("**Cosine similarity**")

        score = max(0.0, min(1.0, float(similarity_score)))

        st.progress(score)

        st.caption(f"Score: {similarity_score:.4f}")

        if overview:
            with st.expander("Movie overview"):
                st.write(overview)
