from movie_recommender import MovieRecommender
from movie_recommender.config import (
    MOVIE_METADATA_PATH,
    TFIDF_MATRIX_PATH,
)


def main() -> None:
    recommender = MovieRecommender.from_artifacts(
        tfidf_matrix_path=TFIDF_MATRIX_PATH,
        movie_metadata_path=MOVIE_METADATA_PATH,
    )

    recommendations = recommender.recommend(
        movie_title="Avatar",
        top_n=5,
    )

    print("\nRecommendations for Avatar:\n")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
