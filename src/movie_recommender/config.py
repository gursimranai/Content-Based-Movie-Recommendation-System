from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "movies_features.csv"

MODELS_DIR = PROJECT_ROOT / "models"

TFIDF_MATRIX_PATH = MODELS_DIR / "tfidf_matrix.joblib"
MOVIE_METADATA_PATH = MODELS_DIR / "movies_metadata.joblib"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
MOVIE_FEATURES_PATH = MODELS_DIR / "movie_features.joblib"
