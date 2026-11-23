import joblib
import pandas as pd
import pytest

import scripts.train_model as training


@pytest.fixture
def sample_movies():
    return pd.DataFrame(
        {
            "id": [101, 102, 103, 104],
            "title": [
                "Space Adventure",
                "Ocean Mystery",
                "Mountain Rescue",
                "Desert Journey",
            ],
            "tags": [
                "space adventure science fiction",
                "ocean mystery underwater",
                "mountain rescue survival",
                "desert journey adventure",
            ],
        }
    )


def test_load_features_success(tmp_path, sample_movies):
    data_path = tmp_path / "movies_features.csv"
    sample_movies.to_csv(data_path, index=False)

    result = training.load_features(data_path)

    assert len(result) == 4
    assert list(result.columns) == ["id", "title", "tags"]
    assert result["id"].is_unique


def test_load_features_missing_file(tmp_path):
    data_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        training.load_features(data_path)


def test_load_features_missing_columns(tmp_path):
    data_path = tmp_path / "invalid.csv"

    pd.DataFrame(
        {
            "id": [101],
            "title": ["Space Adventure"],
        }
    ).to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        training.load_features(data_path)


def test_load_features_duplicate_ids(tmp_path):
    data_path = tmp_path / "duplicates.csv"

    pd.DataFrame(
        {
            "id": [101, 101],
            "title": ["Space Adventure", "Ocean Mystery"],
            "tags": ["space adventure", "ocean mystery"],
        }
    ).to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="duplicate movie IDs"):
        training.load_features(data_path)


def test_load_features_removes_invalid_rows(tmp_path):
    data_path = tmp_path / "invalid_rows.csv"

    pd.DataFrame(
        {
            "id": [101, 102, 103],
            "title": ["Space Adventure", None, "   "],
            "tags": ["space adventure", "ocean mystery", "   "],
        }
    ).to_csv(data_path, index=False)

    result = training.load_features(data_path)

    assert len(result) == 1
    assert result.iloc[0]["title"] == "Space Adventure"


def test_load_features_rejects_empty_dataset(tmp_path):
    data_path = tmp_path / "empty.csv"

    pd.DataFrame(
        {
            "id": [101],
            "title": [None],
            "tags": ["space adventure"],
        }
    ).to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="No valid movie records"):
        training.load_features(data_path)


def test_load_features_warns_about_duplicate_titles(
    tmp_path,
    capsys,
):
    data_path = tmp_path / "duplicate_titles.csv"

    pd.DataFrame(
        {
            "id": [101, 102],
            "title": ["King Kong", "King Kong"],
            "tags": ["giant ape adventure", "giant ape action"],
        }
    ).to_csv(data_path, index=False)

    training.load_features(data_path)

    captured = capsys.readouterr()

    assert "Duplicate movie titles" in captured.out


def test_train_model_saves_artifacts(
    tmp_path,
    sample_movies,
    monkeypatch,
):
    models_dir = tmp_path / "models"

    monkeypatch.setattr(training, "MODELS_DIR", models_dir)
    monkeypatch.setattr(
        training,
        "TFIDF_MATRIX_PATH",
        models_dir / "tfidf_matrix.joblib",
    )
    monkeypatch.setattr(
        training,
        "TFIDF_VECTORIZER_PATH",
        models_dir / "tfidf_vectorizer.joblib",
    )
    monkeypatch.setattr(
        training,
        "MOVIE_METADATA_PATH",
        models_dir / "movies_metadata.joblib",
    )
    monkeypatch.setattr(
        training,
        "MOVIE_FEATURES_PATH",
        models_dir / "movie_features.joblib",
    )

    training.train_model(sample_movies)

    assert models_dir.is_dir()

    tfidf_matrix = joblib.load(models_dir / "tfidf_matrix.joblib")
    vectorizer = joblib.load(models_dir / "tfidf_vectorizer.joblib")
    metadata = joblib.load(models_dir / "movies_metadata.joblib")
    features = joblib.load(models_dir / "movie_features.joblib")

    assert tfidf_matrix.shape[0] == len(sample_movies)
    assert tfidf_matrix.shape[1] > 0
    assert vectorizer.vocabulary_
    assert list(metadata.columns) == ["id", "title"]
    assert len(metadata) == len(sample_movies)
    assert len(features) == len(sample_movies)

    assert metadata["id"].tolist() == sample_movies["id"].tolist()
