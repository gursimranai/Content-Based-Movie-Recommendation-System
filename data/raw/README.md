# 🎬 TMDB 5000 Movie Dataset

This directory contains the **TMDB 5000 Movie Dataset**, a collection of movie metadata used for data exploration, preprocessing, feature engineering, and developing the Movie Recommendation System.

The dataset provides information about movies, including titles, genres, keywords, cast, crew, production details, and plot overviews.

## 📦 Dataset Files

| File | Description |
|---|---|
| `tmdb_5000_movies.csv` | Contains movie metadata, including titles, genres, keywords, overview, popularity, ratings, and release information. |
| `tmdb_5000_credits.csv` | Contains cast and crew information associated with movies. |

## 📊 Dataset Overview

### 1. Movies Dataset

**File:** `tmdb_5000_movies.csv`

Contains movie-level information and metadata.

Important columns include:

| Column | Description |
|---|---|
| `id` | Unique movie identifier |
| `title` | Movie title |
| `original_title` | Original movie title |
| `overview` | Movie storyline or description |
| `genres` | Movie genre information |
| `keywords` | Keywords associated with the movie |
| `popularity` | Movie popularity score |
| `vote_average` | Average audience rating |
| `vote_count` | Number of audience votes |
| `release_date` | Movie release date |
| `runtime` | Movie duration |
| `budget` | Movie production budget |
| `revenue` | Movie revenue |

### 2. Credits Dataset

**File:** `tmdb_5000_credits.csv`

Contains information about the people involved in movie production.

Important columns include:

| Column | Description |
|---|---|
| `movie_id` | Movie identifier used to connect with the movies dataset |
| `title` | Movie title |
| `cast` | Information about movie actors |
| `crew` | Information about directors and other production crew members |

## 🔗 Dataset Relationship

The two datasets can be combined using their movie identifiers:

```python
import pandas as pd

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

movies = movies.merge(credits, left_on="id", right_on="movie_id", how="inner")
```

This creates a combined dataset containing movie metadata, cast, and crew information.

## 🧠 Usage in the Recommendation System

These datasets are primarily used to develop a **Content-Based Movie Recommendation System**.

The following information can be extracted and processed:

- Movie genres
- Keywords
- Movie overview
- Cast members
- Director information

These attributes can be combined into textual features and processed using:

1. Data cleaning and preprocessing
2. Feature extraction
3. Text vectorization using TF-IDF
4. Cosine similarity calculation
5. Movie recommendation generation

## 📥 Dataset Source

The dataset is available through Kaggle:

**TMDB 5000 Movie Dataset**

[Download Dataset – Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

## ⚠️ Dataset Notes

- Keep the original CSV files unchanged inside the raw data directory.
- Perform cleaning and transformations on separate copies.
- Preserve movie identifiers when merging datasets.
- Check for missing values and duplicate records before processing.
- Avoid committing large generated model artifacts or unnecessary intermediate files.

## 📁 Directory Structure

```text
data/
└── raw/
    ├── README.md
    ├── tmdb_5000_movies.csv
    └── tmdb_5000_credits.csv
```

---

**Purpose:** Movie data exploration, preprocessing, feature engineering, and content-based recommendation development.

Part of the [Movie Recommendation System](https://github.com/gursimranai/Movie-Recommendation-System) project.