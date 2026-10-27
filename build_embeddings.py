"""
Builds the real vector-search index for CineSense Agent's recommender, replacing the
CountVectorizer + cosine_similarity approach in build_models.py (kept for comparison)
with:
  - sentence-transformers (all-MiniLM-L6-v2) semantic embeddings of each movie's
    overview/genres/keywords/cast/director, as a natural-language sentence
  - FAISS (IndexFlatIP over L2-normalized vectors = cosine similarity, exact search)
    as the actual vector database

Run once to produce models/catalog.pkl, models/embeddings.npy, models/movie_index.faiss,
which step3_agent.py loads at startup instead of vectors.pkl.
"""
import json

import joblib
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"


def names(text, limit=None):
    items = [x["name"] for x in json.loads(text)]
    return items[:limit] if limit else items


def director(text):
    for person in json.loads(text):
        if person["job"] == "Director":
            return person["name"]
    return ""


def build_catalog():
    movies = pd.read_csv("data/tmdb_5000_movies.csv")
    credits = pd.read_csv("data/tmdb_5000_credits.csv")
    df = movies.merge(credits[["movie_id", "cast", "crew"]], left_on="id", right_on="movie_id")

    df["genre_list"] = df["genres"].apply(names)
    df["keyword_list"] = df["keywords"].apply(names)
    df["top_cast"] = df["cast"].apply(lambda t: names(t, 3))
    df["director"] = df["crew"].apply(director)
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["overview"] = df["overview"].fillna("")

    def make_tags(row):
        parts = [row["overview"]]
        if row["genre_list"]:
            parts.append("Genres: " + ", ".join(row["genre_list"]) + ".")
        if row["keyword_list"]:
            parts.append("Themes: " + ", ".join(row["keyword_list"][:10]) + ".")
        if row["top_cast"]:
            parts.append("Starring " + ", ".join(row["top_cast"]) + ".")
        if row["director"]:
            parts.append("Directed by " + row["director"] + ".")
        return " ".join(p for p in parts if p)

    df["tags"] = df.apply(make_tags, axis=1)

    catalog = df[["id", "title", "year", "genre_list", "vote_average", "vote_count",
                  "overview", "director", "top_cast", "tags"]].reset_index(drop=True)
    return catalog


def main():
    catalog = build_catalog()
    print(f"Loaded {len(catalog)} movies")

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        catalog["tags"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    joblib.dump(catalog.drop(columns=["tags"]), "models/catalog.pkl")
    np.save("models/embeddings.npy", embeddings)
    faiss.write_index(index, "models/movie_index.faiss")

    print(f"Vector index ready: {index.ntotal} movies, dim={embeddings.shape[1]} ({EMBED_MODEL})")
    print("Saved models/catalog.pkl, models/embeddings.npy, models/movie_index.faiss")


if __name__ == "__main__":
    main()
