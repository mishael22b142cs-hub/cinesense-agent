import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ---------- Load + merge ----------
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")
df = movies.merge(credits[["movie_id", "cast", "crew"]], left_on="id", right_on="movie_id")


def names(text, limit=None):
    items = [x["name"] for x in json.loads(text)]
    return items[:limit] if limit else items


def director(text):
    for person in json.loads(text):
        if person["job"] == "Director":
            return person["name"]
    return ""


df["genre_list"] = df["genres"].apply(names)
df["keyword_list"] = df["keywords"].apply(names)
df["top_cast"] = df["cast"].apply(lambda t: names(t, 3))
df["director"] = df["crew"].apply(director)
df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
df["overview"] = df["overview"].fillna("")


def make_tags(row):
    words = row["overview"].split()
    for group in [row["genre_list"], row["keyword_list"], row["top_cast"], [row["director"]]]:
        words += [w.replace(" ", "") for w in group if w]
    return " ".join(words).lower()


df["tags"] = df.apply(make_tags, axis=1)

# ---------- 1. Recommender ----------
cv = CountVectorizer(max_features=5000, stop_words="english")   # 🔴 text → numbers
vectors = cv.fit_transform(df["tags"])                           # 🔴

catalog = df[["id", "title", "year", "genre_list", "vote_average", "vote_count",
              "overview", "director", "top_cast"]].reset_index(drop=True)

joblib.dump(catalog, "models/catalog.pkl")                       # 🔴 model save
joblib.dump(vectors, "models/vectors.pkl")
print(f"Recommender ready: {len(catalog)} movies")

# ---------- 2. Rating predictor ----------
train = df[(df["budget"] > 0) & (df["runtime"] > 0) & (df["vote_count"] >= 50)].dropna(subset=["year"]).copy()
train["main_genre"] = train["genre_list"].apply(lambda g: g[0] if g else "Unknown")
train["budget_millions"] = train["budget"] / 1_000_000

FEATURES = ["budget_millions", "runtime", "year", "main_genre"]
X = train[FEATURES]
y = train["vote_average"]

preprocess = ColumnTransformer(                                                   # 🔴
    [("genre", OneHotEncoder(handle_unknown="ignore"), ["main_genre"])],
    remainder="passthrough",
)
model = Pipeline([                                                                # 🔴
    ("prep", preprocess),
    ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(f"Rating model  R²: {r2_score(y_test, pred):.2f}   MAE: {mean_absolute_error(y_test, pred):.2f}")
joblib.dump(model, "models/rating_model.pkl")
print("All models saved in models/")