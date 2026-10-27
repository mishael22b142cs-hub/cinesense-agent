"""
Trains a small PyTorch MLP to predict a movie's rating — a from-scratch neural-net
alternative to the Random Forest in build_models.py (kept as the baseline comparison;
this script prints both R²/MAE on the same train/test split).

Same features as the original model: budget_millions, runtime, year, main_genre.

Produces:
  models/rating_model_torch.pt        (PyTorch state_dict)
  models/rating_feature_pipeline.pkl  (sklearn ColumnTransformer: scaler + genre one-hot)
"""
import json

import joblib
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ["budget_millions", "runtime", "year", "main_genre"]
EPOCHS = 300
LR = 1e-3


def load_training_data():
    movies = pd.read_csv("data/tmdb_5000_movies.csv")
    credits = pd.read_csv("data/tmdb_5000_credits.csv")
    df = movies.merge(credits[["movie_id", "cast", "crew"]], left_on="id", right_on="movie_id")

    df["genre_list"] = df["genres"].apply(lambda t: [g["name"] for g in json.loads(t)])
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    train = df[(df["budget"] > 0) & (df["runtime"] > 0) & (df["vote_count"] >= 50)].dropna(subset=["year"]).copy()
    train["main_genre"] = train["genre_list"].apply(lambda g: g[0] if g else "Unknown")
    train["budget_millions"] = train["budget"] / 1_000_000
    return train[FEATURES], train["vote_average"]


class RatingMLP(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    X, y = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), ["budget_millions", "runtime", "year"]),
        ("genre", OneHotEncoder(handle_unknown="ignore"), ["main_genre"]),
    ])
    X_train_enc = preprocess.fit_transform(X_train).astype("float32")
    X_test_enc = preprocess.transform(X_test).astype("float32")
    if hasattr(X_train_enc, "toarray"):
        X_train_enc, X_test_enc = X_train_enc.toarray(), X_test_enc.toarray()

    X_train_t = torch.from_numpy(X_train_enc)
    y_train_t = torch.from_numpy(y_train.to_numpy(dtype="float32"))
    X_test_t = torch.from_numpy(X_test_enc)

    model = RatingMLP(n_inputs=X_train_enc.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"epoch {epoch:4d}  train MSE {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        torch_pred = model(X_test_t).numpy()
    torch_r2 = r2_score(y_test, torch_pred)
    torch_mae = mean_absolute_error(y_test, torch_pred)

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train_enc, y_train)
    rf_pred = rf.predict(X_test_enc)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)

    print(f"\nPyTorch MLP   R2: {torch_r2:.3f}   MAE: {torch_mae:.3f}")
    print(f"Random Forest R2: {rf_r2:.3f}   MAE: {rf_mae:.3f}")

    torch.save(model.state_dict(), "models/rating_model_torch.pt")
    joblib.dump({"preprocess": preprocess, "n_inputs": X_train_enc.shape[1]},
                "models/rating_feature_pipeline.pkl")
    print("\nSaved models/rating_model_torch.pt, models/rating_feature_pipeline.pkl")


if __name__ == "__main__":
    main()
