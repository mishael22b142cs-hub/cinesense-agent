import json
import joblib
import pandas as pd
import numpy as np
import faiss
import torch
from dotenv import load_dotenv
from groq import Groq

from train_rating_model import RatingMLP

load_dotenv()
client = Groq()
MODEL = "openai/gpt-oss-120b"
EMBED_MODEL = "all-MiniLM-L6-v2"

# ================= LOAD MODELS (once, at startup) =================
# Real vector search: sentence-transformer embeddings + FAISS, instead of the old
# CountVectorizer + cosine_similarity (still in build_models.py for comparison).
catalog = joblib.load("models/catalog.pkl")
vector_index = faiss.read_index("models/movie_index.faiss")
embeddings = np.load("models/embeddings.npy")

# PyTorch MLP rating predictor (see train_rating_model.py), with its sklearn
# preprocessing pipeline (scaler + genre one-hot) saved alongside it.
rating_pipeline = joblib.load("models/rating_feature_pipeline.pkl")
rating_model = RatingMLP(n_inputs=rating_pipeline["n_inputs"])
rating_model.load_state_dict(torch.load("models/rating_model_torch.pt", map_location="cpu"))
rating_model.eval()


# ================= HELPERS =================
def find_index(title):
    t = title.lower()
    matches = catalog[catalog["title"].str.lower() == t]
    if matches.empty:
        matches = catalog[catalog["title"].str.lower().str.contains(t, regex=False)]
    if matches.empty:
        return None
    return matches.sort_values("vote_count", ascending=False).index[0]


def movie_card(i):
    row = catalog.loc[i]
    return {
        "title": row["title"],
        "year": None if pd.isna(row["year"]) else int(row["year"]),
        "genres": row["genre_list"],
        "rating": float(row["vote_average"]),
        "director": row["director"],
    }


# ================= TOOLS =================
def search_movie(title):
    i = find_index(title)
    if i is None:
        return f"'{title}' not found in the TMDB 5000 dataset."
    card = movie_card(i)
    card["cast"] = catalog.loc[i, "top_cast"]
    card["overview"] = catalog.loc[i, "overview"][:300]
    return json.dumps(card)


def recommend_similar(title, n=5, min_rating=None):
    i = find_index(title)
    if i is None:
        return f"'{title}' not found in the TMDB 5000 dataset."
    n = int(n or 5)
    k = 300 if min_rating is not None else n + 1   # wider pool to filter, like the old version
    query_vector = embeddings[i:i + 1]              # already L2-normalized
    _, indices = vector_index.search(query_vector, min(k + 1, vector_index.ntotal))
    ranked = [int(j) for j in indices[0] if j != i]
    if min_rating is not None:
        ranked = [j for j in ranked
                  if catalog.loc[j, "vote_average"] >= float(min_rating)
                  and catalog.loc[j, "vote_count"] >= 100]
    best = ranked[:n]
    if not best:
        return f"No movies similar to '{title}' found with rating >= {min_rating}."
    return json.dumps([movie_card(j) for j in best])


def top_movies_by_genre(genre, min_votes=500, n=5):
    g = genre.lower()
    mask = catalog["genre_list"].apply(lambda lst: g in [x.lower() for x in lst])
    result = catalog[mask & (catalog["vote_count"] >= int(min_votes or 500))]
    result = result.sort_values("vote_average", ascending=False).head(int(n or 5))
    if result.empty:
        return f"No '{genre}' movies found."
    return json.dumps([movie_card(i) for i in result.index])


def movies_by_director(name, n=5):
    d = name.lower()
    result = catalog[catalog["director"].str.lower().str.contains(d, regex=False)]
    result = result.sort_values("vote_average", ascending=False).head(int(n or 5))
    if result.empty:
        return f"No {name} movies found."
    return json.dumps([movie_card(i) for i in result.index])


def predict_rating(budget_millions, runtime, genre, year=2026):
    X = pd.DataFrame([{
        "budget_millions": float(budget_millions),
        "runtime": float(runtime),
        "year": float(year or 2026),
        "main_genre": genre.title(),
    }])
    X_enc = rating_pipeline["preprocess"].transform(X).astype("float32")
    if hasattr(X_enc, "toarray"):
        X_enc = X_enc.toarray()
    with torch.no_grad():
        pred = rating_model(torch.from_numpy(X_enc)).item()
    return json.dumps({"predicted_rating": round(float(pred), 1),
                       "note": "Estimate from a small PyTorch MLP trained on TMDB data."})


AVAILABLE_TOOLS = {
    "search_movie": search_movie,
    "recommend_similar": recommend_similar,
    "top_movies_by_genre": top_movies_by_genre,
    "predict_rating": predict_rating,
    "movies_by_director": movies_by_director,
}

# ================= SCHEMAS =================
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_movie",
            "description": "Get details of a specific movie: year, genres, rating, director, cast, overview.",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string", "description": "Movie title"}},
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_similar",
            "description": "Recommend movies similar to a given movie, based on story, genres, keywords, cast and director. Can filter by minimum rating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Movie the user liked"},
                    "n": {"type": ["integer", "null"], "description": "How many movies. Default 5."},
                    "min_rating": {"type": ["number", "null"], "description": "Only return movies rated at or above this, e.g. 7.5. Optional."},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "top_movies_by_genre",
            "description": "Get the highest rated movies in a genre, e.g. Action, Drama, Thriller, Science Fiction, Comedy, Horror, Romance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "genre": {"type": "string"},
                    "min_votes": {"type": ["integer", "null"], "description": "Minimum vote count. Default 500."},
                    "n": {"type": ["integer", "null"], "description": "How many movies. Default 5."},
                },
                "required": ["genre"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_rating",
            "description": "Predict the likely rating (0-10) of a new or hypothetical movie using an ML model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "budget_millions": {"type": "number", "description": "Budget in millions of US dollars"},
                    "runtime": {"type": "number", "description": "Runtime in minutes"},
                    "genre": {"type": "string", "description": "Main genre, e.g. Thriller"},
                    "year": {"type": ["integer", "null"], "description": "Release year. Default 2026."},
                },
                "required": ["budget_millions", "runtime", "genre"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "movies_by_director",
            "description": "Get movies by a director, e.g. Christopher Nolan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Director name"},
                    "n": {"type": ["integer", "null"], "description": "How many movies. Default 5."},
                },
                "required": ["name"],
            },
        },
    },
]

# ================= AGENT LOOP (same as Step 2) =================
SYSTEM_PROMPT = (
    "You are CineSense, a movie assistant. Your data is the TMDB 5000 dataset "
    "(mostly Hollywood movies released up to 2017). Always use tools for movie facts "
    "and predictions. If something is not in the data, say so honestly."
)


def run_agent(question, max_steps=6):
    print(f"\n👤 USER: {question}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SCHEMAS,
            tool_choice="auto", temperature=0,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            print(f"🤖 ANSWER: {msg.content}")
            return msg.content

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            print(f"🧾 Step {step}: {name}({args})")
            try:
                result = AVAILABLE_TOOLS[name](**args)
            except Exception as e:
                result = f"Tool error: {e}"
            print(f"🍳 Result: {result[:200]}...")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    print("⛔ Stopped: reached max_steps.")
    return "Stopped: too many steps."


if __name__ == "__main__":
    run_agent("Tell me about Inception.")
    run_agent("Recommend 3 movies like The Dark Knight.")
    run_agent("What are the best science fiction movies?")
    run_agent("If I make a 2 hour thriller with a 50 million dollar budget, what rating might it get?")
    run_agent("Suggest movies like Interstellar, but only ones rated above 7.5.")
