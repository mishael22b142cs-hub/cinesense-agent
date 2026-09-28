import json
import joblib
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
client = Groq()
MODEL = "openai/gpt-oss-120b"

# ================= LOAD MODELS (once, at startup) =================
catalog = joblib.load("models/catalog.pkl")           # 🔴 save ചെയ്ത movie table load ചെയ്യുന്നു, app start ആകുമ്പോൾ ഒരിക്കൽ മാത്രം
vectors = joblib.load("models/vectors.pkl")
rating_model = joblib.load("models/rating_model.pkl")


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


def recommend_similar(title, n=5, min_rating=None):                                # ✅ CHANGED: min_rating ചേർത്തു
    i = find_index(title)
    if i is None:
        return f"'{title}' not found in the TMDB 5000 dataset."
    n = int(n or 5)
    scores = cosine_similarity(vectors[i], vectors).flatten()                      # 🔴 ഈ movie-ക്ക് മറ്റ് എല്ലാ movies-മായുള്ള similarity score കണക്കാക്കുന്നു
    ranked = [j for j in scores.argsort()[::-1] if j != i]                         # 🔴 ഏറ്റവും സാമ്യമുള്ളത് ആദ്യം വരുന്ന ക്രമത്തിൽ sort ചെയ്യുന്നു, ആ movie-യെ തന്നെ ഒഴിവാക്കുന്നു
    if min_rating is not None:                                                     # ✅ CHANGED
        ranked = [j for j in ranked[:300]                                          # ✅ CHANGED 🔴 ഏറ്റവും സാമ്യമുള്ള 300 എണ്ണത്തിൽ നിന്ന് മാത്രം തിരയുന്നു
                  if catalog.loc[j, "vote_average"] >= float(min_rating)           # ✅ CHANGED 🔴 rating filter
                  and catalog.loc[j, "vote_count"] >= 100]                         # ✅ CHANGED 🔴 വളരെ കുറച്ച് votes മാത്രമുള്ള "fake high" ratings ഒഴിവാക്കുന്നു
    best = ranked[:n]                                                              # ✅ CHANGED
    if not best:                                                                   # ✅ CHANGED
        return f"No movies similar to '{title}' found with rating >= {min_rating}."  # ✅ CHANGED
    return json.dumps([movie_card(j) for j in best])


def top_movies_by_genre(genre, min_votes=500, n=5):
    g = genre.lower()
    mask = catalog["genre_list"].apply(lambda lst: g in [x.lower() for x in lst])
    result = catalog[mask & (catalog["vote_count"] >= int(min_votes or 500))]
    result = result.sort_values("vote_average", ascending=False).head(int(n or 5))
    if result.empty:
        return f"No '{genre}' movies found."
    return json.dumps([movie_card(i) for i in result.index])


def predict_rating(budget_millions, runtime, genre, year=2026):
    X = pd.DataFrame([{                                                 # 🔴 training-ൽ ഉപയോഗിച്ച അതേ column names-ൽ ഒരു row ഉണ്ടാക്കുന്നു
        "budget_millions": float(budget_millions),
        "runtime": float(runtime),
        "year": float(year or 2026),
        "main_genre": genre.title(),
    }])
    pred = rating_model.predict(X)[0]                                   # 🔴 Pipeline genre encode ചെയ്ത്, rating predict ചെയ്യുന്നു, ഒരൊറ്റ call-ൽ
    return json.dumps({"predicted_rating": round(float(pred), 1),
                       "note": "Rough estimate from a Random Forest model trained on TMDB data."})


AVAILABLE_TOOLS = {                                                     # 🔴 LLM പറയുന്ന tool name-നെ യഥാർത്ഥ Python function-മായി ബന്ധിപ്പിക്കുന്നു
    "search_movie": search_movie,
    "recommend_similar": recommend_similar,
    "top_movies_by_genre": top_movies_by_genre,
    "predict_rating": predict_rating,
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
            "description": "Recommend movies similar to a given movie, based on story, genres, keywords, cast and director. Can filter by minimum rating.",  # ✅ CHANGED
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Movie the user liked"},
                    "n": {"type": ["integer", "null"], "description": "How many movies. Default 5."},
                    "min_rating": {"type": ["number", "null"], "description": "Only return movies rated at or above this, e.g. 7.5. Optional."},  # ✅ CHANGED
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
    # run_agent("Tell me about Inception.")
    # run_agent("Recommend 3 movies like The Dark Knight.")
    # run_agent("What are the best science fiction movies?")
    # run_agent("If I make a 2 hour thriller with a 50 million dollar budget, what rating might it get?")
    run_agent("Suggest movies like Interstellar, but only ones rated above 7.5.")