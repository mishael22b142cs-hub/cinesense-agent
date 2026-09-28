import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()
MODEL = "openai/gpt-oss-120b"

# ================= 1. TOOLS =================
# Sample data - ratings are example numbers, not real IMDb
MOVIES = {
    "anjaam pathiraa":   {"genre": "thriller", "year": 2020, "rating": 7.8},
    "drishyam":          {"genre": "thriller", "year": 2013, "rating": 8.3},
    "kumbalangi nights": {"genre": "drama",    "year": 2019, "rating": 8.5},
    "premam":            {"genre": "romance",  "year": 2015, "rating": 8.3},
    "romancham":         {"genre": "comedy",   "year": 2023, "rating": 7.7},
}


def get_movie_info(title):
    movie = MOVIES.get(title.lower())
    if movie is None:
        return f"'{title}' not found in database."
    return json.dumps(movie)


def find_movies_by_genre(genre):
    matches = [name.title() for name, m in MOVIES.items() if m["genre"] == genre.lower()]
    if matches:
        return json.dumps(matches)
    return f"No {genre} movies found."


def get_top_rated(min_rating=8.0):
    if min_rating is None:   # ✅ CHANGED: LLM null അയച്ചാൽ default 8.0 എടുക്കുന്നു
        min_rating = 8.0     # ✅ CHANGED
    matches = [(name.title(), m["rating"]) for name, m in MOVIES.items() if m["rating"] >= float(min_rating)]
    matches.sort(key=lambda x: x[1], reverse=True)
    if matches:
        return json.dumps(matches)
    return f"No movies found with rating >= {min_rating}."


AVAILABLE_TOOLS = {
    "get_movie_info": get_movie_info,
    "find_movies_by_genre": find_movies_by_genre,
    "get_top_rated": get_top_rated,
}

# ================= 2. SCHEMAS =================
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_movie_info",
            "description": "Get genre, release year and rating of a specific movie.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Movie title"}
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_movies_by_genre",
            "description": "List movies of a given genre, e.g. thriller, drama, romance, comedy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "genre": {"type": "string", "description": "Genre name"}
                },
                "required": ["genre"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_rated",
            "description": "List movies at or above a minimum rating, sorted highest first. Use this to find the best rated movies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_rating": {"type": ["number", "null"], "description": "Minimum rating, e.g. 8.0. Optional."}  # ✅ CHANGED: null-ഉം അനുവദിക്കുന്നു
                },
                "required": [],
            },
        },
    },
]

# ================= 3. AGENT LOOP =================
def run_agent(question, max_steps=5):
    print(f"\n👤 USER: {question}")
    messages = [
        {"role": "system", "content": "You are a helpful movie assistant. Always use tools for movie facts. If a movie is not found, say so honestly."},
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            print(f"🤖 ANSWER: {msg.content}")
            return msg.content

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            print(f"🧾 Step {step}: LLM asked for {name}({args})")

            try:
                result = AVAILABLE_TOOLS[name](**args)
            except Exception as e:
                result = f"Tool error: {e}"
            print(f"🍳 Tool returned: {result}")

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "Stopped: too many steps."


# ================= 4. TEST =================
if __name__ == "__main__":
    run_agent("Hi, how are you?")
    run_agent("What is the rating of Drishyam?")
    run_agent("Rating of Manjummel Boys?")
    run_agent("Which is the best rated movie?")