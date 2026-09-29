from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

import step3_agent as core   # 🔴 Step 3-ലെ ML functions വീണ്ടും ഉപയോഗിക്കുന്നു, ഒന്നും വീണ്ടും എഴുതേണ്ട

load_dotenv()
MODEL = "openai/gpt-oss-120b"


# ================= TOOLS (schema LangChain തനിയെ ഉണ്ടാക്കും) =================
@tool(parse_docstring=True)   # 🔴 ഈ function-നെ ഒരു tool ആക്കുന്നു; docstring-ൽ നിന്ന് description-ഉം parameter details-ഉം എടുക്കുന്നു
def search_movie(title: str) -> str:   # 🔴 type hints (str) കണ്ടാണ് schema-യിലെ "type" തീരുമാനിക്കുന്നത്
    """Get details of a specific movie: year, genres, rating, director, cast, overview.

    Args:
        title: Movie title.
    """
    return core.search_movie(title)


@tool(parse_docstring=True)
def recommend_similar(title: str, n: int | None = 5, min_rating: float | None = None) -> str:   # 🔴 "| None" = optional, null അയച്ചാലും പ്രശ്നമില്ല
    """Recommend movies similar to a given movie, based on story, genres, keywords, cast and director. Can filter by minimum rating.

    Args:
        title: Movie the user liked.
        n: How many movies. Default 5.
        min_rating: Only return movies rated at or above this, e.g. 7.5.
    """
    return core.recommend_similar(title, n, min_rating)


@tool(parse_docstring=True)
def top_movies_by_genre(genre: str, min_votes: int | None = 500, n: int | None = 5) -> str:
    """Get the highest rated movies in a genre, e.g. Action, Drama, Thriller, Science Fiction, Comedy, Horror, Romance.

    Args:
        genre: Genre name.
        min_votes: Minimum vote count. Default 500.
        n: How many movies. Default 5.
    """
    return core.top_movies_by_genre(genre, min_votes, n)


@tool(parse_docstring=True)
def predict_rating(budget_millions: float, runtime: float, genre: str, year: int | None = 2026) -> str:
    """Predict the likely rating (0-10) of a new or hypothetical movie using an ML model.

    Args:
        budget_millions: Budget in millions of US dollars.
        runtime: Runtime in minutes.
        genre: Main genre, e.g. Thriller.
        year: Release year. Default 2026.
    """
    return core.predict_rating(budget_millions, runtime, genre, year)

@tool(parse_docstring=True)
def movies_by_director(name: str) -> str:
    """Get movies by a director, e.g. Christopher Nolan.

    Args:
        name: Director name.
    """
    return core.movies_by_director(name)

TOOLS = [search_movie, recommend_similar, top_movies_by_genre, predict_rating,movies_by_director]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}   # 🔴 Step 2-ലെ AVAILABLE_TOOLS തന്നെ, പക്ഷേ list-ൽ നിന്ന് തനിയെ ഉണ്ടാക്കുന്നു

# ================= LLM + TOOLS =================
llm = ChatGroq(model=MODEL, temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)   # 🔴 LLM-ന് tools-ന്റെ menu card കൊടുക്കുന്നു; ഇനി എല്ലാ call-ലും tools ഒപ്പം പോകും

SYSTEM_PROMPT = (
    "You are CineSense, a movie assistant. Your data is the TMDB 5000 dataset "
    "(mostly Hollywood movies released up to 2017). Always use tools for movie facts "
    "and predictions. If something is not in the data, say so honestly."
)


# ================= AGENT LOOP (Step 2-ലെ അതേ logic, LangChain രീതിയിൽ) =================
def run_agent(question, max_steps=6):
    print(f"\n👤 USER: {question}")
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]   # 🔴 dict-ന് പകരം message objects

    for step in range(1, max_steps + 1):
        ai_msg = llm_with_tools.invoke(messages)   # 🔴 LLM-നെ വിളിക്കുന്നു; ഉത്തരം ഒരു AIMessage ആയി കിട്ടും
        messages.append(ai_msg)                    # 🔴 LLM-ന്റെ മറുപടി (tool request ഉൾപ്പെടെ) history-യിൽ ചേർക്കുന്നു, ഒരൊറ്റ line-ൽ

        if not ai_msg.tool_calls:
            print(f"🤖 ANSWER: {ai_msg.content}")
            return ai_msg.content

        for tc in ai_msg.tool_calls:   # 🔴 ഓരോ tool_call-ഉം ഒരു dict ആണ്: {"name", "args", "id"}; JSON parse ചെയ്യേണ്ട
            print(f"🧾 Step {step}: {tc['name']}({tc['args']})")
            try:
                result = TOOLS_BY_NAME[tc["name"]].invoke(tc["args"])   # 🔴 tool run ചെയ്യുന്നു
            except Exception as e:
                result = f"Tool error: {e}"
            print(f"🍳 Result: {str(result)[:200]}...")
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))   # 🔴 result LLM-ന് തിരിച്ച് കൊടുക്കുന്നു

    print("⛔ Stopped: reached max_steps.")
    return "Stopped: too many steps."


if __name__ == "__main__":
    # LangChain തനിയെ ഉണ്ടാക്കിയ schema ഒന്ന് കാണാൻ:
    print("🔍 Auto-generated schema for recommend_similar:")
    print(recommend_similar.args)

    run_agent("Recommend 3 movies like The Dark Knight.")
    run_agent("Suggest movies like Interstellar, but only ones rated above 7.5.")
    run_agent("If I make a 2 hour thriller with a 50 million dollar budget, what rating might it get?")