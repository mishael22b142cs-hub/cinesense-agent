> **Branch note:** `vector-search-rag` is **not merged into `main`** yet. The live
> Streamlit demo linked below still runs `main` (CountVectorizer + Random Forest) and
> is untouched by this branch. See "What's new on this branch" below before merging.

I built an AI movie assistant where the LLM uses my own ML models as tools. 🎬

Meet CineSense Agent. Ask it things like:
• "Recommend movies like The Dark Knight"
• "Movies like Interstellar, but only rated above 7.5"
• "What rating might a 2-hour, $50M thriller get?"

The agent decides which tool to call, runs it, and answers from the results:
🔹 A content-based recommender (CountVectorizer + cosine similarity) on 4,803 TMDB movies
🔹 A Random Forest rating predictor, built without post-release features to avoid data leakage
🔹 Conversation memory, so it remembers what you said earlier

Three things I learned while building it:
1⃣ Build it by hand first. I wrote the agent loop without any framework before moving to LangChain and LangGraph, so I understand what the frameworks do internally.
2⃣ When an agent loops, fix the tool, not the prompt. My agent kept retrying because the recommender couldn't filter by rating. One new parameter turned 6 steps into 1.
3⃣ Test for hallucination. The LLM sometimes quoted IMDb ratings from memory instead of my dataset. A stricter, grounded prompt fixed it.

🔗 Try it live: https://cinesense-agent-mishael.streamlit.app/
💻 Code: https://github.com/mishael22b142cs-hub/cinesense-agent

I'm a 2026 Computer Science graduate looking for data science and ML roles. I'd love your feedback!

#DataScience #MachineLearning #LangGraph #LangChain #GenAI #Python #OpenToWork

---

## What's new on this branch (`vector-search-rag`)

**Real vector-search RAG, replacing keyword matching.** `build_embeddings.py` encodes
each movie's overview/genres/keywords/cast/director as a sentence with
`sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) and indexes it with **FAISS**
(`IndexFlatIP` over L2-normalized vectors = exact cosine-similarity search). The
`recommend_similar` tool in `step3_agent.py` now queries that index directly, instead
of `CountVectorizer` bag-of-words + `sklearn.metrics.pairwise.cosine_similarity`
(still in `build_models.py` for comparison) — bag-of-words only matches shared words,
so it misses e.g. "heist" and "robbery" meaning the same thing; embeddings catch that.

**PyTorch rating model vs. Random Forest baseline.** `train_rating_model.py` trains a
small `nn.Module` MLP (64→32→1) on budget/runtime/year/genre and prints R²/MAE for
both models on the same train/test split, as a direct comparison.

**A second deployment surface.** `api.py` is a minimal FastAPI wrapper (`POST /ask`)
around the same agent loop, independent of the Streamlit UI. `Dockerfile` +
`.github/workflows/deploy.yml` build and push that image to Amazon ECR on every push
to `main`; `DEPLOYMENT.md` has the exact AWS CLI commands to run it on AWS App Runner
with auto-redeploy — real CI/CD, not a one-off deploy.

**Can't build/train locally on this machine.** `torch`/`pandas` fail to import here
with "An Application Control policy has blocked this file" (a Windows WDAC/AppLocker
policy blocking specific compiled DLLs — not a code bug). `colab_build_models.ipynb`
runs the same build/train code in Google Colab and produces a downloadable zip to
extract into `models/`.

**Before merging to `main`:** deploy a second, throwaway Streamlit Community Cloud app
pointed at this branch first. torch + sentence-transformers are much heavier than
scikit-learn alone, and the free tier's cold-start time/memory may not fit — confirm
before replacing the production app's dependencies.

New/changed files: `build_embeddings.py`, `train_rating_model.py`, `step3_agent.py`
(rewritten `recommend_similar`/`predict_rating` internals), `app.py` (updated model
bootstrap), `api.py`, `Dockerfile`, `.dockerignore`, `.github/workflows/deploy.yml`,
`DEPLOYMENT.md`, `colab_build_models.ipynb`, `requirements.txt`.
