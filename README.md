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
