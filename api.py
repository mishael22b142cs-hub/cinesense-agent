"""
Minimal FastAPI wrapper around the core agent loop (step3_agent.run_agent), so the
agent is reachable as a plain HTTP service — this is what gets containerized and
deployed to AWS (see Dockerfile, DEPLOYMENT.md). Separate from app.py (the Streamlit
UI, already live on Streamlit Community Cloud) — this is an additional interface, not
a replacement.
"""
from fastapi import FastAPI
from pydantic import BaseModel

from step3_agent import run_agent

app = FastAPI(title="CineSense Agent API")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    answer = run_agent(req.question)
    return AskResponse(answer=answer or "")
