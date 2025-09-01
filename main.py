from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Union

from . import config
from .memory import get_session, add_message, clear_session

app = FastAPI(
    title="Chat with PDF (OSS) – API",
    version="0.1.0",
    description="Minimal API skeleton. RAG & LangGraph will be added next."
)

class ChatMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    session_id: Optional[int] = 0

class ChatInput(BaseModel):
    question: str
    chat_history: List[Dict[str, Union[str, int, None]]] = []
    session_id: Optional[int] = 0

class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: List[dict] = []
    usage: dict = {}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=ChatResponse)
def ask(input: ChatInput):
    # TEMP: echo-style response so you can verify wiring.
    # We'll swap this for LangGraph RAG later.
    session = get_session(input.session_id or 0)

    # Persist user question into session memory
    add_message(input.session_id or 0, "user", input.question)

    fake_answer = f"Echo: {input.question}"
    add_message(input.session_id or 0, "ai", fake_answer)

    return ChatResponse(
        question=input.question,
        answer=fake_answer,
        citations=[],
        usage={"tokens_prompt": 0, "tokens_completion": 0}
    )

@app.post("/memory/clear")
def memory_clear(session_id: Optional[int] = 0):
    clear_session(session_id or 0)
    return {"ok": True}
