import uvicorn
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
from contextlib import asynccontextmanager

_SESSIONS: Dict[str, List[Dict[str, str]]] = {}

def _get_session(session_id: str) -> List[Dict[str, str]]:
    return _SESSIONS.setdefault(session_id, [])

def _append_message(session_id: str, role: str, content: str):
    _get_session(session_id).append({"role": role, "content": content})

context: Dict[str, object] = {}

def get_rag_agent():
    agent = context.get("rag_agent")
    if agent is None:
        raise RuntimeError({"msg": "agent is not built"})
    yield agent

@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.rag_agent import rag_agent
    context["rag_agent"] = rag_agent()
    yield
    print("Shutting down AI Chatbot Server...")

app = FastAPI(
    title="Chat With PDF Backend",
    version="0.5",
    description="RAG + Web-search router with session memory.",
    lifespan=lifespan,
    docs_url="/docs",
)

class ChatMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatInput(BaseModel):
    question: str
    session_id: Optional[str] = "default"
    chat_history: List[ChatMessage] = [] 

@app.get("/")
def health_check():
    return {"ok": True}

@app.post("/ask")
def ask(
    chat_input: ChatInput,
    rag_agent_graph = Depends(get_rag_agent),
):
    session_id = chat_input.session_id or "default"
    server_hist = _get_session(session_id)
    client_hist = [
        {"role": m.role, "content": m.content}
        for m in chat_input.chat_history
        if m.role and m.content
    ]
    merged_hist = server_hist + client_hist

    chat_history_msg = [(m["role"], m["content"]) for m in merged_hist]
    question_pair = ("user", chat_input.question)
    messages = chat_history_msg + [question_pair]

    graph_input = {
        "messages": messages,
        "question": [question_pair],
        "chat_history": chat_history_msg,
    }

    response = rag_agent_graph.invoke(input=graph_input)

    final_msg = response["messages"][-1]
    final_answer = getattr(final_msg, "content", None) or final_msg.get("content")
    usage_metadata = getattr(final_msg, "usage_metadata", None)

    source_metadata = []
    try:
        from app.rag_agent import structure_metadata_to_json
        for i, rsp in enumerate(response["messages"]):
            rsp_type = getattr(rsp, "type", None) or (rsp.get("type") if isinstance(rsp, dict) else None)
            tool_calls = getattr(rsp, "tool_calls", None) or (rsp.get("tool_calls") if isinstance(rsp, dict) else None)
            if rsp_type == "ai" and tool_calls:
                nxt = response["messages"][i + 1]
                nxt_content = getattr(nxt, "content", None) or (nxt.get("content") if isinstance(nxt, dict) else "")
                source_metadata = structure_metadata_to_json(nxt_content) or []
                break
    except Exception:
        source_metadata = []

    _append_message(session_id, "user", chat_input.question)
    _append_message(session_id, "assistant", final_answer or "")

    return {
        "question": chat_input.question,
        "answer": final_answer,
        "token_usage": usage_metadata,
        "context": source_metadata,
        "session_id": session_id,
    }

class ClearInput(BaseModel):
    session_id: Optional[str] = "default"

@app.post("/memory/clear")
def clear_memory(payload: ClearInput):
    sid = payload.session_id or "default"
    _SESSIONS[sid] = []
    return {"cleared": True, "session_id": sid}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
    )
