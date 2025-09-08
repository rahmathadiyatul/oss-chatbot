import os
import re
import requests
import traceback
from typing import Dict, Union, Annotated, Sequence, Literal, List, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
)
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool

# State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: Sequence
    chat_history: Sequence

# Env & switches
def _env(name: str, required: bool = True) -> Optional[str]:
    v = os.environ.get(name)
    if required and not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v

PROVIDER_LLM = os.getenv("PROVIDER_LLM").lower()        # azure | groq
PROVIDER_VS  = os.getenv("PROVIDER_VS").lower()       # azure | qdrant

# Tavily
_TAVILY = os.environ.get("TAVILY_API_KEY")

# Vector store + Embeddings
retriever = None
doc_prompt = PromptTemplate.from_template(
    "<context>\n{page_content}\n\n<meta>\nsource: {source}\npage: {page}\n</meta>\n</context>"
)

if PROVIDER_VS == "azure":
    from langchain_openai import AzureOpenAIEmbeddings
    from langchain_community.vectorstores import AzureSearch

    _azure_embeddings_deployment = _env("AZURE_OPENAI_EMBEDDINGS")
    _embedding_api_version = _env("EMBEDDING_API_VERSION")
    _azure_oai_endpoint = _env("AZURE_OPENAI_ENDPOINT")
    _azure_oai_key = _env("AZURE_OPENAI_API_KEY")

    _search_endpoint = _env("AZURE_AI_SEARCH_ADDRESS")
    _search_key = _env("AZURE_SEARCH_AI_KEY")
    _search_index = _env("AZURE_SEARCH_INDEX_NAME")

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=_azure_embeddings_deployment,
        openai_api_version=_embedding_api_version,
        azure_endpoint=_azure_oai_endpoint,
        api_key=_azure_oai_key,
    )
    vector_store = AzureSearch(
        azure_search_endpoint=_search_endpoint,
        azure_search_key=_search_key,
        index_name=_search_index,
        embedding_function=embeddings.embed_query,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

elif PROVIDER_VS == "qdrant":
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from qdrant_client import QdrantClient
    from langchain_community.vectorstores import Qdrant

    HF_MODEL = os.getenv("EMBEDDING_MODEL")
    QDRANT_URL = _env("QDRANT_URL")
    QDRANT_API_KEY = _env("QDRANT_API_KEY")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

    embeddings = HuggingFaceEmbeddings(model_name=HF_MODEL)

    client = QdrantClient(
        url=QDRANT_URL.rstrip("/"),
        api_key=QDRANT_API_KEY,
        prefer_grpc=False,
        timeout=60.0,
    )
    vector_store = Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embeddings=embeddings,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

else:
    raise RuntimeError(f"Unsupported PROVIDER_VS: {PROVIDER_VS}")

# LLMs
LLM_SUPPORTS_TOOLS = False

if PROVIDER_LLM == "azure":
    from langchain_openai import AzureChatOpenAI
    _chat_api_version = _env("OPENAI_API_VERSION")
    _chat_deployment = _env("AZURE_OPENAI_CHAT_DEPLOYMENT")
    _azure_oai_endpoint = _env("AZURE_OPENAI_ENDPOINT")
    _azure_oai_key = _env("AZURE_OPENAI_API_KEY")

    llm = AzureChatOpenAI(
        azure_deployment=_chat_deployment,
        api_version=_chat_api_version,
        temperature=0,
        timeout=None,
        max_retries=2,
    )
    LLM_SUPPORTS_TOOLS = True

elif PROVIDER_LLM == "groq":
    from langchain_groq import ChatGroq
    GROQ_API_KEY = _env("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL")

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0,
        max_retries=2,
    )
    LLM_SUPPORTS_TOOLS = False

else:
    raise RuntimeError(f"Unsupported PROVIDER_LLM: {PROVIDER_LLM}")

# Utilities
def _normalize_messages(msgs: Sequence) -> List[BaseMessage]:
    out: List[BaseMessage] = []
    for m in msgs:
        if isinstance(m, BaseMessage):
            out.append(m)
            continue
        if isinstance(m, tuple) and len(m) == 2:
            role, content = m
            role = (role or "").lower()
            if role in ("user", "human"):
                out.append(HumanMessage(content=content or ""))
            elif role in ("assistant", "ai"):
                out.append(AIMessage(content=content or ""))
            elif role == "tool":
                out.append(ToolMessage(content=content or "", tool_call_id="tool_call"))
            else:
                out.append(HumanMessage(content=content or ""))
            continue
        if isinstance(m, dict):
            role = (m.get("role") or "").lower()
            content = m.get("content") or ""
            if role in ("user", "human"):
                out.append(HumanMessage(content=content))
            elif role in ("assistant", "ai"):
                out.append(AIMessage(content=content))
            elif role == "tool":
                out.append(ToolMessage(content=content, tool_call_id="tool_call"))
            else:
                out.append(HumanMessage(content=content))
            continue
        out.append(HumanMessage(content=str(m)))
    return out

def _needs_clarification(q: str) -> bool:
    vague = ["enough", "recent", "these", "those", "this month", "best", "good accuracy", "improve it"]
    return len(q.strip()) < 8 or any(w in q.lower() for w in vague)

def _prefer_web(q: str) -> bool:
    flags = [
        "search online", "browse the web", "on the web", "latest", "news",
        "release", "today", "this week", "this month",
        "imdb", "movie", "movies", "film", "rating", "ratings", "tmdb", "rotten tomatoes"
    ]
    ql = (q or "").lower()
    return any(f in ql for f in flags)

def _get_question(state: AgentState) -> str:
    try:
        return state["question"][0][1]
    except Exception:
        msgs = state.get("messages", [])
        if msgs:
            last = msgs[-1]
            return getattr(last, "content", "") or (last.get("content") if isinstance(last, dict) else "")
    return ""

def _get_tool_context(state: AgentState, limit_chars: int = 2500) -> str:
    parts = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage) and (msg.content or "").strip():
            parts.append(str(msg.content))
        elif isinstance(msg, dict) and msg.get("type") == "tool" and (msg.get("content") or "").strip():
            parts.append(str(msg["content"]))
    text = "\n\n".join(parts).strip()
    return text[:limit_chars] if len(text) > limit_chars else text

# Tools
retriever_tool = create_retriever_tool(
    retriever,
    name="retrieve_internal_knowledge",
    description="Search the indexed PDFs (academic papers on generative AI).",
    document_prompt=doc_prompt,
)

def _tavily_rest(query: str, max_results: int = 8) -> dict:
    if not _TAVILY:
        raise RuntimeError("TAVILY_API_KEY not set")
    r = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {_TAVILY}", "Content-Type": "application/json"},
        json={
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
            "include_images": False,
            "include_raw_content": False,
        },
        timeout=10,
    )
    r.raise_for_status()
    return r.json()

@tool("web_search", return_direct=False)
def web_search_tool(query: str) -> str:
    """Search the public web via Tavily REST API (Bearer TAVILY_API_KEY). Falls back to DuckDuckGo if Tavily fails or returns no results."""
    print("[web_search] Tavily REST query:", query)
    try:
        data = _tavily_rest(query, max_results=8)
        results = data.get("results") or []
        print("[web_search] tavily results:", len(results))
        items = [f"- {r.get('title','').strip()} — {r.get('url','')}" for r in results if r.get("url")][:5]
        if items:
            return "Web results (Tavily):\n" + "\n".join(items)
        print("[web_search] Tavily empty; fallback to DDG")
    except Exception as e:
        print("[web_search] Tavily REST error:", repr(e))
        traceback.print_exc()
    try:
        print("[web_search] using DDG")
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=8,
        )
        data = r.json()
        lines = []
        if data.get("AbstractText"):
            lines.append(f"- {data.get('Heading','Result')}: {data['AbstractText']}")
        for t in data.get("RelatedTopics", [])[:5]:
            if isinstance(t, dict) and t.get("Text") and t.get("FirstURL"):
                lines.append(f"- {t['Text']} — {t['FirstURL']}")
        return "Web results (DDG):\n" + ("\n".join(lines) if lines else "- No hits")
    except Exception as e:
        print("[web_search] DDG error:", repr(e))
        traceback.print_exc()
        return "Web results:\n- (error calling web search)"

pdf_tools = [retriever_tool]
web_tools = [web_search_tool]

# Nodes
def route_decider(state: AgentState) -> Literal["clarify", "pdf", "web"]:
    q = ""
    try:
        q = state["question"][0][1]
    except Exception:
        msgs = state.get("messages", [])
        if msgs:
            last = msgs[-1]
            q = getattr(last, "content", "") or (last.get("content") if isinstance(last, dict) else "")
    if _needs_clarification(q):
        return "clarify"
    if _prefer_web(q):
        return "web"
    return "pdf"

def router_node(state: AgentState):
    return {"messages": []}

def clarify(state: AgentState):
    q = ""
    try:
        q = state["question"][0][1]
    except Exception:
        q = ""
    sys = SystemMessage(content=(
        "You are a lightweight router for greetings vs. clarification. "
        "If the message is a greeting/thanks/small talk, reply with a brief friendly one-liner and ask how you can help. "
        "Otherwise, if the question is underspecified, ask exactly ONE targeted clarifying question. "
        "Match the user's language. Do not explain your reasoning."
    ))
    out = llm.invoke([sys, HumanMessage(content=q or "")])
    return {"messages": [out]}

def agent_pdf(state: AgentState):
    msgs = _normalize_messages(state["messages"])
    if LLM_SUPPORTS_TOOLS:
        llm_with_tools = llm.bind_tools(pdf_tools)
        return {"messages": [llm_with_tools.invoke(msgs)]}

    q = _get_question(state)
    try:
        docs = retriever.get_relevant_documents(q, k=4)
    except Exception as e:
        print("[agent_pdf] retriever error:", repr(e))
        docs = []
    blocks = []
    for d in docs or []:
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        page = d.metadata.get("page", 0)
        blocks.append(doc_prompt.format(page_content=d.page_content, source=src, page=page))
    payload = "\n\n".join(blocks) if blocks else "No PDF results."
    tm = ToolMessage(content=payload, tool_call_id="retrieve_internal_knowledge")
    return {"messages": [tm]}

def agent_web(state: AgentState):
    msgs = _normalize_messages(state["messages"])
    if LLM_SUPPORTS_TOOLS:
        llm_with_tools = llm.bind_tools(web_tools)
        return {"messages": [llm_with_tools.invoke(msgs)]}

    q = _get_question(state)
    try:
        web_out = web_search_tool.invoke(q)
    except Exception as e:
        print("[agent_web] web tool error:", repr(e))
        web_out = "Web results:\n- (error calling web search)"
    tm = ToolMessage(content=str(web_out or "").strip(), tool_call_id="web_search")
    return {"messages": [tm]}

def grade_documents(state: AgentState) -> Literal["generate", "web"]:
    ctx = _get_tool_context(state)
    if not ctx or len(ctx) < 100:
        return "web"
    q = _get_question(state)
    judge = (
        "You are a verifier. Decide if the retrieved context likely contains enough "
        "specific information to answer the question directly. Reply with exactly one token: YES or NO.\n\n"
        f"Question:\n{q}\n\nRetrieved context:\n{ctx}\n"
        "Answer (YES or NO) only:"
    )
    verdict = llm.invoke([HumanMessage(content=judge)]).content.strip().upper()
    return "generate" if verdict.startswith("YES") else "web"

def _sys() -> str:
    return (
        "You are a helpful research assistant. Prefer retrieved PDF context. "
        "Cite sources with bullet list at the end as 'source (page)'. "
        "If no relevant context exists, route to web search (already handled by the graph)."
    )

def _history_no_tools(state: AgentState) -> List[BaseMessage]:
    """Filter out ToolMessage for models that don't accept it (e.g., Groq)."""
    msgs = []
    for m in state["messages"]:
        if isinstance(m, (HumanMessage, AIMessage)):
            msgs.append(m)
        elif isinstance(m, dict):
            role = (m.get("role") or "").lower()
            content = m.get("content") or ""
            if role in ("user", "human"):
                msgs.append(HumanMessage(content=content))
            elif role in ("assistant", "ai"):
                msgs.append(AIMessage(content=content))
    return msgs

def generate(state: AgentState):
    if LLM_SUPPORTS_TOOLS:
        msgs = _normalize_messages(state["messages"])
        msgs.insert(0, SystemMessage(content=_sys()))
        out = llm.invoke(msgs)
        return {"messages": [out]}

    q = _get_question(state)
    ctx = _get_tool_context(state)
    history = _history_no_tools(state)

    prompt_parts = [ _sys() ]
    if ctx:
        prompt_parts.append(f"Retrieved context:\n{ctx}")
    prompt_parts.append(f"User question:\n{q}")
    final_prompt = "\n\n".join(prompt_parts)

    msgs = [SystemMessage(content="You are a helpful research assistant.")]
    msgs.extend(history[-6:])
    msgs.append(HumanMessage(content=final_prompt))

    out = llm.invoke(msgs)
    return {"messages": [out]}

_meta_block_re = re.compile(
    r"<meta>\s*source:\s*(?P<source>.+?)\s*page:\s*(?P<page>.+?)\s*</meta>",
    re.IGNORECASE | re.DOTALL,
)

def structure_metadata_to_json(text: str) -> List[Dict[str, Union[str, int]]]:
    if not text:
        return []
    items: List[Dict[str, Union[str, int]]] = []
    for m in _meta_block_re.finditer(text):
        source = (m.group("source") or "").strip()
        page_raw = (m.group("page") or "").strip()
        try:
            page = int(re.sub(r"[^\d]", "", page_raw)) if page_raw else 0
        except Exception:
            page = 0
        if source:
            items.append({"source": source, "page": page})
    return items

# Graph
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("clarify", clarify)
workflow.add_node("agent_pdf", agent_pdf)
workflow.add_node("agent_web", agent_web)
workflow.add_node("generate", generate)

# Tool runners (only used when LLM supports tool-calls)
if LLM_SUPPORTS_TOOLS:
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("webtool", ToolNode([web_search_tool]))

workflow.add_edge(START, "router")
workflow.add_conditional_edges(
    "router",
    route_decider,
    {"clarify": "clarify", "pdf": "agent_pdf", "web": "agent_web"},
)

if LLM_SUPPORTS_TOOLS:
    # Azure/tool-calling path
    workflow.add_edge("agent_pdf", "retrieve")
    workflow.add_conditional_edges("retrieve", grade_documents, {"generate": "generate", "web": "agent_web"})
    workflow.add_edge("agent_web", "webtool")
    workflow.add_edge("webtool", "generate")
else:
    # Manual path (Groq): skip ToolNode
    workflow.add_conditional_edges("agent_pdf", grade_documents, {"generate": "generate", "web": "agent_web"})
    workflow.add_edge("agent_web", "generate")

workflow.add_edge("clarify", END)
workflow.add_edge("generate", END)

graph = workflow.compile()

def rag_agent():
    return graph
