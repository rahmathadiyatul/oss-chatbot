# Chat With PDF — RAG + Web Router (LangGraph)
Vertical slice: **ingest multiple PDFs, ask questions with session memory, & web fallback**.  
Supports **Azure (LLM + Azure AI Search)** or **Free tier (Groq + Qdrant Cloud)**, switchable via `.env`.


## Architecture Overview
  Query --->|  Router (heur.)    |----> Clarify (LLM one-liner Q / greeting)
                       |
           +-----------+-----------+
           |                       |
        PDF branch              Web branch
   (agent_pdf -> retrieve)     (agent_web -> web_search)
           |                       |
       grade_documents (LLM “enough? YES/NO”)
           |            \ 
         generate <------ \-- if NO --> web_search
           |
        Final answer

- **Router**: heuristic route:
  - underspecified → **Clarify**
  - news/“search online”/IMDB-ish → **Web**
  - else → **PDF**
- **PDF branch**:
  - `retrieve_internal_knowledge` (Azure AI Search or Qdrant)  
  - `grade_documents`: LLM judge decides if context is enough  
  - `generate`: crafts final answer; **cites** as `source (page)` from `<meta>` blocks
- **Web branch**:
  - `web_search` tool via **Tavily REST** (`TAVILY_API_KEY`) → fallback **DuckDuckGo Instant Answer**
- **Session memory**: in-memory per `session_id` (simple dict in `main.py`)

### Agents / Nodes (in `app/rag_agent.py`)
- `router` → returns `clarify | agent_pdf | agent_web`
- `clarify` → greeting reply or **one** targeted question
- `agent_pdf` → calls retriever tool (Azure/Qdrant)
- `agent_web` → calls `web_search` tool (Tavily/DDG)
- `retrieve` / `webtool` → LangGraph `ToolNode` runners
- `grade_documents` → LLM YES/NO on context sufficiency
- `generate` → answer synthesis + citations

## Run with Docker Compose
> **Prereq**: put PDFs in `./data` and create `.env` (see below).  
> Docker files included: `Dockerfile`, `docker-compose.yml`.

```bash
docker compose up --build
```

Compose will:
1) run ingestion (`python ./app/ingest.py`)  
2) start API: `http://localhost:8000`

### Switch providers (no rebuild)
Set in `.env`:
- **Azure mode**: `PROVIDER=azure`
- **Free tier**: `PROVIDER=free` (Groq + Qdrant Cloud + HF embeddings)

## Local (without Docker)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# ingest
python ./app/ingest.py
# run api
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
## Access after run
Swagger: `http://localhost:8000/docs`

## Ingestion
- Put PDFs into `./data`
- `app/ingest.py`:
  - **Azure mode** → `AzureOpenAIEmbeddings` + `Azure AI Search` index upsert
  - **Free mode** → `HuggingFaceEmbeddings` + `Qdrant` upsert

## Trade-offs
- **Citations** rely on prompt-wrapped `<meta>` blocks; reliable for PDF chunks but not verified against full docs.
- **Memory** is in-process (volatile). Good enough for the assignment; not multi-user production-ready.
- **Free tier** stack uses a small HF embedding model → lower recall/semantic depth vs. Azure embeddings, but inexpensive.
- **No streaming** responses yet (keeps code simple).

## How I’d Improve Next
- **Reranking**: add cross-encoder reranker (e.g., `bge-reranker`) before `grade_documents` to boost precision.
- **Session store**: swap in Redis with TTL + user auth; thread-safe memory.
- **Answer guardrails**: claim verification, source-grounding checks, “no-answer” thresholding.
- **Streaming + UI**: SSE/WebSocket for token streaming; a minimal chat UI.

## Notes
- **Provider switching** is runtime via `.env` → no rebuild needed.
- **HF cache** is persisted in Docker (`./hf_cache`) to speed up cold starts in free-tier mode.
- Web search uses **Tavily REST** if `TAVILY_API_KEY` is set; otherwise falls back to **DuckDuckGo Instant Answer**.
