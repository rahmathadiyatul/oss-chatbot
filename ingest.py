import os
import glob
import uuid
import time
from typing import List, Iterable

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

VECTOR_BACKEND = os.getenv("VECTOR_BACKEND").lower()
DATA_DIR = os.getenv("PDF_DIR")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))

# utils
def _require(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise SystemExit(f"Missing required env: {name}")
    return v

def _chunked(it: List, size: int) -> Iterable[List]:
    for i in range(0, len(it), size):
        yield it[i : i + size]

def load_pdfs(paths: List[str]):
    docs = []
    for p in paths:
        loader = PyPDFLoader(p)
        pages = loader.load()
        docs.extend(pages)
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for d in chunks:
        d.metadata["id"] = d.metadata.get("id") or str(uuid.uuid4())
        d.metadata["source"] = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        d.metadata["page"] = int(d.metadata.get("page", 0))
    return chunks

# qdrant backend
def ingest_qdrant(chunks):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from qdrant_client import QdrantClient
    from langchain_community.vectorstores import Qdrant

    QDRANT_URL = _require("QDRANT_URL").rstrip("/")
    QDRANT_API_KEY = _require("QDRANT_API_KEY")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

    HF_MODEL = os.getenv("EMBEDDING_MODEL")
    print(f"[qdrant] Using HF embedding model: {HF_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=HF_MODEL)

    print(f"[qdrant] Connecting to {QDRANT_URL} (collection='{QDRANT_COLLECTION}')")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60.0,
        prefer_grpc=False,
    )

    vs = Qdrant(client=client, collection_name=QDRANT_COLLECTION, embeddings=embeddings)
    batch_size = 16
    retries = 3
    backoff = 1.0

    total = len(chunks)
    done = 0
    print(f"[qdrant] Upserting {total} chunks in batches of {batch_size}…")
    for batch in _chunked(chunks, batch_size):
        for attempt in range(1, retries + 1):
            try:
                vs.add_documents(batch)
                done += len(batch)
                break
            except Exception as e:
                if attempt == retries:
                    raise
                print(f"[qdrant] upsert failed (attempt {attempt}/{retries}): {e!r} — retrying in {backoff}s")
                time.sleep(backoff)
                backoff *= 2 
    print(f"[qdrant] Done. Upserted {done}/{total} ✅")

# azure backend
def ingest_azure(chunks):
    from langchain_openai import AzureOpenAIEmbeddings
    from langchain_community.vectorstores import AzureSearch

    AZURE_OPENAI_EMBEDDINGS = _require("AZURE_OPENAI_EMBEDDINGS")
    EMBEDDING_API_VERSION = _require("EMBEDDING_API_VERSION")
    AZURE_OPENAI_ENDPOINT = _require("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = _require("AZURE_OPENAI_API_KEY")

    AZURE_AI_SEARCH_ADDRESS = _require("AZURE_AI_SEARCH_ADDRESS")
    AZURE_SEARCH_AI_KEY = _require("AZURE_SEARCH_AI_KEY")
    AZURE_SEARCH_INDEX_NAME = _require("AZURE_SEARCH_INDEX_NAME")

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBEDDINGS,
        openai_api_version=EMBEDDING_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
    )

    vs = AzureSearch(
        azure_search_endpoint=AZURE_AI_SEARCH_ADDRESS,
        azure_search_key=AZURE_SEARCH_AI_KEY,
        index_name=AZURE_SEARCH_INDEX_NAME,
        embedding_function=embeddings.embed_query,
    )

    print(f"[azure] Upserting to Azure AI Search index: {AZURE_SEARCH_INDEX_NAME}")
    vs.add_documents(chunks)
    print("[azure] Done. ✅")

# main
def main():
    pdf_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {DATA_DIR}. Put at least one PDF there.")

    print(f"Found {len(pdf_paths)} PDF(s). Loading…")
    docs = load_pdfs(pdf_paths)
    print(f"Loaded {len(docs)} pages.")

    chunks = split_docs(docs)
    print(f"Produced {len(chunks)} chunks.")

    if VECTOR_BACKEND == "qdrant":
        ingest_qdrant(chunks)
    elif VECTOR_BACKEND == "azure":
        ingest_azure(chunks)
    else:
        raise SystemExit(f"Unsupported VECTOR_BACKEND='{VECTOR_BACKEND}'. Use 'qdrant' or 'azure'.")

if __name__ == "__main__":
    main()
