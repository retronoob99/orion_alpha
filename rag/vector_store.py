from __future__ import annotations

import os
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from loguru import logger
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Env vars ────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR  = os.getenv("CHROMA_PERSIST_DIR",  "data/processed/embeddings_index")
CHROMA_COLLECTION   = os.getenv("CHROMA_COLLECTION",   "orion_alpha_vc")
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL",     "all-MiniLM-L6-v2")
TOP_K_DEFAULT       = int(os.getenv("VECTOR_TOP_K",    "5"))

# ── Singleton handles ────────────────────────────────────────────────────────
_client:     Optional[chromadb.PersistentClient]   = None
_collection: Optional[chromadb.Collection]         = None
_embedder:   Optional[SentenceTransformer]         = None


# ── Internal init helpers ────────────────────────────────────────────────────

def _get_client() -> chromadb.PersistentClient:
    """Return (or create) the shared ChromaDB persistent client."""
    global _client
    if _client is None:
        logger.info(f"Connecting to ChromaDB at: {CHROMA_PERSIST_DIR}")
        _client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        logger.success("ChromaDB client connected.")
    return _client


def _get_collection() -> chromadb.Collection:
    """Return (or fetch) the existing Chroma collection created by ingest.py."""
    global _collection
    if _collection is None:
        client = _get_client()
        existing = [c.name for c in client.list_collections()]
        if CHROMA_COLLECTION not in existing:
            raise RuntimeError(
                f"Collection '{CHROMA_COLLECTION}' not found in ChromaDB at "
                f"'{CHROMA_PERSIST_DIR}'. Run `python -m data.ingest` first."
            )
        _collection = client.get_collection(
            name=CHROMA_COLLECTION,
            embedding_function=None,   # we supply our own vectors on query
        )
        count = _collection.count()
        logger.info(f"Loaded collection '{CHROMA_COLLECTION}' — {count:,} documents.")
    return _collection


def _get_embedder() -> SentenceTransformer:
    """Return (or load) the shared sentence-transformer model."""
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.success(f"Embedding model '{EMBEDDING_MODEL}' ready.")
    return _embedder


# ── Public API ───────────────────────────────────────────────────────────────

def query(
    text: str,
    top_k: int = TOP_K_DEFAULT,
    where: Optional[dict] = None,
) -> List[dict]:
    """
    Semantic similarity search against the ChromaDB collection.

    Args:
        text:   Natural-language query string.
        top_k:  Number of results to return (default from VECTOR_TOP_K env var).
        where:  Optional ChromaDB metadata filter dict
                e.g. {"sector": {"$eq": "fintech"}}

    Returns:
        List of dicts, each with keys:
            - id       : document id
            - document : original text chunk
            - distance : cosine distance (lower = more similar)
            - metadata : dict of metadata stored at ingest time
    """
    if not text or not text.strip():
        logger.warning("query() called with empty text — returning empty results.")
        return []

    embedder   = _get_embedder()
    collection = _get_collection()

    query_vector = embedder.encode(text, show_progress_bar=False).tolist()

    query_kwargs: dict = {
        "query_embeddings": [query_vector],
        "n_results":        min(top_k, collection.count() or 1),
        "include":          ["documents", "distances", "metadatas"],
    }
    if where:
        query_kwargs["where"] = where

    logger.debug(f"Querying ChromaDB | top_k={top_k} | query='{text[:80]}...'")
    results = collection.query(**query_kwargs)

    docs       = results.get("documents",  [[]])[0]
    distances  = results.get("distances",  [[]])[0]
    metadatas  = results.get("metadatas",  [[]])[0]
    ids        = results.get("ids",        [[]])[0]

    formatted = [
        {
            "id":       ids[i],
            "document": docs[i],
            "distance": round(distances[i], 4),
            "metadata": metadatas[i] if metadatas else {},
        }
        for i in range(len(docs))
    ]

    logger.debug(f"ChromaDB returned {len(formatted)} results.")
    return formatted


def query_as_context(
    text: str,
    top_k: int = TOP_K_DEFAULT,
    where: Optional[dict] = None,
) -> str:
    """
    Convenience wrapper — returns results as a single newline-joined
    string ready to inject directly into an LLM prompt.

    Each line: "[rank]. <document text>"
    """
    results = query(text, top_k=top_k, where=where)
    if not results:
        return "No relevant context found in vector store."

    lines = [f"{i + 1}. {r['document']}" for i, r in enumerate(results)]
    return "\n".join(lines)


def collection_count() -> int:
    """Return total number of documents currently indexed."""
    return _get_collection().count()


def collection_info() -> dict:
    """Return metadata about the active collection."""
    col = _get_collection()
    return {
        "name":        col.name,
        "count":       col.count(),
        "persist_dir": CHROMA_PERSIST_DIR,
        "model":       EMBEDDING_MODEL,
        "top_k":       TOP_K_DEFAULT,
    }