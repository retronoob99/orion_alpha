from __future__ import annotations

import os
from dotenv import load_dotenv
from loguru import logger
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer
import chromadb

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL         = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/processed/embeddings_index")
CHROMA_COLLECTION  = os.getenv("CHROMA_COLLECTION", "orion_alpha_vc")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_TOP_K       = int(os.getenv("VECTOR_TOP_K", "10"))
LLM_MAX_TOKENS     = int(os.getenv("LLM_MAX_TOKENS", "512"))

if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Add it to your .env file."
    )

# ── Singletons (lazy-loaded once) ─────────────────────────────────────────────
_embedder:   SentenceTransformer | None = None
_collection: chromadb.Collection | None = None
_llm:        ChatGroq | None            = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.debug(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        logger.debug(f"Connecting to ChromaDB at: {CHROMA_PERSIST_DIR}")
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        try:
            _collection = client.get_collection(name=CHROMA_COLLECTION)
            logger.info(
                f"Connected to ChromaDB collection '{CHROMA_COLLECTION}' "
                f"({_collection.count()} documents)"
            )
        except Exception:
            raise RuntimeError(
                f"ChromaDB collection '{CHROMA_COLLECTION}' not found. "
                "Run `python -m data.ingest` first to index your datasets."
            )
    return _collection


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        logger.debug(f"Initialising Groq LLM: {GROQ_MODEL}")
        _llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=0.2,
            max_tokens=LLM_MAX_TOKENS,
        )
    return _llm


# ── Core steps ────────────────────────────────────────────────────────────────

def _embed_query(query: str) -> list[float]:
    """Embed a plain-text query into a vector."""
    embedder = _get_embedder()
    vector = embedder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    logger.debug(f"Query embedded — dim={len(vector)}")
    return vector.tolist()


def _retrieve_chunks(query_vector: list[float], top_k: int = VECTOR_TOP_K) -> list[str]:
    """Similarity search on ChromaDB — returns top_k document strings."""
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "distances"],
    )
    docs: list[str] = []
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for doc, dist in zip(documents, distances):
        if doc:
            docs.append(doc)
            logger.debug(f"Retrieved chunk (distance={dist:.4f}): {doc[:80]}…")
    logger.info(f"Retrieved {len(docs)} chunks from ChromaDB")
    return docs


def _format_context(chunks: list[str]) -> str:
    """Format retrieved chunks into a numbered context block."""
    if not chunks:
        return "No relevant context found in the knowledge base."
    lines = [f"{i + 1}. {chunk}" for i, chunk in enumerate(chunks)]
    return "\n".join(lines)


def _call_llm(query: str, context: str) -> str:
    """Pass context + query to Groq LLM and return plain-string response."""
    llm = _get_llm()

    system_content = (
        "You are Orion Alpha, an expert AI investment research assistant for pre-seed VCs. "
        "You have access to a knowledge base of startup funding data, founder backgrounds, "
        "and VC investment history. Use the provided context to answer the user's question "
        "accurately and concisely. If the context does not contain enough information, "
        "say so clearly — do not fabricate data."
    )

    user_content = (
        f"Context from knowledge base:\n"
        f"──────────────────────────────\n"
        f"{context}\n"
        f"──────────────────────────────\n\n"
        f"Question: {query}"
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]

    logger.info(f"Calling Groq ({GROQ_MODEL}) with {len(context.splitlines())} context lines")
    response = llm.invoke(messages)
    answer = response.content.strip()
    logger.info(f"LLM response received — {len(answer)} chars")
    return answer


# ── Public entry point ────────────────────────────────────────────────────────

def run_rag_query(query: str) -> str:
    """
    Full RAG pipeline:
      1. Embed the incoming query
      2. Retrieve top-5 similar chunks from ChromaDB
      3. Format chunks into a context string
      4. Pass context + query to Groq LLM
      5. Return the LLM response as a plain string

    Args:
        query: Natural-language research question or company description.

    Returns:
        Plain-string answer from the LLM grounded in the vector store context.
    """
    logger.info(f"RAG query started: '{query[:80]}{'…' if len(query) > 80 else ''}'")

    # Step 1 — Embed query
    query_vector = _embed_query(query)

    # Step 2 — Retrieve top-k chunks
    chunks = _retrieve_chunks(query_vector, top_k=VECTOR_TOP_K)

    # Step 3 — Format context string
    context = _format_context(chunks)

    # Step 4 — Call Groq LLM with context + query
    answer = _call_llm(query, context)

    # Step 5 — Return plain string
    logger.info("RAG pipeline complete.")
    return answer


# ── CLI smoke-test ─────────────────────────────────────────────────────────────
# python -m rag.pipeline "Tell me about Stripe's funding history"
if __name__ == "__main__":
    import sys
    _query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Tell me about early-stage fintech startups"
    print("\n── Orion Alpha RAG Pipeline ──")
    print(f"Query: {_query}\n")
    result = run_rag_query(_query)
    print(f"Answer:\n{result}")
