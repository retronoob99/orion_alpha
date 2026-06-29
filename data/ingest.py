from __future__ import annotations

import os
import uuid

import chromadb
import pandas as pd
from chromadb.config import Settings
from dotenv import load_dotenv
from loguru import logger
from sentence_transformers import SentenceTransformer

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()

CRUNCHBASE_CSV_PATH  = os.getenv("CRUNCHBASE_CSV_PATH",  "data/raw/crunchbase_investments.csv")
VC_FUNDING_CSV_PATH  = os.getenv("VC_FUNDING_CSV_PATH",  "data/raw/vc_funding_rounds.csv")
PEOPLE_CSV_PATH      = os.getenv("PEOPLE_CSV_PATH",      "data/raw/people.csv")
CHROMA_PERSIST_DIR   = os.getenv("CHROMA_PERSIST_DIR",   "data/processed/embeddings_index")
EMBEDDING_MODEL      = os.getenv("EMBEDDING_MODEL",      "all-MiniLM-L6-v2")
CHROMA_COLLECTION    = os.getenv("CHROMA_COLLECTION",    "orion_alpha_vc")
BATCH_SIZE           = int(os.getenv("INGEST_BATCH_SIZE", "256"))
CSV_ENCODINGS        = ("utf-8-sig", "utf-8", "cp1252", "latin1")

# ── Column name normalisation map (funding datasets) ─────────────────────────
COLUMN_ALIASES: dict[str, list[str]] = {
    "company":       ["company_name", "name", "startup_name", "organization_name", "company"],
    "sector":        ["market", "category_list", "sector", "industry", "vertical"],
    "funding_round": ["funding_round_type", "round_type", "series", "funding_round", "round"],
    "amount_usd":    ["raised_amount_usd", "funding_total_usd", "amount_usd", "amount", "funding_amount"],
    "investors":     ["investor_names", "investors", "lead_investor", "investor_list"],
    "year":          ["funded_at", "funding_year", "year", "date", "announced_date"],
}

# ── Column name normalisation map (people dataset) ───────────────────────────
PEOPLE_COLUMN_ALIASES: dict[str, list[str]] = {
    "person":    ["full_name", "name", "person_name", "founder_name", "person"],
    "role":      ["title", "job_title", "role", "position", "designation"],
    "company":   ["company_name", "organization_name", "employer", "startup", "company"],
    "location":  ["city", "region", "country", "location", "geography"],
    "education": ["school", "university", "degree", "alma_mater", "education", "institution"],
}


def _normalise_columns(df: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    """Rename dataset-specific columns to canonical names."""
    rename_map: dict[str, str] = {}
    lowered = {c.lower(): c for c in df.columns}
    for canonical, alias_list in aliases.items():
        if canonical not in lowered:
            for alias in alias_list:
                if alias.lower() in lowered:
                    rename_map[lowered[alias.lower()]] = canonical
                    break
    return df.rename(columns=rename_map)


def _read_csv_with_fallback(path: str) -> tuple[pd.DataFrame, str]:
    """Read a CSV using a small set of encodings commonly seen in raw exports."""
    last_error: UnicodeDecodeError | None = None

    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, low_memory=False, encoding=encoding), encoding
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    return pd.read_csv(path, low_memory=False), "default"


def _load_csv(path: str, label: str) -> pd.DataFrame:
    """Load a CSV, normalise columns, clean nulls and duplicates."""
    if not os.path.exists(path):
        logger.warning(f"[{label}] File not found at '{path}' — skipping.")
        return pd.DataFrame()

    logger.info(f"[{label}] Loading '{path}' …")
    df, encoding = _read_csv_with_fallback(path)
    logger.info(f"[{label}] Loaded using encoding: {encoding}")
    logger.info(f"[{label}] Raw rows: {len(df):,}  |  columns: {list(df.columns)}")

    df = df.dropna(how="all")
    df = df.drop_duplicates()
    logger.info(f"[{label}] After clean: {len(df):,} rows")
    return df


def _extract_year(val) -> str:
    """Pull a 4-digit year from a date string or numeric value."""
    if pd.isna(val):
        return "N/A"
    s = str(val).strip()
    if len(s) >= 4 and s[:4].isdigit():
        return s[:4]
    return s


def _row_to_funding_chunk(row: pd.Series) -> str:
    """Convert one funding DataFrame row into a human-readable text chunk."""
    company       = str(row.get("company",       "Unknown")).strip()
    sector        = str(row.get("sector",        "Unknown")).strip()
    funding_round = str(row.get("funding_round", "Unknown")).strip()
    investors     = str(row.get("investors",     "Unknown")).strip()
    year          = _extract_year(row.get("year", "N/A"))

    raw_amount = row.get("amount_usd", None)
    try:
        amount_val = float(raw_amount)
        if amount_val >= 1_000_000:
            amount_str = f"${amount_val / 1_000_000:.1f}M"
        elif amount_val >= 1_000:
            amount_str = f"${amount_val / 1_000:.0f}K"
        else:
            amount_str = f"${amount_val:,.0f}"
    except (TypeError, ValueError):
        amount_str = str(raw_amount) if raw_amount and str(raw_amount) != "nan" else "N/A"

    return (
        f"Company: {company} | "
        f"Sector: {sector} | "
        f"Funding Round: {funding_round} | "
        f"Amount: {amount_str} | "
        f"Investors: {investors} | "
        f"Year: {year}"
    )


def _row_to_people_chunk(row: pd.Series) -> str:
    """Convert one people DataFrame row into a human-readable text chunk."""
    person    = str(row.get("person",    "Unknown")).strip()
    role      = str(row.get("role",      "Unknown")).strip()
    company   = str(row.get("company",   "Unknown")).strip()
    location  = str(row.get("location",  "Unknown")).strip()
    education = str(row.get("education", "Unknown")).strip()

    return (
        f"Person: {person} | "
        f"Role: {role} | "
        f"Company: {company} | "
        f"Location: {location} | "
        f"Education: {education}"
    )


def _embed_and_store(
    chunks: list[str],
    model: SentenceTransformer,
    collection: chromadb.Collection,
) -> int:
    """Embed chunks in batches and upsert into ChromaDB. Returns count stored."""
    total = len(chunks)
    stored = 0

    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        ids   = [str(uuid.uuid4()) for _ in batch]

        logger.debug(
            f"Embedding batch {start // BATCH_SIZE + 1} "
            f"({start}–{min(start + BATCH_SIZE, total)} / {total}) …"
        )

        embeddings = model.encode(batch, show_progress_bar=False).tolist()
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=batch,
        )
        stored += len(batch)

    return stored


def run_ingestion() -> None:
    """Full ingestion pipeline: load → clean → chunk → embed → store."""
    logger.info("═" * 60)
    logger.info("  Orion Alpha — Data Ingestion Pipeline")
    logger.info("═" * 60)

    # ── 1. Load funding CSVs ──────────────────────────────────────────────────
    cb_df = _load_csv(CRUNCHBASE_CSV_PATH, "Crunchbase")
    vc_df = _load_csv(VC_FUNDING_CSV_PATH, "VC Funding")

    if cb_df.empty and vc_df.empty:
        logger.error("Both funding CSVs failed to load. Aborting ingestion.")
        return

    # Normalise funding columns
    if not cb_df.empty:
        cb_df = _normalise_columns(cb_df, COLUMN_ALIASES)
    if not vc_df.empty:
        vc_df = _normalise_columns(vc_df, COLUMN_ALIASES)

    # ── 2. Combine funding datasets ───────────────────────────────────────────
    funding_frames = [df for df in [cb_df, vc_df] if not df.empty]
    combined_df = pd.concat(funding_frames, ignore_index=True).drop_duplicates()
    logger.info(f"Combined funding dataset: {len(combined_df):,} rows")

    # ── 3. Chunk funding rows ─────────────────────────────────────────────────
    logger.info("Building text chunks from funding rows …")
    funding_chunks = [_row_to_funding_chunk(row) for _, row in combined_df.iterrows()]
    logger.info(f"Funding chunks: {len(funding_chunks):,}")

    # ── 4. Load people.csv ────────────────────────────────────────────────────
    logger.info("Loading people.csv …")
    people_df = _load_csv(PEOPLE_CSV_PATH, "People")
    people_chunks: list[str] = []

    if not people_df.empty:
        people_df = _normalise_columns(people_df, PEOPLE_COLUMN_ALIASES)
        people_chunks = [_row_to_people_chunk(row) for _, row in people_df.iterrows()]
        logger.info(f"People chunks: {len(people_chunks):,}")
    else:
        logger.warning("people.csv not loaded — founder/team data will not be indexed.")

    # ── 5. Combine all chunks ─────────────────────────────────────────────────
    all_chunks = funding_chunks + people_chunks
    logger.info(f"Total chunks to embed: {len(all_chunks):,}")

    # ── 6. Load embedding model ───────────────────────────────────────────────
    logger.info(f"Loading embedding model: '{EMBEDDING_MODEL}' …")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model ready.")

    # ── 7. Init ChromaDB ──────────────────────────────────────────────────────
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    logger.info(f"Initialising ChromaDB at '{CHROMA_PERSIST_DIR}' …")
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"Collection '{CHROMA_COLLECTION}' ready.")

    # ── 8. Embed + store all chunks ───────────────────────────────────────────
    logger.info("Embedding and storing all chunks …")
    stored_count = _embed_and_store(all_chunks, embed_model, collection)

    # ── 9. Done ───────────────────────────────────────────────────────────────
    logger.info("═" * 60)
    print(
        f"Ingestion complete. {stored_count:,} documents indexed. "
        f"(includes founder/people data)"
    )
    logger.info(f"ChromaDB persisted → '{CHROMA_PERSIST_DIR}'")
    logger.info("═" * 60)


if __name__ == "__main__":
    run_ingestion()
