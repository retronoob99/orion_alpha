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
    "company":   ["company_name", "organization_name", "employer", "startup", "company", "affiliation_name"],
    "location":  ["city", "region", "country", "location", "geography", "birthplace"],
    "education": ["school", "university", "degree", "alma_mater", "education", "institution"],
}


def _normalise_columns(df: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    """Rename dataset-specific columns to canonical names.

    IMPORTANT: this only RENAMES columns it finds — it never drops or merges
    columns, so it cannot itself cause row loss. Each canonical key is mapped
    to at most one source column (first alias match wins).
    """
    rename_map: dict[str, str] = {}
    used_source_cols: set[str] = set()
    lowered = {c.lower(): c for c in df.columns}

    for canonical, alias_list in aliases.items():
        if canonical in df.columns:
            # Already has the canonical name — leave it alone.
            continue
        for alias in alias_list:
            source_col = lowered.get(alias.lower())
            if source_col and source_col not in used_source_cols:
                rename_map[source_col] = canonical
                used_source_cols.add(source_col)
                break

    return df.rename(columns=rename_map)


def _read_csv_with_fallback(path: str) -> tuple[pd.DataFrame, str]:
    """Read CSV using a small set of encodings commonly seen in exports."""
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
    """Load a CSV. NOTE: does NOT drop duplicates here — that decision is
    made later, per-dataset, so we can clearly log row-count impact."""
    if not os.path.exists(path):
        logger.warning(f"[{label}] File not found at '{path}' — skipping.")
        return pd.DataFrame()

    logger.info(f"[{label}] Loading '{path}' ...")
    df, encoding = _read_csv_with_fallback(path)
    logger.info(f"[{label}] Loaded using encoding: {encoding}")
    logger.info(f"[{label}] Raw rows: {len(df):,} | columns: {list(df.columns)}")

    df = df.dropna(how="all")
    logger.info(f"[{label}] After dropping fully-empty rows: {len(df):,} rows")
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
    # 'person' may not exist as a single column if first/last name are split.
    person = str(row.get("person", "")).strip()
    if not person or person.lower() == "nan":
        first = str(row.get("first_name", "")).strip()
        last  = str(row.get("last_name", "")).strip()
        person = f"{first} {last}".strip() or "Unknown"

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
    label: str,
    model: SentenceTransformer,
    collection: chromadb.Collection,
) -> int:
    """Embed chunks in batches and upsert into ChromaDB. Returns count stored."""
    total = len(chunks)
    if total == 0:
        logger.warning(f"[{label}] No chunks to embed — skipping.")
        return 0

    stored = 0
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start: start + BATCH_SIZE]
        ids = [str(uuid.uuid4()) for _ in batch]

        logger.debug(
            f"[{label}] Embedding batch {start // BATCH_SIZE + 1} "
            f"({start}-{min(start + BATCH_SIZE, total)} / {total}) ..."
        )

        embeddings = model.encode(batch, show_progress_bar=False).tolist()

        try:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=batch,
            )
            stored += len(batch)
        except Exception as e:
            logger.error(f"[{label}] ChromaDB upsert FAILED for batch at offset {start}: {e}")

    logger.success(f"[{label}] Stored {stored:,} / {total:,} chunks")
    return stored


def run_ingestion() -> None:
    """Full ingestion pipeline: load -> clean -> chunk -> embed -> store."""
    logger.info("=" * 60)
    logger.info("  Vektor — Data Ingestion Pipeline")
    logger.info("=" * 60)

    # ── 1. Load funding CSVs (independently, before any merging) ─────────────
    cb_df = _load_csv(CRUNCHBASE_CSV_PATH, "Crunchbase")
    vc_df = _load_csv(VC_FUNDING_CSV_PATH, "VC Funding")

    if cb_df.empty and vc_df.empty:
        logger.error("Both funding CSVs failed to load. Aborting ingestion.")
        return

    # Normalise funding columns (rename only — no row loss possible here)
    if not cb_df.empty:
        cb_df = _normalise_columns(cb_df, COLUMN_ALIASES)
    if not vc_df.empty:
        vc_df = _normalise_columns(vc_df, COLUMN_ALIASES)

    print(f"[DEBUG] Crunchbase rows after normalise: {len(cb_df):,}")
    print(f"[DEBUG] VC Funding rows after normalise: {len(vc_df):,}")

    # ── 2. Chunk EACH funding dataset SEPARATELY (no cross-dataset dedup) ─────
    # We deliberately do NOT concat + drop_duplicates across the two datasets,
    # since that was the likely cause of mass row loss (rows with similar
    # values in only a few overlapping columns were being treated as dupes).
    funding_chunks: list[str] = []

    if not cb_df.empty:
        cb_chunks = [_row_to_funding_chunk(row) for _, row in cb_df.iterrows()]
        print(f"[DEBUG] Crunchbase chunks created: {len(cb_chunks):,}")
        funding_chunks.extend(cb_chunks)

    if not vc_df.empty:
        vc_chunks = [_row_to_funding_chunk(row) for _, row in vc_df.iterrows()]
        print(f"[DEBUG] VC Funding chunks created: {len(vc_chunks):,}")
        funding_chunks.extend(vc_chunks)

    # Only dedupe EXACT identical chunk strings (safe — true duplicates only)
    before_dedup = len(funding_chunks)
    funding_chunks = list(dict.fromkeys(funding_chunks))
    logger.info(
        f"Funding chunks: {before_dedup:,} before exact-dedup -> "
        f"{len(funding_chunks):,} after"
    )

    # ── 3. Load + chunk people.csv ─────────────────────────────────────────────
    people_df = _load_csv(PEOPLE_CSV_PATH, "People")
    print(f"[DEBUG] people.csv rows loaded: {len(people_df):,}")

    people_chunks: list[str] = []

    if not people_df.empty:
        people_df = _normalise_columns(people_df, PEOPLE_COLUMN_ALIASES)
        people_chunks = [_row_to_people_chunk(row) for _, row in people_df.iterrows()]

        before_dedup_p = len(people_chunks)
        people_chunks = list(dict.fromkeys(people_chunks))

        print(f"[DEBUG] People chunks created: {before_dedup_p:,} -> {len(people_chunks):,} after exact-dedup")
        print("[DEBUG] First 3 people chunks:")
        for i, chunk in enumerate(people_chunks[:3]):
            print(f"  {i + 1}. {chunk}")
    else:
        logger.warning("people.csv not loaded — founder/team data will not be indexed.")

    # ── 4. Load embedding model ────────────────────────────────────────────────
    logger.info(f"Loading embedding model: '{EMBEDDING_MODEL}' ...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model ready.")

    # ── 5. Init ChromaDB (fresh collection — old folder contents already deleted) ─
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    logger.info(f"Initialising ChromaDB at '{CHROMA_PERSIST_DIR}' ...")
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        f"Collection '{CHROMA_COLLECTION}' ready — "
        f"documents before this run: {collection.count():,}"
    )

    # ── 6. Embed + store funding chunks ────────────────────────────────────────
    logger.info("Embedding and storing funding chunks ...")
    funding_stored = _embed_and_store(funding_chunks, "Funding", embed_model, collection)

    # ── 7. Embed + store people chunks (same collection) ──────────────────────
    logger.info("Embedding and storing people chunks ...")
    people_stored = _embed_and_store(people_chunks, "People", embed_model, collection)

    # ── 8. Final counts ─────────────────────────────────────────────────────────
    total_stored = funding_stored + people_stored
    logger.info("=" * 60)
    print(
        f"Funding chunks: {funding_stored:,}, "
        f"People chunks: {people_stored:,}, "
        f"Total stored this run: {total_stored:,}"
    )
    print(f"Final ChromaDB collection count: {collection.count():,}")
    print(
        f"Ingestion complete. {total_stored:,} documents indexed this run "
        f"(includes founder/people data)."
    )
    logger.info(f"ChromaDB persisted -> '{CHROMA_PERSIST_DIR}'")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_ingestion()