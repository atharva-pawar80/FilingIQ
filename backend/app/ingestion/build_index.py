import logging
import os

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.ingestion.section_split import split_into_sections
from app.ingestion.chunk_narrative import chunk_narrative_section
from app.ingestion.chunk_tables import extract_tables_from_html, table_to_row_sentences


logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("filingiq.build_index")

FILING_REGISTRY = {
    "apple_10_k_data.html": {"company": "Apple", "fiscal_year": 2025},
    "microsoft_10-K.html": {"company": "Microsoft", "fiscal_year": 2025},
    "jp_morgan_10k_data.html": {"company": "JPMorgan Chase", "fiscal_year": 2025},
}

def load_embedding_model() -> SentenceTransformer:
    logger.info(f"Loading embedding model: {settings.embedding_model} (first run downloads it, ~1.3GB)")
    return SentenceTransformer(settings.embedding_model)


def get_chroma_collection():
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = client.get_or_create_collection(name="filingiq_10k_chunks")
    return collection


def process_one_filing(filepath: str, company: str, fiscal_year: int) -> list[dict]:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    all_chunks = []

    sections = split_into_sections(html)
    for section_name, section_text in sections.items():
        if section_name == "Financial Statements":
            continue
        chunks = chunk_narrative_section(section_text, company, fiscal_year, section_name)
        all_chunks.extend(chunks)
        logger.info(f"  {section_name}: {len(chunks)} narrative chunks")

    try:
        tables = extract_tables_from_html(html)
        table_chunk_count = 0
        for i, df in enumerate(tables):
            if df.shape[0] < 2 or df.shape[1] < 2:
                continue
            chunks = table_to_row_sentences(df, company, fiscal_year, table_name=f"Table {i}")
            all_chunks.extend(chunks)
            table_chunk_count += len(chunks)
        logger.info(f"  Tables: {table_chunk_count} chunks from {len(tables)} detected tables")
    except Exception as e:
        logger.warning(f"  Table extraction failed for {company}: {e}")

    return all_chunks


def build_index():
    embedder = load_embedding_model()
    collection = get_chroma_collection()
    total_chunks = 0

    for filename, meta in FILING_REGISTRY.items():
        filepath = os.path.join(settings.raw_filings_dir, filename)
        if not os.path.exists(filepath):
            logger.warning(f"Skipping {filename} — not found at {filepath}")
            continue

        logger.info(f"Processing {meta['company']} ({filename})...")
        chunks = process_one_filing(filepath, meta["company"], meta["fiscal_year"])

        if not chunks:
            logger.warning(f"  No chunks produced for {meta['company']} — check the file/parser.")
            continue

        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

        clean_metadatas = []
        for c in chunks:
            md = {k: v for k, v in c["metadata"].items() if k != "full_table_markdown"}
            clean_metadatas.append(md)

        ids = [f"{meta['company']}_{filename}_{i}" for i in range(len(chunks))]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=clean_metadatas,
        )

        total_chunks += len(chunks)
        logger.info(f"  Indexed {len(chunks)} chunks for {meta['company']}.")

    logger.info(f"Done. Total chunks indexed: {total_chunks}")
    logger.info(f"Vector store persisted at: {settings.chroma_persist_dir}")


    if __name__ == "__main__":
        build_index()
        