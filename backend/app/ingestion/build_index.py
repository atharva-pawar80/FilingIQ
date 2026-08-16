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
