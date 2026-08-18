"""
chunk_documents.py — Piece 2 of Stage 1.

Takes the loaded PDF pages and splits them into smaller chunks suitable
for embedding. Deliberately simple: one splitter, one chunk size, no
narrative-vs-table distinction yet (that's Stage 5).
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_pdf import load_all_pdfs

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150  # ~15% overlap so ideas spanning a chunk boundary aren't lost

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)