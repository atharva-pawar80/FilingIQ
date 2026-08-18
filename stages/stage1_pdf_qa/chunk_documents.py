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

def chunk_all_documents(documents: list) -> list:
    """
    Splits a list of page-level Documents into smaller chunk-level
    Documents. LangChain's splitter automatically carries over each
    original page's metadata (company, source, page number) onto every
    chunk produced from it — we don't have to re-attach it manually.
    """
    chunks = splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    pages = load_all_pdfs()
    print(f"Starting from {len(pages)} pages...")

    chunks = chunk_all_documents(pages)
    print(f"Produced {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print("\n--- Sample chunk ---")
    sample = chunks[10]
    print("Metadata:", sample.metadata)
    print("Content:")
    print(sample.page_content)

    print("\n--- Chunk count per company ---")
    for company in ["Apple", "Microsoft"]:
        count = sum(1 for c in chunks if c.metadata["company"] == company)
        print(f"{company}: {count} chunks")