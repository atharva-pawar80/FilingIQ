"""
chunk_narrative.py
Chunks narrative (prose) sections of a 10-K: Business, Risk Factors, MD&A.

Uses LangChain's RecursiveCharacterTextSplitter, which tries paragraph
breaks first, then sentence breaks, then word breaks — as a last resort —
so it almost never cuts a sentence in half. Overlap ensures an idea that
spans a chunk boundary is still fully readable from either chunk.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ~500 tokens per chunk (roughly 4 chars/token -> ~2000 chars) is the
# balance point for prose: small enough to stay topically coherent,
# large enough to preserve a full line of reasoning (e.g. "revenue fell"
# + "because of X" landing in the same chunk).
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_narrative_section(text: str, company: str, fiscal_year: int, section: str) -> list[dict]:
    """
    Splits one narrative section's text into chunks with metadata attached.
    Returns a list of {"text": ..., "metadata": {...}} dicts ready for embedding.
    """
    raw_chunks = splitter.split_text(text)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "company": company,
                "fiscal_year": fiscal_year,
                "section": section,
                "chunk_type": "narrative",
                "chunk_index": i,
            },
        })
    return chunks


if __name__ == "__main__":
    sample_text = (
        "Our business faces significant competition from other consumer "
        "electronics manufacturers. We compete on price, innovation, and "
        "brand loyalty. Additionally, our supply chain is concentrated in "
        "a small number of regions, which exposes us to geopolitical risk. "
    ) * 20  # repeat to simulate a longer real section

    result = chunk_narrative_section(
        sample_text, company="Apple", fiscal_year=2023, section="Risk Factors"
    )
    print(f"Produced {len(result)} chunks.")
    print("First chunk metadata:", result[0]["metadata"])
    print("First chunk text (first 150 chars):", result[0]["text"][:150])
