import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from retrieve import load_vectorstore, retrieve

load_dotenv()

# Key change from Stage 1's prompt: we explicitly number each source and
# instruct the model to cite them inline. This turns "trust me" into
# "here's exactly which chunk backs this claim" — the model can't cite
# a source that doesn't exist, since it only sees the numbered list we give it.
PROMPT_TEMPLATE = """You are a financial analyst assistant. Answer the question using ONLY the numbered sources below, extracted from SEC 10-K filings.

Cite your sources inline using [1], [2], etc. matching the source numbers below. Every factual claim should have a citation.

If the answer isn't in the sources, say "I don't have enough information in the retrieved context to answer this."

Do not use any outside knowledge. Do not guess or estimate numbers that aren't explicitly stated.

Sources:
{context}

Question: {question}

Answer (with inline citations like [1], [2]):"""


def build_numbered_sources(chunks: list) -> tuple[str, list[dict]]:
    """
    Numbers each chunk (1-indexed, matching how humans naturally cite),
    builds the context string for the prompt, AND builds a separate
    lookup list so we can print a clean "References" section afterward
    that maps [1] -> actual company/page, without the LLM needing to
    know or repeat that metadata itself.
    """
    context_parts = []
    source_lookup = []

    for i, doc in enumerate(chunks, start=1):
        company = doc.metadata.get("company", "Unknown")
        page = doc.metadata.get("page", "?")

        context_parts.append(f"[{i}] {doc.page_content}")
        source_lookup.append({
            "number": i,
            "company": company,
            "page": page,
            "snippet": doc.page_content[:150],
        })

    context = "\n\n".join(context_parts)
    return context, source_lookup