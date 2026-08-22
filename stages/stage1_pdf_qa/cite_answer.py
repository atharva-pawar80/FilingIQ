import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from retrieve import load_vectorstore, retrieve_with_scores
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


def build_numbered_sources(scored_chunks: list) -> tuple[str, list[dict]]:
    context_parts = []
    source_lookup = []

    for i, (doc, score) in enumerate(scored_chunks, start=1):
        company = doc.metadata.get("company", "Unknown")
        page = doc.metadata.get("page", "?")

        context_parts.append(f"[{i}] {doc.page_content}")
        source_lookup.append({
            "number": i,
            "company": company,
            "page": page,
            "score": score,
            "snippet": doc.page_content[:150],
        })

    context = "\n\n".join(context_parts)
    return context, source_lookup



def ask_with_citations(question: str, k: int = 6, score_threshold: float = 0.9) -> dict:
    vectorstore = load_vectorstore()
    scored_chunks = retrieve_with_scores(vectorstore, question, k=k, score_threshold=score_threshold)

    context, source_lookup = build_numbered_sources(scored_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0,
    )

    response = llm.invoke(prompt)

    return {
        "question": question,
        "answer": response.content,
        "sources": source_lookup,
    }

def print_result(result: dict):
    print(f"Question: {result['question']}\n")
    print("--- Answer ---")
    print(result["answer"])
    print("\n--- References ---")
    for src in result["sources"]:
        print(f"[{src['number']}] {src['company']}, page {src['page']} (distance: {src['score']:.3f})")
        print(f"    \"{src['snippet']}...\"")


if __name__ == "__main__":
    result = ask_with_citations("What are the main risk factors mentioned in Apple?")
    print_result(result)