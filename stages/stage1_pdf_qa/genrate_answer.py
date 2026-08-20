import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from retrieve import load_vectorstore, retrieve

load_dotenv()

# Why we build the prompt this way: the instruction to answer ONLY from
# the provided context is the core anti-hallucination mechanism at this
# stage. It's not foolproof (that's what Stage 4's evaluation harness
# will actually measure), but it's the single highest-leverage thing we
# can do with a simple prompt, before any fancier techniques.
PROMPT_TEMPLATE = """You are a financial analyst assistant. Answer the question using ONLY the context below, extracted from SEC 10-K filings.

If the answer isn't in the context, say "I don't have enough information in the retrieved context to answer this."

Do not use any outside knowledge. Do not guess or estimate numbers that aren't explicitly stated.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(chunks: list, question: str) -> str:
    """Combines retrieved chunks into one context block, tagging each
    with its source (company + page) so the LLM's answer can reference
    where information came from."""
    context_parts = []
    for doc in chunks:
        company = doc.metadata.get("company", "Unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[{company}, page {page}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def ask(question: str, k: int = 4) -> str:
    vectorstore = load_vectorstore()
    chunks = retrieve(vectorstore, question, k=k)

    prompt = build_prompt(chunks, question)

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0,  # 0 = deterministic, favors sticking to given facts
                         # over creative phrasing — important for a
                         # finance Q&A tool where consistency matters.
    )

    response = llm.invoke(prompt)
    return response.content, chunks


if __name__ == "__main__":
    question = "What are the main risk factors in apple mentioned?"
    print(f"Question: {question}\n")

    answer, source_chunks = ask(question)

    print("--- Answer ---")
    print(answer)

    print("\n--- Retrieved from ---")
    for doc in source_chunks:
        print(f"  {doc.metadata['company']}, page {doc.metadata['page']}")