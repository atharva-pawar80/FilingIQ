"""
load_pdf.py — Piece 1 of Stage 1.

Loads multiple PDFs and shows what we got. Nothing else yet — we want to
see the raw extracted text/page structure before deciding how to chunk it.
"""

from langchain_community.document_loaders import PyPDFLoader


PDF_FILES = {
    "data/aapl-20250927 apple.pdf": "Apple",
    "data/10-K microsoft.pdf": "Microsoft",
}


def load_pdf(pdf_path: str, company: str):
    """Loads one PDF, tagging every page with which company it belongs to."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Add our own 'company' field into each page's metadata, alongside
    # what PyPDFLoader already gives us (source, page number).
    for doc in documents:
        doc.metadata["company"] = company

    return documents


def load_all_pdfs() -> list:
    """Loads every PDF in PDF_FILES and returns one combined list of pages."""
    all_documents = []
    for pdf_path, company in PDF_FILES.items():
        docs = load_pdf(pdf_path, company)
        print(f"Loaded {len(docs)} pages from {company} ({pdf_path})")
        all_documents.extend(docs)
    return all_documents


if __name__ == "__main__":
    all_docs = load_all_pdfs()

    print()
    print(f"Total pages loaded across all PDFs: {len(all_docs)}")
    print()
    print("--- Sample: first page of each company ---")
    seen_companies = set()
    for doc in all_docs:
        company = doc.metadata["company"]
        if company not in seen_companies:
            seen_companies.add(company)
            print(f"\n[{company}] page {doc.metadata['page']}:")
            print(doc.page_content[:300])