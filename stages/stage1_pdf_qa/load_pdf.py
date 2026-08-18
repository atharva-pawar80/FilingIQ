from langchain_community.document_loaders import PyPDFLoader

# Swap this filename to match whichever PDF you saved in data/
PDF_PATH = "data/aapl-20250927 apple.pdf"


def load_pdf(pdf_path: str):
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


if __name__ == "__main__":
    docs = load_pdf(PDF_PATH)

    print(f"Loaded {len(docs)} pages from {PDF_PATH}")
    print()
    print("--- First page metadata ---")
    print(docs[0].metadata)
    print()
    print("--- First page content (f) --irst 500 chars-")
    print(docs[0].page_content[:500])