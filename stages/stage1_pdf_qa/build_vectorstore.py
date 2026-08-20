from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from chunk_documents import chunk_all_documents
from load_pdf import load_all_pdfs

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

VECTORSTORE_DIR = "./vectorstore"
COLLECTION_NAME = "stage1_pdf_chunks"


def build_vectorstore():
    print(f"Loading embedding model: {EMBEDDING_MODEL} (first run downloads it, ~1.3GB)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Loading and chunking PDFs...")
    pages = load_all_pdfs()
    chunks = chunk_all_documents(pages)
    print(f"Have {len(chunks)} chunks to embed and store.")

    print("Embedding and storing in Chroma (this takes a few minutes on CPU)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTORSTORE_DIR,
    )

    print(f"Done. {len(chunks)} chunks embedded and persisted to {VECTORSTORE_DIR}")
    return vectorstore


if __name__ == "__main__":
    build_vectorstore()