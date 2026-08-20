from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
VECTORSTORE_DIR = "./vectorstore"
COLLECTION_NAME = "stage1_pdf_chunks"


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )
    return vectorstore