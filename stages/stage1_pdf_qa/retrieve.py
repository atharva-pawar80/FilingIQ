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

def retrieve(vectorstore, query: str, k: int = 4):
    
    results = vectorstore.similarity_search(query, k=k)
    return results

def retrieve_with_scores(vectorstore, query: str, k: int = 6, score_threshold: float = 0.5):
    
    results = vectorstore.similarity_search_with_score(query, k=k)

    filtered = [(doc, score) for doc, score in results if score <= score_threshold]

    return filtered

if __name__ == "__main__":
    vectorstore = load_vectorstore()

    test_query = "What are the main risk factors mentioned?"
    print(f"Query: {test_query}\n")

    results = retrieve(vectorstore, test_query, k=4)

    for i, doc in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(f"Company: {doc.metadata['company']}, Page: {doc.metadata['page']}")
        print(doc.page_content[:300])
        print()
    print("\n--- Checking raw similarity scores (to calibrate threshold) ---")
    scored_results = vectorstore.similarity_search_with_score(test_query, k=6)
    for doc, score in scored_results:
        print(f"Score: {score:.4f} | {doc.metadata['company']}, page {doc.metadata['page']} | {doc.page_content[:80]}")


    filtered_results = retrieve_with_scores(vectorstore, test_query, k=6, score_threshold=0.9)
    print(f"\n{len(filtered_results)} of 6 passed the 0.9 threshold")