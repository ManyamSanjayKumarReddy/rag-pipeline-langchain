from src.document_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
import os


def run_rag_system():
    data_dir = "data"
    persist_dir = "faiss_store"

    # Initialize vector store
    store = FaissVectorStore(persist_dir)

    # Paths to check
    faiss_index_path = os.path.join(persist_dir, "faiss.index")
    metadata_path = os.path.join(persist_dir, "metadata.pkl")

    # Check if FAISS DB exists
    if os.path.exists(faiss_index_path) and os.path.exists(metadata_path):
        print("[INFO] Existing FAISS vector DB found. Loading...")
        store.load()

    else:
        print("[INFO] No FAISS index found. Building new one...")

        # Load documents ONLY when building the index
        docs = load_all_documents(data_dir)

        # Build FAISS index
        store.build_from_documents(docs)

    # Ask for user query
    query = input("Enter your query: ")

    # Perform RAG search
    rag_search = RAGSearch()
    summary = rag_search.search_and_summarize(query, top_k=3)

    print("Summary:", summary)


if __name__ == "__main__":
    run_rag_system()
