from src.document_loader import load_all_documents
from src.chroma_vectorstore import ChromaVectorStore
from src.search import RAGSearch
import os


def run_rag_system():
    data_dir = "ejl_data"
    persist_dir = "chroma_ejl"

    # Initialize Chroma vector store
    store = ChromaVectorStore(persist_dir=persist_dir)

    # Check if Chroma collection already has data
    if store.collection.count() > 0:
        print("[INFO] Existing Chroma DB found. Loading...")
        store.load()
    else:
        print("[INFO] No Chroma DB found. Building new one...")

        # Load training documents for embeddings
        docs = load_all_documents(data_dir)

        # Build vector store
        store.build_documents(docs)

    # RAG Search system (LLM + VectorStore)
    rag_search = RAGSearch(
        persist_dir=persist_dir,
        collection_name="ejl_kb_1"
    )

    # Take user query
    query = input("Enter your query: ")

    # Generate ORM Query using context-driven RAG
    orm_query = rag_search.search_and_summarize(query, top_k=3)

    print("\nGenerated ORM Query:")
    print(orm_query)


if __name__ == "__main__":
    run_rag_system()

"""
Flow :

Step 1 : Create 2 functions's 
         a1: Integrate a new step before loading documents. 
             Convert pdf or any files to MD and then load them.
         a1.1: Also Make the endpoint for Create , Update , Load and Delete Vector DB
         a2: For adding documents and saving vector store
          b: RAG Search with llm

Completed : Sanjay
Step 2 : Integrate with the Emerald Knowledge base

 
Step 3 : Develop AI Service Class for loading ChatOpen AI along with conversational bot.

    Notes: Class which uses openai with latest version ,supports memory and state maintainance


Step 4 : Integrate the AI Service with the RAG DB which has ejl kb

    Improvements: Add Parallel Runnable integration


Step 5 : Check with the final flow and then try to convert as API and integrate in dev

Improvements : 

1. Try to make the RAG Optimized with HYbird RAG or Agentic RAG or 2 step RAG
2. Implement the best industry standards for langchain
3. Make the flagged user queries included in training using that endpoint
"""