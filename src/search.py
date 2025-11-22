import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_openai import ChatOpenAI
from src.chroma_vectorstore import ChromaVectorStore
from src.prompts.prompts_v1 import ORM_QUERY_PROMPT
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class RAGSearch:
    
    def __init__(self, persist_dir: str = "chroma_ejl", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "gpt-4.1-nano-2025-04-14", collection_name='ejl_kb_1'):

        # self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        # Load or build vectorstore
        # faiss_path = os.path.join(persist_dir, "faiss.index")
        # meta_path = os.path.join(persist_dir, "metadata.pkl")


        # if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
        #     from document_loader import load_all_documents
        #     docs = load_all_documents("training_data")
        #     self.vectorstore.build_from_documents(docs)
        # else:
        #     self.vectorstore.load()

         # Create Chroma vector store
        self.vectorstore = ChromaVectorStore(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
            collection_name=collection_name
        )

        # Detect if Vector DB already exists
        chroma_index_dir = os.path.join(persist_dir)
        has_existing = (
            os.path.exists(chroma_index_dir) and
            len(os.listdir(chroma_index_dir)) > 0 and
            self.vectorstore.collection.count() > 0
        )

        if not has_existing:
            from src.document_loader import load_all_documents
            print("[INFO] No existing Chroma DB found. Building from documents...")
            docs = load_all_documents("ejl_data")
            self.vectorstore.build_documents(docs)
        else:
            print("[INFO] Loading existing Chroma DB...")
            self.vectorstore.load()


        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=llm_model)
        print(f"[INFO] OpenAI LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)

        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        
        # prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""

        prompt = ORM_QUERY_PROMPT.format(query=query, context=context)

        response = self.llm.invoke([prompt])

        return response.content