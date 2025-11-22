import os
import chromadb
import numpy as np
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embeddings import EmbeddingPipeline
import uuid

class ChromaVectorStore:

    def __init__(self, persist_dir: str = 'chroma_ejl', embedding_model: str = 'all-MiniLM-L6-v2', chunk_size: int = 1000, chunk_overlap: int = 200, collection_name: str = "ejl_kb_1"):

        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name


        self.client = None
        self.collection = None
        self.model = None 
        self._initilize_store()
    
    def _load_model(self):
        
        if self.model is None:
            print(f"[INFO] Loading embedding model: {self.embedding_model}")
            self.model = SentenceTransformer(self.embedding_model)
        return self.model

    
    def _initilize_store(self):
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_dir)

            self.collection = self.client.get_or_create_collection(
                name = self.collection_name,
                metadata={"description": "Schema Document embeddings for RAG"}
            )

            print(f"[INFO] Chroma initialized. Collection: {self.collection_name}")
            print(f"[INFO] Existing documents: {self.collection.count()}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Chroma: {e}")
            raise
    
    def build_documents(self, documents: List[Any]):
        print(f"[INFO] Building Chroma store from {len(documents)} raw documents...")

        # embedding pipeline building
        embedding_pipline = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        chunks = embedding_pipline.chunk_documents(documents)
        embeddings = embedding_pipline.embed_chunks(chunks)

        metadatas = [{"text": chunk.page_content} for chunk in chunks]

        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()

        print(f"[INFO] Chroma vector store built at {self.persist_dir}")

    def add_embeddings(self,embeddings: np.ndarray, metadatas: List[Any] = None):

        if metadatas and len(metadatas) != len(embeddings):
            raise ValueError("Metadata count must match embeddings count")
        
        print(f"[INFO] Adding {embeddings.shape[0]} vectors to Chroma...")

        ids = []
        metadatas_list = []
        documents_text = []
        embeddings_list  = []

        for i, (embedding, metadata) in enumerate(zip(embeddings, metadatas)):
            uid = f"vec_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(uid)

            metadatas_list.append(metadata)
            documents_text.append(metadata.get('text', ''))

            embeddings_list.append(embedding.tolist())
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas_list,
            documents=documents_text
        )

        print(f"[INFO] Added {len(embeddings)} embeddings to Chroma.")
        print(f"[INFO] Total count: {self.collection.count()}")
    
    def save(self):
        print(f"[INFO] Chroma auto-persists to: {self.persist_dir}")
    
    def load(self):
        print(f"[INFO] Reloading Chroma from: {self.persist_dir}")

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_collection(self.collection_name)

        print(f"[INFO] Loaded Chroma. Count: {self.collection.count()}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        # Perform query
        chroma_results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
        )

        results = []
        ids = chroma_results["ids"][0]
        distances = chroma_results["distances"][0]
        metadatas = chroma_results["metadatas"][0]
        documents = chroma_results["documents"][0]

        for i in range(len(ids)):
            results.append({
                "index": i,
                "id": ids[i],
                "distance": distances[i],
                "metadata": metadatas[i],
                "text": documents[i],
            })

        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying Chroma for: '{query_text}'")

        model = self._load_model()
        query_emb = model.encode(query_text).astype("float32")

        return self.search(query_emb, top_k=top_k)