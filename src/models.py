# src/models.py
from sentence_transformers import SentenceTransformer

# Load once and reuse everywhere
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("[INFO] Global embedding model loaded: all-MiniLM-L6-v2")
