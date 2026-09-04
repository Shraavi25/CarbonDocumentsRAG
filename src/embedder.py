"""Embed chunks from chunker.py and write them to a persistent Chroma collection.

Uses a local sentence-transformers model so the whole pipeline -- load, clean,
chunk, embed, store -- runs offline with no API key.
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from chunker import chunk_documents

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "carbon_flat_rag"
DB_DIR = Path(__file__).resolve().parent.parent / "db" / "chroma_store"

_model = None
_client = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts):
    return get_model().encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(DB_DIR))
    return _client


def build_index():
    """(Re)build the collection from scratch using the current chunker output."""
    chunks = chunk_documents()
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"{m['doc_id']}_p{m['page']}_{i}" for i, m in enumerate(metadatas)]

    embeddings = embed_texts(texts)

    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return collection


if __name__ == "__main__":
    collection = build_index()
    print(f"Indexed {collection.count()} chunks into '{COLLECTION_NAME}' at {DB_DIR}")
