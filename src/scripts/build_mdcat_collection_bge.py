import json
import sys
sys.path.insert(0, ".")

from src.embeddings.huggingface_embeddings import get_embeddings_batch
from src.vectorstore.chroma_store import create_collection, add_documents

CHUNKS_PATH = "datasets/mdcat_chunks.json"
COLLECTION_NAME = "mdcat_v3_bge"

if __name__ == "__main__":
    subject_filter = sys.argv[1] if len(sys.argv) > 1 else None

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if subject_filter:
        chunks = [c for c in chunks if c["subject"] == subject_filter]
        print(f"Filtered to subject={subject_filter}: {len(chunks)} chunks")

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [
        {"subject": chunk["subject"], "grade": chunk["grade"], "chapter": chunk["chapter"]}
        for chunk in chunks
    ]
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    print(f"Embedding {len(texts)} chunks with BGE-base...")
    embeddings = get_embeddings_batch(texts)

    collection = create_collection(COLLECTION_NAME, persist_directory="chroma_db")
    add_documents(collection, texts, metadatas, embeddings, chunk_ids)

    print(f"Collection count: {collection.count()}")
