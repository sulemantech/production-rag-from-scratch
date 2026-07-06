from rank_bm25 import BM25Okapi
import re

STOP_WORDS = {"what", "is", "the", "of", "a", "an", "in", "to", "and", "how", "does", "do", "it", "for"}

def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return [t for t in tokens if t not in STOP_WORDS]

def retrieve(question: str, chunks: list[str], top_k: int = 5) -> list:
    # create a BM25 object
    tokenized_chunks = [tokenize(chunk) for chunk in chunks]
   
    bm25 = BM25Okapi(tokenized_chunks)

    # get the scores for the query
    scores = bm25.get_scores(tokenize(question) if question else [])
    # get the top k indices
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    # return the top k chunks
    return [chunks[i] for i in top_k_indices]


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.vectorstore.chroma_store import create_collection
    from src.embeddings.huggingface_embeddings import get_embedding

    # load chunks and embeddings from cache
    import json
    with open("datasets/sample_text/biology_embeddings.json", "r") as f:
        data = json.load(f)
    chunks = data["chunks"]
    embeddings = data["embeddings"]

    # store in ChromaDB
    collection = create_collection("biology", persist_directory="chroma_db")

    questions = [
        "what is the role of mitochondria?",
        "how does DNA replication work?",
        "what is the function of ribosomes?"
    ]

    for q in questions:
        results = retrieve(q, chunks, top_k=3)
        print(f"\nQ: {q}")
        for doc in results:
            print("---")
            print(doc[:200])