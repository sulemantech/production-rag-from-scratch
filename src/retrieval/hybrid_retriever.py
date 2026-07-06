import sys
sys.path.insert(0, ".")
from src.retrieval.semantic_retriever import retrieve as semantic_retrieve
from src.retrieval.bm25_retriever import retrieve as bm25_retrieve


def retrieve(question: str, chunks: list[str], collection, top_k: int = 5) -> list[str]:
    # TODO: get top_k*2 results from semantic retriever
    semantic_results = semantic_retrieve(question, collection, top_k=top_k*2)
    # Extract documents from ChromaDB result dict
    semantic_chunks = semantic_results["documents"][0]
    
    # TODO: get top_k*2 results from BM25 retriever
    bm25_results = bm25_retrieve(question, chunks, top_k=top_k*2)
    
    # TODO: combine them using Reciprocal Rank Fusion
    # Create dictionaries mapping document -> rank (1-indexed)
    semantic_ranks = {doc: idx + 1 for idx, doc in enumerate(semantic_chunks)}
    bm25_ranks = {doc: idx + 1 for idx, doc in enumerate(bm25_results)}
    
    # Get all unique documents from both retrievers
    all_docs = set(semantic_chunks) | set(bm25_results)
    
    # Calculate RRF score for each document
    k = 60  # RRF constant
    scores = {}
    for doc in all_docs:
        score = 0
        if doc in semantic_ranks:
            score += 1 / (semantic_ranks[doc] + k)
        if doc in bm25_ranks:
            score += 1 / (bm25_ranks[doc] + k)
        scores[doc] = score
    
    # Sort by score in descending order
    combined_results = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    # TODO: return top_k results
    return combined_results[:top_k]


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.vectorstore.chroma_store import create_collection

    # load chunks from cache
    import json
    with open("datasets/sample_text/biology_embeddings.json", "r") as f:
        data = json.load(f)
    chunks = data["chunks"]

    # store in ChromaDB
    collection = create_collection("biology", persist_directory="chroma_db")

    questions = [
        "what is the role of mitochondria?",
        "how does DNA replication work?",
        "what is the function of ribosomes?"
    ]

    for q in questions:
        results = retrieve(q, chunks, collection, top_k=3)
        print(f"\nQ: {q}")
        for doc in results:
            print("---")
            print(doc[:200])