import chromadb

def create_collection(name: str, persist_directory: str):
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name=name)
    return collection


def add_documents(collection, chunks: list, embeddings: list):
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(chunks))]
    )


def query(collection, query_embedding: list, n_results: int = 5):
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results


if __name__ == "__main__":
    import json
    import sys
    sys.path.append(".")
    from src.embeddings.huggingface_embeddings import get_embedding

    # load chunks and embeddings from cache
    with open("datasets/sample_text/biology_embeddings.json", "r") as f:
        data = json.load(f)
    chunks = data["chunks"]
    embeddings = data["embeddings"]

    # store in ChromaDB
    collection = create_collection("biology", persist_directory="chroma_db")
    add_documents(collection, chunks, embeddings)
    print(f"Stored {collection.count()} documents")

    # test a query
    question = "what is the role of mitochondria?"
    query_embedding = get_embedding(question)
    results = query(collection, query_embedding, n_results=3)

    print("\nTop 3 results:")
    for doc in results["documents"][0]:
        print("---")
        print(doc[:200])
