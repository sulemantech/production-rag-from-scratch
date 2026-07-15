import chromadb

def create_collection(name: str, persist_directory: str):
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name=name)
    return collection


def add_documents(collection, chunks: list, metadata: list, embeddings: list):
    collection.add(
        documents=chunks,
        metadatas=metadata,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(chunks))]
    )


def query(collection, query_embedding: list, metadata_filter: dict = None, n_results: int = 5):
    results = collection.query(query_embeddings=[query_embedding], where=metadata_filter, n_results=n_results)
    return results


