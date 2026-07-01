
def retrieve(question:str, collection, top_k:int=5):
    from src.embeddings.huggingface_embeddings import get_embedding
    # get the embedding of the question
    question_embedding = get_embedding(question)
    # query the collection
    results = collection.query(query_embedding=question_embedding, n_results=top_k)
    return results