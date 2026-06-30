from sentence_transformers import SentenceTransformer

def get_embedding(text: str) -> list:
    # load the model "all-MiniLM-L6-v2"
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # encode the text
    embedding = model.encode(text)
    # return the result
    return embedding.tolist()
    pass

# add main to use the above function to get the embedding of a text file and save it to a json file
if __name__ == "__main__":
    # your existing embedding call here
    
    # add this below it
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    e1 = get_embedding("mitochondria produces ATP through cellular respiration")
    e2 = get_embedding("ATP is the energy currency of the cell")
    e3 = get_embedding("the French Revolution began in 1789")

    e1, e2, e3 = np.array(e1), np.array(e2), np.array(e3)

    print("Biology vs Biology:", cosine_similarity([e1], [e2])[0][0])
    print("Biology vs History:", cosine_similarity([e1], [e3])[0][0])
