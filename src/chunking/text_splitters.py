import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

RECURSIVE_SEPARATORS = ["\n\n", "\n", " ", ""]

def fixed_size_split(document, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(document):
        end = start + chunk_size
        chunk = document[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap   
    return chunks

def sentence_split(document, chunk_size=1000):

    sentences = re.split(r'(?<=[.!?])\s+', document)

    chunks = []
    chunk = ""

    for sentence in sentences:

        # keep adding sentences until chunk is full
        if len(chunk) + len(sentence) <= chunk_size:
            chunk += sentence + " "

        else:
            chunks.append(chunk.strip())
            chunk = sentence + " "

    # add last chunk
    if chunk:
        chunks.append(chunk.strip())

    return chunks


def recursive_split(document, chunk_size=100, overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=RECURSIVE_SEPARATORS
    )

    chunks = splitter.split_text(document)

    print("Original Text length:", len(document))
    print("Number of Chunks:", len(chunks))
    print("Chunks Sizes:", [len(chunk) for chunk in chunks])

    return chunks

if __name__ == "__main__":
    with open("datasets/sample_text/biology.txt", "r", encoding="utf-8") as f:
        text = f.read()

    fixed = fixed_size_split(text, chunk_size=1000, overlap=200)
    recursive = recursive_split(text, chunk_size=1000, overlap=200)
    sentence = sentence_split(text, chunk_size=1000)

    print(f"Fixed:     {len(fixed)} chunks")
    print(f"Recursive: {len(recursive)} chunks")
    print(f"Sentence:  {len(sentence)} chunks")

    print("\n--- Fixed middle chunk ---")
    print(fixed[len(fixed)//2])
    print("\n--- Recursive middle chunk ---")
    print(recursive[len(recursive)//2])
    print("\n--- Sentence middle chunk ---")
    print(sentence[len(sentence)//2])
