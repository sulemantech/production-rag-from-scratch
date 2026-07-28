import sys
sys.path.insert(0, ".")
import json

from src.vectorstore.chroma_store import create_collection
from src.retrieval.hybrid_retriever import retrieve as hybrid_retrieve
from src.retrieval.reranker import rerank

with open("phys_chem_failures_v2.json", "r", encoding="utf-8") as f:
    failures = json.load(f)

with open("datasets/mdcat_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

collection = create_collection("mdcat_v2", persist_directory="chroma_db")

TARGET_INDICES = [10, 23]  # coulomb's constant k; Cr/Zn d-orbital pair (rerank-miss, scores 4 and 6)

for TARGET_INDEX in TARGET_INDICES:
    item = failures[TARGET_INDEX - 1]
    question = item["question"]
    subject = item["subject"]
    correct = item["correct_answer"]

    print(f"\n{'='*70}\nQUESTION #{TARGET_INDEX}: {question}")
    print(f"CORRECT:  {correct}\n")

    candidates = hybrid_retrieve(question, chunks, collection, metadata_filter={"subject": subject}, top_k=20)
    reranked = rerank(question, candidates, top_k=5)
    reranked_set = set(reranked)

    for i, c in enumerate(candidates, 1):
        tag = "[IN TOP-5]" if c in reranked_set else ""
        print(f"--- candidate {i} {tag}")
        print(c[:220].replace("\n", " "))
        print()
