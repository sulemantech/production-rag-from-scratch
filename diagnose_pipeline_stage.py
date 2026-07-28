"""
Ad-hoc diagnostic: for each of the 29 Physics/Chemistry failures, figure out
WHICH stage loses the relevant content -- initial hybrid retrieval (top-20)
or the reranker (top-20 -> top-5).

We don't have labeled "correct chunk" ids, so we proxy relevance with a
simple keyword-overlap score between each candidate's text and the correct
answer option text. If no candidate in the 20-pool scores meaningfully, the
miss is upstream (embeddings/BM25 never surfaced it). If a high-scoring
candidate exists in the pool but doesn't survive into the final top-5, the
miss is the reranker's.
"""
import sys
sys.path.insert(0, ".")
import json
import re

from src.vectorstore.chroma_store import create_collection
from src.retrieval.hybrid_retriever import retrieve as hybrid_retrieve
from src.retrieval.reranker import rerank
from src.retrieval.bm25_retriever import STOP_WORDS

TOKEN_RE = re.compile(r"\b\w+\b")


def keywords(text: str) -> set:
    return {t for t in TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS and len(t) > 2}


def overlap_score(candidate: str, target_kw: set) -> int:
    return len(keywords(candidate) & target_kw)


def main():
    with open("phys_chem_failures_v2.json", "r", encoding="utf-8") as f:
        failures = json.load(f)

    with open("datasets/mdcat_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    collection = create_collection("mdcat_v2", persist_directory="chroma_db")

    retrieval_miss = []
    rerank_miss = []
    reached_generation = []

    for i, item in enumerate(failures, 1):
        question = item["question"]
        subject = item["subject"]
        correct = item["correct_answer"]
        target_kw = keywords(correct) | keywords(question)

        candidates = hybrid_retrieve(
            question, chunks, collection,
            metadata_filter={"subject": subject},
            top_k=20,
        )
        reranked = set(rerank(question, candidates, top_k=5))

        scored = sorted(
            ((overlap_score(c, target_kw), c) for c in candidates),
            key=lambda x: x[0], reverse=True,
        )
        best_score, best_chunk = scored[0]

        in_final = best_chunk in reranked

        print(f"[{i}/29] {subject}: {question[:70]!r}")
        print(f"  best pool overlap score = {best_score}  |  survived into top-5 = {in_final}")

        if best_score <= 1:
            retrieval_miss.append(item)
            print("  -> RETRIEVAL-STAGE MISS (nothing relevant even in top-20 pool)")
        elif not in_final:
            rerank_miss.append(item)
            print("  -> RERANK-STAGE MISS (relevant chunk in pool, dropped before top-5)")
        else:
            reached_generation.append(item)
            print("  -> reached generation with relevant-ish context (likely generation error)")
        print()

    print("=" * 60)
    print(f"Retrieval-stage miss (embeddings/BM25 never found it): {len(retrieval_miss)}")
    print(f"Rerank-stage miss (found, but reranked out of top-5):  {len(rerank_miss)}")
    print(f"Reached generation with relevant context:              {len(reached_generation)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
