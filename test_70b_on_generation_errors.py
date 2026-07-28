"""
Diagnostic: re-run the 27 Biology questions confirmed as genuine
generation/reasoning errors (correct label, model answered wrong) through
the 70B model instead of the 8B instant model. Retrieval/rerank stay
identical -- only the generation model changes. Tests whether these are a
model-capability ceiling or something else.
"""
import sys
sys.path.insert(0, ".")
import json

from src.vectorstore.chroma_store import create_collection
from src.retrieval.hybrid_retriever import retrieve as hybrid_retrieve
from src.retrieval.reranker import rerank
from src.generation.generator import generate_mcq
from src.evaluation.evaluator import parse_options, extract_letter

# (test_key, id) pairs for the 27 confirmed generation-error failures
TARGETS = [
    ("test1", 6), ("test1", 7), ("test1", 8), ("test1", 11), ("test1", 32),
    ("test1", 36), ("test1", 43), ("test1", 47),
    ("test2", 6), ("test2", 36), ("test2", 37), ("test2", 38), ("test2", 47),
    ("test2", 56), ("test2", 59), ("test2", 62),
    ("test3", 5), ("test3", 10), ("test3", 13), ("test3", 16), ("test3", 34),
    ("test3", 37), ("test3", 39), ("test3", 43), ("test3", 59), ("test3", 60),
    ("test3", 61),
]

MODEL_70B = "llama-3.3-70b-versatile"


def main():
    with open("datasets/evaluation/mdcat_mcqs.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    with open("datasets/mdcat_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    collection = create_collection("mdcat_v2", persist_directory="chroma_db")

    by_key = {}
    for tkey, qid in TARGETS:
        by_key.setdefault(tkey, set()).add(qid)

    questions = []
    for tkey, ids in by_key.items():
        for q in data[tkey]["questions"]:
            if q["id"] in ids:
                q = dict(q)
                q["subject"] = data[tkey]["subject"]
                q["_source"] = tkey
                questions.append(q)

    assert len(questions) == 27, f"expected 27, found {len(questions)}"

    correct_count = 0
    for i, q in enumerate(questions, 1):
        options_dict = parse_options(q["options"])
        correct_letter = extract_letter(q["correct_answer"])

        candidates = hybrid_retrieve(
            q["question"], chunks, collection,
            metadata_filter={"subject": q["subject"]}, top_k=20,
        )
        reranked = rerank(q["question"], candidates, top_k=5)
        predicted = generate_mcq(q["question"], options_dict, reranked, model=MODEL_70B)
        predicted_letter = extract_letter(predicted)

        is_correct = predicted_letter == correct_letter
        correct_count += is_correct

        print(f"[{i}/27] {q['_source']} id={q['id']}: {q['question'][:60]!r}")
        print(f"  correct={correct_letter}  70b_predicted={predicted_letter}  {'PASS' if is_correct else 'FAIL'}")
        print()

    print("=" * 60)
    print(f"70B correct on known 8B failures: {correct_count} / 27")
    print("=" * 60)


if __name__ == "__main__":
    main()
