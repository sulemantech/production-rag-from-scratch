"""
Feature 2 Step 2c: Chemistry-only checkpoint comparing BGE-base (mdcat_v3_bge)
against the known MiniLM baseline (73.5%, 25/34, from full_eval_70b_results.json),
before committing to the full ~9000-chunk re-embed.
"""
import sys
sys.path.insert(0, ".")
import json

from src.vectorstore.chroma_store import create_collection
from src.retrieval.hybrid_retriever import retrieve as hybrid_retrieve
from src.retrieval.reranker import rerank
from src.generation.generator import generate_mcq
from src.evaluation.evaluator import parse_options
from full_eval_70b import load_all_questions, extract_letter_robust, MODEL_70B

RESULTS_PATH = sys.argv[1] if len(sys.argv) > 1 else "chemistry_bge_checkpoint_results.json"


def load_results():
    try:
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_results(results):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    questions = [q for q in load_all_questions() if q["subject"] == "Chemistry"]
    results = load_results()

    with open("datasets/mdcat_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    chemistry_chunks = [c for c in chunks if c["subject"] == "Chemistry"]

    collection = create_collection("mdcat_v3_bge", persist_directory="chroma_db")

    print(f"Chemistry questions: {len(questions)}. Already done: {len(results)}")

    for i, q in enumerate(questions, 1):
        if q["key"] in results:
            continue

        options_dict = parse_options(q["options"])
        valid_letters = set(options_dict.keys())
        correct_letter = extract_letter_robust(q["correct_answer"], valid_letters)

        candidates = hybrid_retrieve(
            q["question"], chemistry_chunks, collection,
            metadata_filter={"subject": "Chemistry"}, top_k=20,
        )
        reranked = rerank(q["question"], candidates, top_k=5)
        predicted = generate_mcq(q["question"], options_dict, reranked, model=MODEL_70B)
        predicted_letter = extract_letter_robust(predicted, valid_letters)
        is_correct = predicted_letter == correct_letter

        results[q["key"]] = {
            "question": q["question"],
            "correct": correct_letter,
            "predicted": predicted_letter,
            "is_correct": is_correct,
        }
        save_results(results)
        status = "PASS" if is_correct else "FAIL"
        print(f"[{i}/{len(questions)}] {q['key']}: {status} (correct={correct_letter} predicted={predicted_letter})")

    correct = sum(1 for r in results.values() if r["is_correct"])
    total = len(results)
    print(f"\nBGE-base Chemistry checkpoint: {correct}/{total} ({correct/total:.1%})")
    print("MiniLM baseline was: 25/34 (73.5%)")


if __name__ == "__main__":
    main()
