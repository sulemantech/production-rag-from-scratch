"""
Full 332-question eval using the 70B model for generation, comparing
against the existing 8B baseline. Saves results incrementally after every
question so that hitting the 70B daily quota (RateLimitError) doesn't lose
progress -- just re-run this script later and it picks up where it left off.
"""
import sys
sys.path.insert(0, ".")
import json
import re

from groq import RateLimitError

from src.vectorstore.chroma_store import create_collection
from src.retrieval.hybrid_retriever import retrieve as hybrid_retrieve
from src.retrieval.reranker import rerank
from src.generation.generator import generate_mcq
from src.evaluation.evaluator import parse_options

RESULTS_PATH = "full_eval_70b_results.json"
MODEL_70B = "llama-3.3-70b-versatile"


def extract_letter_robust(text: str, valid_letters: set) -> str:
    """
    Prefers the LAST standalone valid option letter mentioned in the text
    (handles verbose/reasoning-style responses that conclude with the
    answer rather than leading with it). Falls back to the simple
    first-character heuristic used elsewhere in the pipeline.
    """
    if not text:
        return ""
    matches = [m for m in re.findall(r"\b([A-Ea-e])\b", text) if m.upper() in valid_letters]
    if matches:
        return matches[-1].upper()
    match = re.match(r"^([A-Za-z])[\.\)\s]?", text.strip())
    if match:
        return match.group(1).upper()
    return text.strip()[0].upper() if text.strip() else ""


def load_all_questions():
    with open("datasets/evaluation/mdcat_mcqs.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    for test_key, test_data in data.items():
        subject = test_data.get("subject")
        for q in test_data["questions"]:
            questions.append({
                "key": f"{test_key}:{q['id']}",
                "subject": subject,
                "question": q["question"],
                "options": q["options"],
                "correct_answer": q["correct_answer"],
            })
    return questions


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
    questions = load_all_questions()
    results = load_results()

    with open("datasets/mdcat_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    collection = create_collection("mdcat_v2", persist_directory="chroma_db")

    already_done = len(results)
    print(f"Resuming: {already_done}/{len(questions)} already completed.\n")

    for i, q in enumerate(questions, 1):
        if q["key"] in results:
            continue

        options_dict = parse_options(q["options"])
        valid_letters = set(options_dict.keys())
        correct_letter = extract_letter_robust(q["correct_answer"], valid_letters)

        try:
            candidates = hybrid_retrieve(
                q["question"], chunks, collection,
                metadata_filter={"subject": q["subject"]}, top_k=20,
            )
            reranked = rerank(q["question"], candidates, top_k=5)
            predicted = generate_mcq(q["question"], options_dict, reranked, model=MODEL_70B)
        except RateLimitError:
            print(f"\n[{i}/{len(questions)}] RATE LIMIT HIT -- saved {len(results)} results so far.")
            print("Re-run this script later to resume from here.")
            save_results(results)
            sys.exit(0)

        predicted_letter = extract_letter_robust(predicted, valid_letters)
        is_correct = predicted_letter == correct_letter

        results[q["key"]] = {
            "subject": q["subject"],
            "question": q["question"],
            "correct": correct_letter,
            "predicted": predicted_letter,
            "is_correct": is_correct,
        }
        save_results(results)

        status = "PASS" if is_correct else "FAIL"
        print(f"[{i}/{len(questions)}] {q['subject']} {q['key']}: {status} (correct={correct_letter} predicted={predicted_letter})")

    print("\nAll questions complete.")
    print_summary(results)


def print_summary(results):
    by_subject = {}
    for r in results.values():
        s = r["subject"]
        by_subject.setdefault(s, [0, 0])
        by_subject[s][1] += 1
        if r["is_correct"]:
            by_subject[s][0] += 1

    print("\n" + "=" * 50)
    print("70B FULL EVAL SUMMARY")
    print("=" * 50)
    total_correct = total = 0
    for subject, (correct, total_s) in sorted(by_subject.items()):
        print(f"{subject}: {correct}/{total_s} ({correct/total_s:.2%})")
        total_correct += correct
        total += total_s
    print("-" * 50)
    if total:
        print(f"Overall: {total_correct}/{total} ({total_correct/total:.2%})")
    else:
        print("Overall: 0/0")
    print("=" * 50)


if __name__ == "__main__":
    main()
