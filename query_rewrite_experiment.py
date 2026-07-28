"""
Experiment: does rewriting terse MCQ stems into fuller, textbook-style
statements before retrieval surface better chunks?

For each of the 29 known Physics/Chemistry failures:
  1. Re-run the original pipeline (retrieval query = raw MCQ stem) to
     reconfirm it still fails (sanity check against drift).
  2. Rewrite the stem into a fuller declarative/search-friendly form via an
     LLM call, re-run retrieval+rerank with the REWRITTEN query, but
     generate the final answer using the ORIGINAL question + options
     (rewriting is only meant to help retrieval, not change what's asked).

Reports how many flip from wrong -> correct.
"""
import sys
sys.path.insert(0, ".")
import json

from src.vectorstore.chroma_store import create_collection
from src.retrieval.hybrid_retriever import retrieve as hybrid_retrieve
from src.retrieval.reranker import rerank
from src.generation.generator import generate_mcq, _call_groq, DEFAULT_MODEL
from src.evaluation.evaluator import parse_options, extract_letter

REWRITE_PROMPT = """The following is a multiple-choice exam question stem, sometimes terse \
or abbreviated. Rewrite it as a single, fuller, self-contained question or statement in \
plain textbook prose, preserving all technical terms and meaning exactly. Do not answer it. \
Return ONLY the rewritten text, nothing else.

Original: {question}

Rewritten:"""


def rewrite_query(question: str) -> str:
    response = _call_groq(
        DEFAULT_MODEL,
        [{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
    )
    return response.choices[0].message.content.strip()


def run_pipeline(query_for_retrieval, question, options_dict, chunks, collection, subject):
    candidates = hybrid_retrieve(
        query_for_retrieval, chunks, collection,
        metadata_filter={"subject": subject}, top_k=20,
    )
    reranked = rerank(query_for_retrieval, candidates, top_k=5)
    predicted = generate_mcq(question, options_dict, reranked)
    return extract_letter(predicted)


def main():
    with open("phys_chem_failures_v2.json", "r", encoding="utf-8") as f:
        failures = json.load(f)

    with open("datasets/mdcat_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    collection = create_collection("mdcat_v2", persist_directory="chroma_db")

    flipped_to_correct = []
    still_wrong = []
    baseline_now_correct = []  # sanity: shouldn't happen since these are known failures

    for i, item in enumerate(failures, 1):
        question = item["question"]
        subject = item["subject"]
        options_dict = parse_options(item["options"])
        correct_letter = extract_letter(item["correct_answer"])

        baseline_pred = run_pipeline(question, question, options_dict, chunks, collection, subject)

        rewritten = rewrite_query(question)
        rewritten_pred = run_pipeline(rewritten, question, options_dict, chunks, collection, subject)

        print(f"[{i}/29] {subject}: {question[:60]!r}")
        print(f"  rewritten -> {rewritten[:100]!r}")
        print(f"  correct={correct_letter}  baseline={baseline_pred}  rewritten_query={rewritten_pred}")

        if baseline_pred == correct_letter:
            baseline_now_correct.append(item)
            print("  -> NOTE: baseline now correct (drift from earlier eval run)")
        elif rewritten_pred == correct_letter:
            flipped_to_correct.append(item)
            print("  -> FLIPPED TO CORRECT")
        else:
            still_wrong.append(item)
            print("  -> still wrong")
        print()

    print("=" * 60)
    print(f"Baseline drifted to correct (ignore, pre-existing):  {len(baseline_now_correct)}")
    print(f"Flipped wrong -> correct via query rewriting:        {len(flipped_to_correct)}")
    print(f"Still wrong after query rewriting:                   {len(still_wrong)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
