"""
Milestone A / Latency Profiling.

Runs a handful of real questions through the actual pipeline (retrieval ->
rerank -> generation) and records how long each stage takes.
"""

import sys
sys.path.insert(0, ".")

import json

from src.vectorstore.chroma_store import create_collection
from src.retrieval.hybrid_retriever import retrieve as hybrid_retrieve
from src.retrieval.reranker import rerank
from src.generation.generator import generate_mcq
from src.evaluation.evaluator import parse_options
from src.observability.timing import summarize, timed_stage


def profile_question(q: dict, chunks: list, collection) -> dict:
    """
    Runs one question through the full pipeline, timing each stage.

    Returns:
        {
            "retrieval": ...,
            "rerank": ...,
            "generation": ...,
            "total": ...
        }
    """
    timings = {}
    options_dict = parse_options(q["options"])

    with timed_stage("retrieval", timings):
        candidates = hybrid_retrieve(
            q["question"],
            chunks,
            collection,
            metadata_filter={"subject": q["subject"]},
            top_k=20,
            timings=timings,
        )

    with timed_stage("rerank", timings):
        reranked = rerank(
            q["question"],
            candidates,
            top_k=5,
        )

    with timed_stage("generation", timings):
        generate_mcq(
            q["question"],
            options_dict,
            reranked,
        )

    timings["total"] = (
        timings["retrieval"]
        + timings["rerank"]
        + timings["generation"]
    )

    return timings


def load_all_questions(data):
    """
    Flatten mdcat_mcqs.json into the structure expected by profile_question().
    """
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


def main():
    with open("datasets/mdcat_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open("datasets/evaluation/mdcat_mcqs.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = load_all_questions(data)

    print(repr(questions[0]["subject"]))

    collection = create_collection(
        "mdcat_v2",
        persist_directory="chroma_db",
    )

    # Test ONE real question first
    biology_questions  = [
         q for q in questions
         if q["subject"] == "Biology"
    ][:5]

    print("Testing one question...\n")
    print(f"length of the questions : {len(biology_questions)}")
    # print(f"Question: {biology_questions[0]['question']}\n")

    all_timings = []
    for q in biology_questions:
        timings = profile_question(q, chunks, collection)
        all_timings.append(timings) 
    
    summary = summarize(all_timings)

    # timings = profile_question(
    #     q,
    #     chunks,
    #     collection,
    # )

    print("\nTimings:")
    print(summary)


if __name__ == "__main__":
    main()