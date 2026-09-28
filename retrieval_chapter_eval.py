"""
Milestone A: retrieval evaluation using chapter metadata as a proxy metric.

Doesn't call generate_mcq() -- reuses the already-cached correctness data in
full_eval_70b_results.json instead of spending more Groq quota. This
measures retrieval/rerank coherence in isolation from generation quality:
do the reranked chunks agree with each other on which chapter they came
from, and does that agreement correlate with whether the final answer
(already measured, separately) was correct?
"""
import sys
sys.path.insert(0, ".")
import json
from collections import Counter

from src.vectorstore.chroma_store import create_collection
from src.retrieval.hybrid_retriever import retrieve as hybrid_retrieve
from src.retrieval.reranker import rerank
from full_eval_70b import load_all_questions

UNCITABLE = {"Unknown chapter", "Front Matter"}

# Threshold for the spot-check below: "high concentration" means the
# reranked chunks agreed with each other at least this often. 0.8 means
# at least 4 of the 5 reranked chunks shared the same chapter.
SPOT_CHECK_THRESHOLD = 0.8
    

def chapter_concentration(chunks: list[dict]) -> float | None:
    """
    Given the reranked chunks for one question, return what fraction agree
    on the single most common chapter -- a self-consistency score computed
    purely from the chunks themselves, not a comparison against any known
    "correct" chapter (no such label exists for these questions).

    Returns None if none of the chunks have a real, citable chapter --
    there's nothing meaningful to measure agreement over in that case.
    """
    # TODO: build a list of chapters from `chunks`, excluding any chunk
    #       whose "chapter" is in UNCITABLE
    chapters = []

    for chunk in chunks:
        chapter = chunk.get("chapter")
        if chapter and chapter not in UNCITABLE:
            chapters.append(chapter)
    if not chapters:
        return None

    # Counter(chapters) builds a dict-like {chapter_name: how_many_times_seen}.
    # .most_common(1) returns a list of (item, count) pairs, largest count
    # first -- we only want the single top one, so [0] grabs that pair and
    # [1] grabs just the count out of it (ignoring the chapter name itself,
    # since we only need the number for the score).
    top_chapter, top_count = Counter(chapters).most_common(1)[0]

    # This is the actual "concentration" score: what fraction of the
    # citable chunks agree with the majority. len(chapters) is the filtered
    # count (uncitable chunks already excluded above), which is why this
    # isn't divided by len(chunks) -- an excluded chunk shouldn't be able
    # to drag the score down just for being uncitable.
    return top_count / len(chapters)


def main():
    with open("datasets/mdcat_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open("full_eval_70b_results.json", "r", encoding="utf-8") as f:
        cached_results = json.load(f)

    collection = create_collection("mdcat_v2", persist_directory="chroma_db")
    questions = load_all_questions()

    rows = []  # each entry: {"key": ..., "concentration": float, "is_correct": bool}
    uncitable_count = 0

    # Separate, smaller collection: full chunk detail (question, letters,
    # chapter + text snippet per chunk) is only worth keeping for the
    # specific questions we actually want to read by hand afterward --
    # incorrect AND confidently-concentrated. Keeping this out of `rows`
    # avoids holding full chunk text for all 332 questions in memory when
    # we only need it for a handful.
    spot_check_candidates = []

    for i, q in enumerate(questions, 1):
        # Defensive skip: full_eval_70b_results.json was built by a
        # separate script run, possibly at a different time. If it's
        # missing a question (e.g. that earlier run got interrupted before
        # finishing), we have no correctness label to compare against, so
        # there's nothing useful this question can tell us. Move on rather
        # than crashing on a KeyError below.
        if q["key"] not in cached_results:
            continue

        # Same retrieve -> rerank shape used everywhere else in this
        # project (latency_profile.py, the eval scripts). top_k=20 for
        # retrieval, narrowed to top_k=5 by reranking -- matches exactly
        # what the real pipeline does, so this measures the real system,
        # not a simplified stand-in for it.
        candidates = hybrid_retrieve(
            q["question"], chunks, collection,
            metadata_filter={"subject": q["subject"]}, top_k=20,
        )
        reranked = rerank(q["question"], candidates, top_k=5)

        # This is the actual metric we're computing -- see
        # chapter_concentration()'s docstring for what it means.
        score = chapter_concentration(reranked)

        # None means every one of the 5 reranked chunks was uncitable
        # (mostly Physics 11th, which has no chapter detection at all --
        # see docs/LEARNINGS.md). Track how often that happens as its own
        # number rather than silently dropping it, since a high uncitable
        # rate would mean this whole metric is blind for a meaningful
        # chunk of the benchmark.
        if score is None:
            uncitable_count += 1
            continue

        # The correctness label already exists from a previous, separate
        # 70B run -- we're joining our new retrieval-only measurement onto
        # data that's already paid for, instead of spending more Groq
        # quota re-generating answers we already have.
        cached = cached_results[q["key"]]
        is_correct = cached["is_correct"]

        rows.append({
            "key": q["key"],
            "concentration": score,
            "is_correct": is_correct,
        })

        # This is the spot-check: does a high concentration score actually
        # mean the retrieved chunks were RIGHT, or just that they were
        # CONSISTENT (possibly consistently on the wrong topic)? Only
        # questions that are both wrong and confidently-concentrated can
        # answer that -- everything else isn't relevant to the question.
        if not is_correct and score >= SPOT_CHECK_THRESHOLD:
            spot_check_candidates.append({
                "key": q["key"],
                "question": q["question"],
                "correct_answer": cached["correct"],
                "predicted_answer": cached["predicted"],
                "concentration": score,
                # Keep just enough of each chunk to recognize its topic by
                # eye -- the full text isn't needed for a quick read, and
                # printing 5 full chunks per candidate would bury the
                # signal in noise.
                "chunks": [
                    {"chapter": c.get("chapter"), "snippet": c["text"][:180]}
                    for c in reranked
                ],
            })

        if i % 50 == 0:
            print(f"[{i}/{len(questions)}] processed", flush=True)

    # Split into two groups using the correctness label -- this is the
    # actual comparison the whole script exists to make: does retrieval
    # look more "coherent" (higher concentration) on questions the LLM
    # answered right, versus ones it got wrong?
    correct_rows = [r for r in rows if r["is_correct"]]
    incorrect_rows = [r for r in rows if not r["is_correct"]]

    def avg_concentration(group):
        # Guard against an empty group (e.g. if every single question
        # happened to be answered correctly) -- avoids a ZeroDivisionError
        # for what would otherwise be a rare edge case.
        if not group:
            return None
        return sum(r["concentration"] for r in group) / len(group)

    avg_correct = avg_concentration(correct_rows)
    avg_incorrect = avg_concentration(incorrect_rows)

    # "n/a" formatting kept separate from the number formatting on purpose --
    # trying to cram both into one f-string ternary reads worse than just
    # writing the two cases out plainly.
    correct_label = f"{avg_correct:.3f}" if avg_correct is not None else "n/a"
    incorrect_label = f"{avg_incorrect:.3f}" if avg_incorrect is not None else "n/a"

    print("\n=== Chapter concentration vs. correctness ===")
    print(f"Correct answers   ({len(correct_rows)} questions): avg concentration = {correct_label}")
    print(f"Incorrect answers ({len(incorrect_rows)} questions): avg concentration = {incorrect_label}")
    print(f"Skipped as fully uncitable: {uncitable_count} questions")

    print(f"\n=== Spot check: wrong answers with concentration >= {SPOT_CHECK_THRESHOLD} ===")
    print(f"{len(spot_check_candidates)} question(s) qualify. Showing up to 5.\n")
    for c in spot_check_candidates[:5]:
        print(f"[{c['key']}] concentration={c['concentration']:.2f}  "
              f"correct={c['correct_answer']}  predicted={c['predicted_answer']}")
        print(f"  Q: {c['question']}")
        for i, chunk in enumerate(c["chunks"], 1):
            print(f"    {i}. [{chunk['chapter']}] {chunk['snippet']!r}")
        print()


if __name__ == "__main__":
    main()
