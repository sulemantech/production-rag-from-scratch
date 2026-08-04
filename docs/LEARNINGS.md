# Learnings Log

A running log of real bugs hit while building this pipeline: what broke, why,
how it was fixed, and why it matters beyond this one project. Format per entry:

> **Issue** — what went wrong (symptom)
> **Root cause** — the actual mechanism
> **Fix** — what changed
> **Why it matters** — the transferable lesson

Add an entry whenever a bug takes more than a few minutes to understand, or
when you make a decision you'd want to justify to your future self.

---

### Naive `.split()` tokenizer silently degraded BM25 quality
**Issue** — BM25 retrieval quality regressed after a refactor.
**Root cause** — tokenizer had been simplified to `text.split()`, which doesn't
lowercase, strip punctuation, or remove stopwords — so token overlap between
query and documents was much noisier than it looked.
**Fix** — restored `re.findall(r'\b\w+\b', text.lower())` + a stopword filter.
**Why it matters** — a tokenizer "working" (no crash, returns tokens) is not
the same as it working *well*. Lexical retrieval quality is invisible until
you measure it end-to-end.

---

### Hybrid retriever returned dicts, silently broke the reranker
**Issue** — `CrossEncoder` raised `ValueError: Multimodal dict input contains
unrecognized modality keys`.
**Root cause** — `hybrid_retriever.retrieve()` returned `{id, text, ...}`
dicts after an RRF-fusion refactor, but `rerank()` (and `generate_mcq()`)
assumed plain strings.
**Fix** — final return extracts `.["text"]` before returning.
**Why it matters** — implicit type contracts between pipeline stages are easy
to break silently when one stage is refactored in isolation. Worth a comment
or type hint at the boundary.

---

### Positional chunk IDs broke incremental ingestion
**Issue** — didn't want to re-embed all 8,900+ chunks just to add 2 new
Physics books.
**Root cause** — `add_documents()` assigned IDs as `str(i) for i in
range(len(chunks))` — purely positional. Any change in chunk order/count
(e.g. adding books) shifts every ID, causing collisions with what's already
in the collection.
**Fix** — `attach_chunk_ids()` hashes each chunk's own text
(`md5(text).hexdigest()`) so the ID is stable regardless of processing order
or when the chunk was added.
**Why it matters** — content-addressed IDs are the general pattern for safe
incremental writes to any store (vector DB, cache, CDN). Positional IDs only
work if the full set is always rebuilt atomically.

---

### Non-deterministic generation made a real A/B comparison look noisy
**Issue** — first BGE vs. MiniLM embedding comparison gave inconsistent
results across repeated runs with identical inputs.
**Root cause** — Groq generation calls didn't pin `temperature=0`, so the LLM
step itself was non-deterministic — impossible to tell if a score delta was
from the embedding change or from generation randomness.
**Fix** — pinned `temperature=0` everywhere, then re-ran the *same* config
twice to confirm zero difference (the noise floor) before trusting any
before/after comparison.
**Why it matters** — always establish the noise floor (rerun the identical
config) before attributing a score change to the variable you actually
changed.

---

### `document_cleaner.py`: duplicate `import re` inside the function
**Issue** — `UnboundLocalError: cannot access local variable 're'`.
**Root cause** — a second `import re` was pasted inside the function body
(while adding new cleaning rules). Python treats any name assigned/imported
anywhere in a function as local to the *entire* function scope — so the
first use of `re.sub(...)` earlier in the function now referenced the
not-yet-assigned local, not the module-level import.
**Fix** — deleted the inner duplicate import.
**Why it matters** — classic Python scoping gotcha: an assignment anywhere
in a function retroactively makes that name local for the whole function,
even above the assignment line.

---

### Groq daily token quota differs 5x by model
**Issue** — a 335-question eval run against `llama-3.3-70b-versatile`
couldn't complete — ran out of quota partway through.
**Root cause** — the 70B model has a 100,000 TPD quota vs. the 8B model's
500,000 TPD. Not documented anywhere obvious; only discovered by hitting it.
**Fix** — switched `DEFAULT_MODEL` back to `llama-3.1-8b-instant` for
full-corpus runs; reserve the 70B model for smaller/partial comparisons.
**Why it matters** — free/cheap API tiers often gate by model, not just
account — check quota-per-model before designing an experiment around it.

---

### PDF section-splitting bled content across boundaries
**Issue** — extracted Physics questions from a past-papers PDF contained
fragments of the Logical Reasoning section.
**Root cause** — sections were sliced by *page number*, but a section
boundary fell mid-page, not at a page break.
**Fix** — `split_into_sections()` now finds each header's exact character
position in the full concatenated text and slices between consecutive
header positions, independent of page boundaries.
**Why it matters** — never assume a document's logical structure aligns
with its physical pagination.

---

### Multi-line question stems got truncated to their first line
**Issue** — some extracted questions were cut off mid-sentence.
**Root cause** — the parser took only the first line of text before the
options block, but some question stems wrap across multiple lines before
reaching their `?`.
**Fix** — `extract_question_stem()` scans the whole preamble for the first
`?` and takes everything up to it; falls back to the first line only for
fill-in-the-blank stems with no `?` at all.
**Why it matters** — text-extraction parsers should key off structural
markers (punctuation, delimiters) rather than assumptions about line layout.

---

### A truncation safety-net filter dropped valid questions
**Issue** — a "does this look cut off?" heuristic flagged 10 legitimate
Biology fill-in-the-blank questions (ending in `:`) as truncated.
**Root cause** — first version did `rstrip(".:")` before checking the last
word against a stopword list — stripping the very punctuation that proved
the sentence was intentionally complete.
**Fix** — simplified to a single unconditional rule: keep the question only
if its stem ends in `?`, `:`, or `.`; drop the stopword-heuristic entirely.
**Why it matters** — a safety-net filter is itself a piece of logic that
needs testing against edge cases — don't trust a heuristic just because it's
"just a filter."

---

### Adding "clean" table-derived facts made Chemistry retrieval worse, not better
**Issue** — Chemistry accuracy stayed stuck around 65-74% despite the corpus
containing every fact needed. Diagnosed via chunk-boundary inspection: PDF
table data (electron configs, periodic trends) was flattening into
unreadable blobs via `extract_text()`, so specific facts (e.g. Cr's and
Zn's electron configurations) never reached the model.
**Fix attempted** — used `pdfplumber.extract_tables()` to recover clean
per-row chunks, then annotated each with a derived fact (Hund's-rule
unpaired-electron counts) since questions ask about the *derived* property,
not the raw config string. Iterated three times: raw table rows, annotated
rows, then de-templated annotations (after discovering the annotation
phrasing itself was diluting BM25's ability to tell rows apart — see next
entry).
**Result** — measured on both the 8B and 70B generation models, across all
three iterations: **every version scored equal to or worse than doing
nothing at all** (8B: consistently -3pts vs. baseline; 70B: flat at best,
-3pts at worst). Reverted all 251 added chunks back out of the corpus and
collection.
**Why it matters** — a technically correct, well-reasoned fix (verified
facts, verified formula, verified bug fixes) can still be a net negative if
it adds retrieval-pool noise faster than it adds signal. More candidate
chunks competing for the same top-k slots can crowd out content that was
already working. Always measure the *net* effect on the full eval set, not
just whether the specific target case improved — this is the same
discipline as the wider-candidate-pool regression, applied one level
deeper (content changes, not just retrieval-window size).

### BM25 "self-cannibalization": identical annotation phrasing dilutes its own signal
**Issue** — after annotating ~50 electron-configuration table rows with a
derived fact, BM25 still couldn't surface the one row a question needed.
**Root cause** — BM25 weights a term by how rare it is across the corpus
(inverse document frequency). The annotation used the *same* phrase
template ("N unpaired e-, d: M unpaired") on every row. A word that should
have been a rare, high-signal match ("unpaired") ended up repeated
near-verbatim across ~50 sibling documents, so it could no longer
distinguish one row from another — the fix consumed its own signal.
**Fix** — rewrote the annotator to add a distinctive phrase only for the
two genuinely rare/extreme cases a real exam asks about (a subshell that's
exactly half-filled = "all orbitals unpaired", or exactly full = "all
orbitals paired"), leaving ordinary rows with just a plain per-row number
(which varies naturally and isn't a repeated label).
**Why it matters** — any time the same generated phrase is stamped onto
many similar chunks, it destroys the very term-rarity signal lexical
retrieval (BM25/TF-IDF) depends on. Generated annotations need to be
*distinctive*, not just *correct* — correctness alone doesn't help
retrieval if it's spread identically across every sibling candidate.

### BGE-base vs MiniLM Chemistry checkpoint didn't reproduce — reverted
**Issue** — tried swapping the embedding model from `all-MiniLM-L6-v2`
(384-dim) to `BAAI/bge-base-en-v1.5` (768-dim), hoping for a retrieval
quality gain. Before committing to the ~40-minute full-corpus re-embed, ran
a cheap checkpoint: embedded only the Chemistry subset (3,357 chunks, ~14
min) into a new `mdcat_v3_bge` collection and re-ran the 34-question
Chemistry benchmark against 70B generation. First run: 26/34 (76.5%) vs.
the known MiniLM baseline of 25/34 (73.5%) — a +3pt delta, encouraging but
below the pre-committed +5pt bar for proceeding.
**Root cause of the non-reproduction** — re-ran the identical config to
check the delta wasn't noise, and got 24/34 (70.6%) instead — a 2-question
swing on an unchanged model, unchanged collection, temp=0 generation.
Traced this to `hybrid_retriever.retrieve()`'s RRF fusion:
`all_ids = set(semantic_ranks.keys()) | set(bm25_ranks.keys())` iterates a
Python `set`, whose order for string keys depends on per-process hash
randomization (`PYTHONHASHSEED`, randomized by default). When two chunks
land on an exact fused-score tie, `sorted(scores, key=..., reverse=True)`
is stable but breaks the tie by insertion order into `scores` — which
inherits the set's randomized order. Different process runs can therefore
select a different chunk at the top-k cutoff, changing what the LLM sees
and, occasionally, its answer.
**Decision** — averaging the two runs (~25/34, 73.5%) lands exactly on the
MiniLM baseline. The observed variance (±3pts) is larger than the effect
being measured, so the Chemistry checkpoint does not clear the gate.
Reverted `MODEL_NAME` back to `all-MiniLM-L6-v2`, deleted the partial
`mdcat_v3_bge` collection (only ever had the Chemistry subset embedded —
not a real "negative result" collection worth keeping since it was never
completed), and did not proceed to the full 9-book re-embed.
**Why it matters** — this is the noise-floor check working exactly as
designed: it caught a false-positive-looking +3pt gain before a 40-minute
re-embed and a full 332-question re-eval got spent chasing it. It also
surfaced a real, separate reliability bug (non-deterministic tie-breaking
in RRF fusion) that pre-dates this experiment and would affect *any* eval
comparison run across separate processes, not just this one. Worth a
follow-up: add a deterministic secondary sort key (e.g. `chunk_id`) to
`ranked_ids = sorted(scores, key=lambda x: (scores[x], x), reverse=True)`
so ties resolve the same way every run.

### Latency profiling: no single stage dominates, and retrieval's variance traces to disk, not BM25
**Context** — before optimizing anything (caching, model choice, cost), built
reusable instrumentation (`src/observability/timing.py`'s `timed_stage()`
context manager, safe to no-op when no `timings` dict is passed so every
existing caller of `hybrid_retriever.retrieve()` keeps working unchanged)
and profiled real per-stage wall-clock latency via `latency_profile.py`
across a 5-question Biology sample, first at coarse granularity
(retrieval / rerank / generation), then with `hybrid_retrieve()`'s internals
split into `semantic_retrieve` / `bm25_retrieve` / `rrf_fusion`.

**Finding 1 — no stage reliably dominates.** The "biggest stage" ranking
flipped across three separate measurements: retrieval 1.05s vs. rerank
1.82s (N=1); retrieval avg 0.81s vs. rerank avg 0.53s (first N=5, coarse);
retrieval avg 0.47s vs. rerank avg 0.80s (N=5, fine-grained). `generation`
was consistently the smallest of the three. At this sample size, retrieval
and reranking are comparably expensive — treating either alone as "the"
bottleneck would have been wrong in at least two of the three measurements.

**Finding 2 — within retrieval, the volatility is entirely in the semantic
search, not BM25.** Across the same 5 questions: `semantic_retrieve` ranged
0.067s–0.736s (~11x spread), `bm25_retrieve` stayed tight at 0.238s–0.312s
(~1.3x), and `rrf_fusion` was negligible (tens of microseconds). Sanity
check passed cleanly: `semantic_retrieve + bm25_retrieve + rrf_fusion`
(0.206 + 0.268 + 0.00005 ≈ 0.474s avg) matched the outer `retrieval` timer's
own average (0.475s) almost exactly, confirming the sub-stage instrumentation
is measuring the same work the coarse timer sees.

**Likely root cause of the variance (flagged, not yet confirmed with a
dedicated test)** — `bm25_retrieve` rebuilds a fresh `BM25Okapi` index from
chunks already resident in process memory on every call: pure CPU work, no
I/O, so it's steady. `semantic_retrieve` calls out to ChromaDB, which reads
its HNSW index from a SQLite-backed file on disk
(`chroma_db/chroma.sqlite3`); whether that read hits the OS file cache or
requires an actual disk read plausibly varies run to run, which is the most
likely source of the 11x swing.

**Why it matters** — same discipline as the BGE-checkpoint entry above,
applied to latency instead of accuracy: don't trust a single measurement
(or even one N=5 run) to declare a bottleneck. Only the sub-stage breakdown
revealed a stable, explainable pattern — the coarse 3-stage view alone
would have kept pointing at a different "biggest stage" every time. This
also confirms the reranker's cost is structural, not a bug worth "fixing":
a cross-encoder score is a property of the *(query, chunk)* pair, so unlike
chunk embeddings it can't be precomputed — its cost scales directly with
however many candidates `hybrid_retrieve()` hands it (currently `top_k=20`),
which is a real, tunable lever for a future latency-vs-accuracy tradeoff,
not something to eliminate.

## Cross-cutting best practices validated this project

- **Measure the noise floor before trusting any comparison.** Rerun the
  identical config; if it doesn't reproduce the same score, nothing you
  compare against it means anything yet.
- **Prefer better sourcing over cleverer cleaning.** Validated three times
  (Biology, Chemistry, Physics) — finding a real text-based PDF beat any
  amount of regex engineering against a broken/scanned one.
- **Root-cause before re-running.** When accuracy looks wrong, audit actual
  failures (pull real retrieved context per failure) before assuming a fix
  worked or a component is at fault.
- **Content-addressed IDs for anything incrementally written to.** Positional
  IDs are a footgun the moment the input set can grow or reorder.
