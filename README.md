# Production RAG From Scratch

A hands-on learning project. Every component is built and understood before moving on.
Not a tutorial to copy — a roadmap to *do*.

---

## How This Works

1. Read the task for the current phase
2. Build it yourself (look things up, experiment, break things)
3. Answer the comprehension questions with Claude before moving to the next phase
4. Only then move on

Claude will NOT write the solution for you. It will ask questions, give hints, and explain when you're stuck.

---

## Project Structure

```
production-rag-from-scratch/
│
├── src/                        # Production-quality implementations (built phase by phase)
│   ├── ingestion/              # Phase 1 — Document loading
│   ├── chunking/               # Phase 2 — Text splitting strategies
│   ├── embeddings/             # Phase 3 — Embedding models
│   ├── vectorstore/            # Phase 4 — Vector DB layer
│   ├── retrieval/              # Phase 5 — Retrieval pipeline
│   ├── reranking/              # Phase 6 — Re-ranking
│   ├── llm/                    # Phase 7 — LLM integration & prompting
│   ├── evaluation/             # Phase 8 — Metrics & benchmarking
│   └── pipeline/               # Phase 9 — Full orchestration
│
├── experiments/                # YOUR playground — messy, exploratory, no pressure
│   ├── phase_1_ingestion/
│   ├── phase_2_chunking/
│   ├── phase_3_embeddings/
│   ├── phase_4_vectorstore/
│   ├── phase_5_retrieval/
│   ├── phase_6_reranking/
│   ├── phase_7_generation/
│   └── phase_8_evaluation/
│
├── notebooks/                  # Jupyter notebooks for visual experiments
│
├── datasets/                   # Documents to load, chunk, and retrieve from
│   ├── sample_text/            # Start here — plain .txt files
│   ├── pdfs/                   # Phase 1 extension
│   └── web/                    # Phase 1 extension
│
├── tests/                      # Tests written AFTER you understand what you're testing
│
├── api/                        # Phase 10 — Production API (FastAPI)
│
├── docs/                       # Architecture diagrams, design decisions, benchmark results
│
├── text_splitters.py           # Early experiment — will move to src/chunking/ in Phase 2
├── chunking.py                 # Early experiment
├── chroma_db/                  # Early ChromaDB experiment
│
├── requirements.txt
├── .env.example
├── CLAUDE.md                   # Teaching rules for Claude in this project
└── README.md
```

---

## The Learning Roadmap

### Phase 1 — Document Ingestion
**What you'll build:** A loader that reads a document and turns it into a Python string (or list of strings) you can work with.

**Your tasks:**
- [ ] Put a `.txt` file (any text you care about) in `datasets/sample_text/`
- [ ] Write `src/ingestion/txt_loader.py` — a function that reads it and returns the raw text
- [ ] Print the first 500 characters to verify it worked
- [ ] Add a PDF to `datasets/pdfs/` and write `src/ingestion/pdf_loader.py` using `PyPDF2` or `pdfplumber`
- [ ] Write `src/ingestion/document_cleaner.py` — strip extra whitespace, weird characters, page headers/footers

**Comprehension checkpoint (answer these before Phase 2):**
1. Why do we load documents as raw text first rather than going straight to chunks?
2. What information might you lose when extracting text from a PDF that you wouldn't lose from a .txt file?
3. If your document had tables, what problem would raw text extraction cause for retrieval later?

---

### Phase 2 — Chunking
**What you'll build:** Multiple chunking strategies and an experiment comparing them.

**Starting point:** `text_splitters.py` already has a recursive chunker. Read it, run it, understand it.

**Your tasks:**
- [ ] Run the existing `recursive_split()` on your loaded document — print and inspect the chunks
- [ ] Change `chunk_size` from 100 → 500 → 1000 — what do you observe?
- [ ] Change `chunk_overlap` from 10 → 0 → 50 — what changes?
- [ ] Write `src/chunking/fixed_chunker.py` — split by character count, no awareness of sentences
- [ ] Write `src/chunking/sentence_chunker.py` — split on sentence boundaries using `nltk` or `spacy`
- [ ] Write `experiments/phase_2_chunking/compare.py` — run all three on the same document and print: number of chunks, average chunk size, min/max sizes
- [ ] Move and refactor the good code from `text_splitters.py` into `src/chunking/recursive_chunker.py`

**Comprehension checkpoint:**
1. Why does chunk overlap exist? What problem does it solve?
2. If chunk_size is too small, what happens to retrieval quality? What if it's too large?
3. A user asks "what happened in Q3?" — which chunking strategy would handle this better: fixed-size or sentence-based? Why?

---

### Phase 3 — Embeddings
**What you'll build:** A function that takes a chunk of text and returns a vector (list of numbers).

**Your tasks:**
- [ ] Install `sentence-transformers` — use `all-MiniLM-L6-v2` as your first model (it's small and fast)
- [ ] Write `src/embeddings/huggingface_embeddings.py` — function takes a string, returns a numpy array
- [ ] Embed 5 sentences, print their shapes — all the same length?
- [ ] Compute cosine similarity between: (a) two similar sentences, (b) two unrelated sentences — do the numbers match your intuition?
- [ ] Embed all the chunks from Phase 2 — how long does it take? How much memory?
- [ ] Try a different model (`all-mpnet-base-v2`) — compare embedding quality on the same 5 sentences

**Comprehension checkpoint:**
1. What does cosine similarity actually measure? Why do we use it instead of Euclidean distance?
2. If you embed "dog" and "cat" — do you expect their vectors to be similar or different? Why?
3. Why do all embeddings from the same model have the same dimension (e.g. 384)?

---

### Phase 4 — Vector Store
**What you'll build:** A database that stores your embeddings and lets you query them by similarity.

**Your tasks:**
- [ ] Write `src/vectorstore/chroma_store.py` — functions: `create_collection`, `add_documents`, `query`
- [ ] Take your chunks + embeddings from Phases 2 & 3 and store them in ChromaDB
- [ ] Query it with a question — print the top-5 returned chunks
- [ ] Look at the raw `chroma_db/` folder that already exists — understand what's in those `.bin` files (hint: HNSW index)
- [ ] Run the same query twice — confirm results are deterministic

**Comprehension checkpoint:**
1. What is an HNSW index? Why do vector stores use it instead of comparing every vector to every other vector?
2. ChromaDB stores both the embedding AND the original text. Why do we need to store both?
3. If you had 10 million chunks, what problems would you start to hit with ChromaDB?

---

### Phase 5 — Retrieval
**What you'll build:** A retriever that takes a user question and returns the most relevant chunks.

**Your tasks:**
- [ ] Write `src/retrieval/semantic_retriever.py` — embed the query, search ChromaDB, return top-k chunks
- [ ] Write `src/retrieval/bm25_retriever.py` — keyword-based retrieval using `rank_bm25`
- [ ] Write `experiments/phase_5_retrieval/compare.py` — for 10 test questions, compare what each retriever returns
- [ ] Write `src/retrieval/hybrid_retriever.py` — combine semantic + BM25 results using Reciprocal Rank Fusion (RRF)
- [ ] Test edge cases: very short queries, very long queries, queries with typos

**Comprehension checkpoint:**
1. When would BM25 outperform semantic search? Give a concrete example.
2. What is Reciprocal Rank Fusion and why is it a simple but effective way to combine ranked lists?
3. A user searches for "Q3 revenue" — the document says "third-quarter earnings." Which retriever finds it, and why?

---

### Phase 6 — Reranking
**What you'll build:** A second-pass scorer that re-orders retrieved chunks by true relevance.

**Your tasks:**
- [ ] Install `sentence-transformers` cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- [ ] Write `src/reranking/cross_encoder.py` — takes (query, list of chunks) → returns chunks sorted by score
- [ ] Compare: take top-10 from Phase 5 retriever → rerank → take top-3. Do the top-3 change?
- [ ] Measure latency: retrieval alone vs retrieval + reranking — what's the tradeoff?

**Comprehension checkpoint:**
1. Why can't we just use the cross-encoder directly for retrieval (skipping Phase 5 entirely)?
2. What is the difference between a bi-encoder (used in Phase 3) and a cross-encoder?
3. Why do we retrieve top-20 then rerank to top-5, rather than just retrieving top-5 directly?

---

### Phase 7 — Generation
**What you'll build:** The final answer generation step — take retrieved chunks + user question → answer.

**Your tasks:**
- [ ] Install and run Ollama locally — pull `llama3.2` (3B, fast enough for experiments)
- [ ] Write `src/llm/prompt_builder.py` — formats: system prompt + context chunks + user question
- [ ] Write `src/llm/ollama_client.py` — sends the prompt, streams the response
- [ ] Write `src/llm/response_generator.py` — orchestrates prompt building + LLM call
- [ ] Test: ask a question your document can answer vs. a question it cannot — how does the LLM behave?
- [ ] Experiment with the prompt: what happens if you remove "only answer from the provided context"?

**Comprehension checkpoint:**
1. What is "hallucination" in this context and why does RAG reduce (but not eliminate) it?
2. Your prompt includes retrieved chunks — what's the risk if you include too many chunks?
3. What is the difference between "faithfulness" and "relevance" when evaluating a RAG answer?

---

### Phase 8 — Evaluation
**What you'll build:** A way to measure whether your RAG system actually works.

**Your tasks:**
- [ ] Create a small "golden dataset": 10 questions + the correct answers + which document passages contain the answer
- [ ] Write `src/evaluation/hit_rate.py` — for each question, did the correct passage appear in top-k?
- [ ] Write `src/evaluation/precision_at_k.py` — of the top-k results, what fraction are relevant?
- [ ] Write `src/evaluation/benchmark_runner.py` — runs all questions through your pipeline and computes all metrics
- [ ] Use your benchmark to compare: (Phase 2 chunking strategy A) vs (chunking strategy B) — which gives better retrieval?
- [ ] Run the benchmark before and after adding the reranker from Phase 6 — does it improve metrics?

**Comprehension checkpoint:**
1. Why do you need a "golden dataset" to evaluate RAG? Can't you just read the answers and judge?
2. What is the difference between Precision@k and Hit Rate@k?
3. Your Hit Rate@5 is 0.7 but Precision@5 is 0.3 — what does this tell you about your retriever?

---

### Phase 9 — Full Pipeline
**What you'll build:** Wire everything together into a single pipeline object.

**Your tasks:**
- [ ] Write `src/pipeline/rag_pipeline.py` — a class that: loads docs → chunks → embeds → stores → retrieves → reranks → generates
- [ ] Write `demo/cli_chat.py` — interactive terminal chat over your document collection
- [ ] Profile the full pipeline — where is time being spent?
- [ ] Test with a completely new document the system has never seen

---

### Phase 10 — Production API
**What you'll build:** A FastAPI server that exposes your pipeline as an HTTP API.

**Your tasks:**
- [ ] Write `api/app.py` and `api/routes.py` — POST /query endpoint
- [ ] Add `api/schemas.py` — Pydantic models for request/response
- [ ] Write a `docker-compose.yml` that starts your API + ChromaDB
- [ ] Write `api/middleware.py` — add request logging and latency tracking

---

## Key Experiments to Run (Optional but Valuable)

| Experiment | Question it answers |
|---|---|
| `experiments/chunking_comparison/` | Does chunk size affect retrieval quality? |
| `experiments/embedding_comparison/` | Is a bigger model always better? |
| `experiments/long_context_vs_rag/` | When is just using a big context window better than RAG? |
| `experiments/retrieval_quality/` | How does hybrid search compare to pure semantic? |

---

## Environment Setup

```bash
# Python venv already created — activate it
venv\Scripts\activate   # Windows

# Install dependencies as you need them, phase by phase
pip install langchain-text-splitters   # Phase 2
pip install sentence-transformers      # Phase 3
pip install chromadb                   # Phase 4
pip install rank-bm25                  # Phase 5
pip install ollama                     # Phase 7
pip install fastapi uvicorn            # Phase 10
```

---

## Current Status

**Phase 1** — Not started (but `text_splitters.py` gives you a head start on Phase 2)
