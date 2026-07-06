# RAG From Scratch — Learning Q&A Reference

A record of key questions and answers from building this RAG system from scratch.
Useful for anyone learning RAG concepts hands-on.

---

## Baseline — Before Starting

**Q: What problem does RAG solve? Why not just give the LLM a huge document directly?**

RAG solves two problems:
1. **Private/local data** — the LLM was never trained on your documents, so it simply doesn't know what's in them
2. **Knowledge cutoff** — the LLM doesn't know recent events past its training date

RAG doesn't make the LLM "weight" your data more — it literally puts the relevant text into the prompt so the LLM reads it at generation time. The LLM has no memory between calls; RAG is how you inject the right context for each question.

---

**Q: What are the three main phases of a RAG pipeline?**

1. **Indexing** — load documents, chunk them, embed the chunks, store in a vector database
2. **Retrieval** — embed the user's query, find the most similar chunks
3. **Augmented Generation** — pass the retrieved chunks + query to the LLM, get an answer

---

**Q: What does chunk overlap do? Why does setting it to 0 hurt retrieval quality?**

Without overlap, a sentence or idea that spans the boundary between two chunks gets split across both. Example:

```
Chunk 1: "...the company's revenue grew 40% in Q3."
Chunk 2: "This growth was driven by expansion into Southeast Asia..."
```

Chunk 2 starts with "This growth" — but what growth? The word "This" refers to something in Chunk 1 that wasn't retrieved. The LLM gets a dangling reference with no anchor.

With overlap, the tail of Chunk 1 is repeated at the start of Chunk 2, so no sentence gets severed at a boundary. Overlap prevents dangling references.

---

## Phase 1 — Document Ingestion

**Q: Why do we load documents as raw text first rather than going straight to chunks?**

Two reasons:
1. The LLM has a limited context window — sending 1400 pages would blow past it
2. Even with unlimited context, burying the LLM in irrelevant text makes its answers worse — it loses focus

We load first so we can control exactly what text gets passed to the chunker and eventually to the LLM.

---

**Q: What do you lose when extracting text from a PDF that you wouldn't lose from a plain .txt file?**

Three things:
1. **Images and diagrams** — chemical structural formulas, figures, graphs — `pdfplumber` silently skips them
2. **Table structure** — the words survive but rows and columns are destroyed. "Feature | Prokaryotic | Eukaryotic" becomes "Feature Prokaryotic Eukaryotic" — the relational meaning is gone
3. **Page number context** — page numbers get embedded as noise in the text stream, and you lose the ability to cite "this came from page 47" once everything is flattened into one string

---

**Q: Your extracted text has tables. A student asks "what are the differences between prokaryotic and eukaryotic cells?" — what specific problem does flat text extraction cause?**

The table content is present but the structure is destroyed. Instead of:

| Feature | Prokaryotic | Eukaryotic |
|---|---|---|
| Nucleus | Absent | Present |
| Size | 1–10 µm | 10–100 µm |

The extractor produces:

```
Feature Prokaryotic Eukaryotic Nucleus Absent Present Size 1–10 µm 10–100 µm
```

The LLM sees a jumble of words with no clear structure to reason from. It can't reliably reconstruct which value belongs to which organism and which feature.

---

**Q: Where should you validate that a file exists — inside `load_pdf()` or before calling it?**

Inside `load_pdf()`. The function should own its own validation — if the file doesn't exist, it raises a clear error. The caller shouldn't need to know how to validate; it just calls the function and handles the error if one comes back.

```python
if not os.path.exists(file_path):
    raise FileNotFoundError(f"PDF not found: {file_path}")
```

This way, any caller benefits automatically. Validating only at the caller means every new caller has to remember to do it themselves — fragile.

---

## Phase 2 — Chunking

**Q: What does `chunk_size` control and what do `separators` control? How do they work together?**

- **Separators** control *where the text is allowed to be cut* (paragraph break, line break, space, character)
- **chunk_size** controls *the maximum size a chunk can be before the splitter tries a smaller separator*

They work together: the splitter tries `"\n\n"` first, produces pieces, then checks — is each piece under `chunk_size`? If yes → done. If no → try `"\n"`. And so on. `chunk_size` is the exit condition for the recursion. Without it, the splitter wouldn't know when to stop.

chunk_size is a **maximum**, not an exact size. Chunks can be smaller.

---

**Q: Why does `RecursiveCharacterTextSplitter` try separators in this specific order: `["\n\n", "\n", " ", ""]`?**

Each separator is a smaller boundary than the previous:
- `"\n\n"` — paragraph break (most natural, preserves the most meaning)
- `"\n"` — line break
- `" "` — word boundary
- `""` — individual character (last resort, only for words longer than chunk_size)

The splitter always tries the most natural boundary first. It only falls back to smaller units when absolutely necessary. This preserves meaning better than fixed-size splitting — it never cuts mid-sentence if it can avoid it.

---

**Q: If chunk_size is too small, what happens? What if it's too large?**

- **Too small** — a chunk might contain half a sentence or one isolated fact with no surrounding context. The retriever finds it but the LLM can't make sense of it on its own.
- **Too large** — a chunk covers multiple topics. The retriever finds it for one topic, but the LLM gets fed irrelevant content about the other topics too.

The goal is a chunk that contains **one complete idea** — enough context to be understood on its own, focused enough to be about one thing.

---

**Q: Comparing three chunking strategies on biology text — why did recursive produce the best chunks?**

Recursive respects **document structure** (paragraphs, sections) because it splits on `"\n\n"` first. Topics in textbooks are separated by paragraph breaks, so recursive splitting naturally keeps related content together.

Sentence splitting ignores document structure — it just counts characters between punctuation marks. So a chunk can mix content from different sections (biology paragraph + review questions + MCQ fragments).

Fixed splitting ignores everything — it cuts blindly at character positions regardless of where ideas begin and end.

---

**Q: When would you choose fixed-size splitting over recursive in a real project?**

When semantic meaning doesn't matter — for example, simple keyword search, pipeline speed testing, or logging data. For any Q&A system where meaning matters (like MDCAT prep), always use recursive.

---

**Q: The sentence chunks mixed biology content with Review Questions and MCQ fragments. What could you do to prevent this?**

Clean the document **before** chunking. Strip out noise that doesn't belong in a Q&A knowledge base:
- Repeated footers ("Access for free at openstax.org")
- Figure captions ("FIGURE 27.3 Development of...")
- Review question sections
- Page numbers embedded in the text stream

This is the job of `src/ingestion/document_cleaner.py` — run it after loading, before chunking.

---

**Q: Why is overlap harder to implement for sentence-based splitting than for fixed-size splitting?**

In fixed-size splitting, overlap is simple arithmetic: "go back exactly 200 characters." You always land on the exact overlap amount.

In sentence splitting, you can't go back half a sentence — you must repeat whole sentences. If you want ~200 character overlap, you might repeat 1 sentence (150 chars) or 3 sentences (400 chars). You can never land exactly on the target. Sentences are variable length, so overlap becomes approximate rather than exact.

---

## Phase 3 — Embeddings

**Q: Why should the embedding model be loaded at module level instead of inside the function?**

Loading the model inside the function means it reloads from disk on every single call — every embedding takes 2–3 seconds just for setup. At module level, the model loads once when the file is imported and stays in memory. All subsequent calls are fast because the model is already there.

```python
# wrong — reloads every call
def get_embedding(text):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(text).tolist()

# right — loads once at import time
model = SentenceTransformer("all-MiniLM-L6-v2")
def get_embedding(text):
    return model.encode(text).tolist()
```

---

**Q: What is batch embedding and why is it faster than embedding one chunk at a time?**

`model.encode(texts)` accepts a list and processes all items together in one GPU/CPU pass. The model overhead (loading weights into compute units, setting up the computation graph) happens once for the whole batch instead of once per chunk.

Embedding 4911 chunks one at a time: ~8 minutes. Batch: ~2 minutes. Same model, same hardware — just fewer round-trips to the compute engine.

---

**Q: What is an embedding dimension? What does 384 mean for all-MiniLM-L6-v2?**

A dimension is one axis of meaning in an abstract semantic space. 384 dimensions means each chunk is represented as 384 numbers, where each number captures some learned pattern of meaning across language.

Dimensions are NOT character slots. A 1000-character chunk and a 100-character chunk both produce exactly 384 numbers. The model compresses meaning, not length. Think of it like describing a movie with genre (0.9 = action), tone (0.2 = dark), pace (0.8 = fast) — a few numbers capture the semantic "position" of the content without storing the content itself.

---

**Q: How does a whole chunk (many words) get compressed into a single vector?**

Two steps:
1. The transformer processes each token and creates a vector for it, influenced by every other token in the chunk via attention. "Function" in a biology sentence gets a different vector than "function" in a math sentence.
2. All token vectors are averaged together — this is called **mean pooling**. The result is one 384-dimension vector representing the chunk's semantic center of gravity.

This is why very long chunks with many different topics produce blurry vectors — the mean-pooled average points at the "center" of many ideas rather than sharply at one.

---

**Q: What is cosine similarity and what range does it return?**

Cosine similarity measures the angle between two vectors, not their magnitude. Range: -1 to 1.
- 1.0 = identical direction (same meaning)
- 0.0 = perpendicular (unrelated)
- -1.0 = opposite directions (opposite meaning)

For text, most pairs score between 0.0 and 1.0 since embeddings rarely point in opposite directions.

---

**Q: Why cache embeddings to a JSON file instead of recomputing every run?**

Embedding 4911 chunks takes ~2–8 minutes. The chunks don't change between runs — the document is static. Re-embedding the same text every time wastes that time on every run.

The pattern: check if the cache file exists → load from it. If not → embed and save. This is the "embed once, reuse forever" pattern used in production pipelines too (pre-indexing).

---

## Phase 4 — Vector Store (ChromaDB)

**Q: Why do we need a vector database at all? Why not just compute cosine similarity against all chunks in memory?**

For 4911 chunks it would work fine. But at 1 million chunks, computing similarity against every single vector on every query would take seconds per search. Vector databases use approximate nearest neighbor algorithms (like HNSW) that can search millions of vectors in milliseconds by building a graph index — you trade a tiny bit of accuracy for massive speed gains.

---

**Q: What is the difference between `chromadb.Client()` and `chromadb.PersistentClient()`?**

- `Client()` — in-memory only. Data is lost when the process ends.
- `PersistentClient(path=...)` — writes to disk. Data survives restarts.

For development and learning, in-memory is fine. For any real system where you embed once and query many times, you must use PersistentClient.

---

**Q: Why use `get_or_create_collection` instead of `create_collection`?**

`create_collection` crashes if the collection already exists. `get_or_create_collection` returns the existing one if it exists, creates it if it doesn't. This makes the code safe to run multiple times — no crash, no duplicate creation. Always prefer it unless you explicitly need to fail on an existing collection.

---

**Q: ChromaDB has a `query_texts` parameter. Why do we use `query_embeddings` instead?**

`query_texts` tells ChromaDB to embed the query itself — meaning ChromaDB calls its own internal embedding model. But we already have our embedding model (`all-MiniLM-L6-v2`) loaded and have used it to embed all our chunks. If we used `query_texts`, ChromaDB might use a different embedding model, and embeddings from different models are not comparable — they live in completely different vector spaces. Always embed the query with the same model used to embed the chunks, then pass `query_embeddings`.

---

**Q: What does HNSW stand for and what does it do?**

Hierarchical Navigable Small World — a graph-based index for approximate nearest neighbor search. Instead of scanning all vectors, it builds a layered graph where each vector connects to its nearest neighbors. At query time, it navigates the graph to find close vectors without checking every single one. Fast but approximate — it might miss the single closest vector but finds a very close one in milliseconds.

---

## Phase 5 — Retrieval

**Q: What is the core limitation of semantic (vector) search?**

Semantic search finds chunks that *talk about similar things*, not chunks that *answer your question*. It measures how close two vectors are in meaning-space — not whether the chunk is factually relevant to the query.

Example: "what is the role of mitochondria?" returned a ribosome chunk ("essential function of all cells... ribosomes"). Both are about organelle function, so their vectors are close — but the ribosome chunk doesn't answer the question at all.

---

**Q: Semantic search returned a ribosome chunk for a mitochondria question. Which retrieval techniques would specifically prevent this?**

Two direct fixes:
1. **BM25 (hybrid retrieval)** — BM25 is keyword-based. The word "mitochondria" doesn't appear in the ribosome chunk, so BM25 would score it near zero and drop it.
2. **Reranking** — a reranker reads the question AND the chunk together and scores whether this chunk actually answers the question. It would immediately flag the mismatch.

Chunk size reduction is a general quality improvement but wouldn't have specifically prevented this failure — even a small ribosome chunk would still have a vector close to "role of mitochondria."

---

**Q: What is reranking and how is it different from embedding similarity?**

| | Embedding search | Reranking |
|---|---|---|
| What it sees | Question vector vs chunk vector separately | Question + chunk together in one pass |
| Speed | Very fast | Slower |
| Accuracy | Finds similar topics | Checks if chunk actually answers the question |

Reranking is a second stage: retrieve 20 candidates fast with semantic/BM25, then rerank those 20 to find the best 5. You pay the speed cost only on the small candidate set.

---

**Q: Why does BM25 not need ChromaDB or any vector database?**

BM25 is pure keyword counting — it doesn't use vectors at all. It tokenizes chunks, counts word frequencies, and scores chunks based on how many query words they contain weighted by rarity. No vectors = no need to store or search vectors = no vector database needed. BM25 operates directly on raw text lists in memory.

---

**Q: What are the five levers for achieving high retrieval precision in an MDCAT RAG system?**

1. **Hybrid retrieval** — combine semantic (handles paraphrases) + BM25 (handles exact keywords). Covers both failure modes.
2. **Reranking** — re-scores top candidates by reading question + chunk together. Strongest fix for false positives.
3. **Better chunking** — smaller, topic-focused chunks produce sharper vectors. For MDCAT's short factual questions, 300–500 chars may outperform 1000.
4. **Metadata filtering** — tag chunks by subject (Biology/Chemistry/Physics). Filter before retrieval so a biology question only searches biology chunks.
5. **Domain-specific embedding model** — a model fine-tuned on medical/biology text places medical concepts more accurately in vector space than a general-purpose model.

---

## Key Concepts Summary

| Concept | One-line explanation |
|---|---|
| RAG | Inject relevant document text into the LLM prompt at query time |
| Chunk | A piece of the document small enough to be meaningfully retrieved |
| chunk_size | Maximum character length of one chunk |
| chunk_overlap | Characters repeated between adjacent chunks to prevent boundary cutoffs |
| Separators | Preferred cut points tried in order from largest to smallest boundary |
| Recursive splitting | Try paragraph → line → word → character until chunk fits |
| Fixed-size splitting | Cut every N characters regardless of content |
| Sentence splitting | Group complete sentences until combined length reaches chunk_size |
| Vector store | Database that stores embeddings and retrieves by similarity |
| Embedding | A list of numbers representing the meaning of a piece of text |
| Embedding dimension | One axis of meaning in semantic space — not a character slot |
| Mean pooling | Averaging all token vectors into one vector for the whole chunk |
| Cosine similarity | Angle between two vectors — measures semantic closeness (-1 to 1) |
| Embedding cache | Pre-compute embeddings once, save to JSON, reuse on every run |
| HNSW | Graph-based approximate nearest neighbor index used inside ChromaDB |
| PersistentClient | ChromaDB client that saves data to disk across restarts |
| query_embeddings | Pass your own precomputed embedding — must match the model used to embed chunks |
| Semantic retrieval | Find chunks with similar meaning via cosine similarity on vectors |
| BM25 | Keyword-based retrieval scoring chunks by word frequency — no vectors needed |
| Hybrid retrieval | Combine semantic + BM25 to cover both meaning and keyword failures |
| Reranking | Second stage that re-scores retrieved candidates by reading question + chunk together |
