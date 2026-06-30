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
