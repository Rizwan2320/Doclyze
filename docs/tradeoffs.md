# Doclyze — Technical Tradeoffs & Design Decisions

Every decision in this project involved a tradeoff. This document explains what we chose, what we rejected, and why. This is the thinking behind the code.

---

## 1. PDF Loader: PyMuPDF vs Unstructured

### What we chose

PyMuPDF (`fitz`)

### What we rejected

Unstructured (both `fast` and `hi_res` strategies)

### Why

|                          | PyMuPDF      | Unstructured (fast) | Unstructured (hi_res) |
| ------------------------ | ------------ | ------------------- | --------------------- |
| Speed                    | ~2 seconds   | ~2 seconds          | ~25 minutes           |
| Simple PDFs              | ✅ Excellent | ✅ Good             | ✅ Excellent          |
| Two-column academic PDFs | ✅ Correct   | ❌ Misses content   | ✅ Correct            |
| Tables                   | ⚠️ Basic     | ✅ Good             | ✅ Excellent          |
| Scanned PDFs (OCR)       | ❌ No        | ⚠️ Limited          | ✅ Yes                |
| Dependency weight        | Lightweight  | Heavy               | Very heavy            |

Unstructured `fast` missed the entire body content of two-column academic PDFs — it only extracted references. Unstructured `hi_res` worked but took 25 minutes per document, making it unusable for a web portal.

PyMuPDF processes any PDF in under 2 seconds and handles two-column layouts correctly.

### What we gave up

Table extraction quality. PyMuPDF treats tables as plain text. For documents where table data matters (financial reports, data sheets), Unstructured or a dedicated table extractor would be better.

### When to revisit

Add Unstructured `hi_res` as an optional slow-but-thorough mode for documents where table extraction matters and processing time is acceptable.

---

## 2. Chunking Strategy: Hierarchical vs Semantic vs Simple

### What we chose

Hierarchical chunking — small chunks (600 chars, 100 overlap) + parent chunks (2000 chars, 200 overlap)

### What we rejected

- Simple `RecursiveCharacterTextSplitter` only
- Semantic chunking (embedding-based boundary detection)

### Why

|                           | Simple | Hierarchical | Semantic                    |
| ------------------------- | ------ | ------------ | --------------------------- |
| Implementation complexity | Low    | Medium       | High                        |
| Retrieval precision       | Medium | High         | High                        |
| Context for LLM           | Low    | High         | Medium                      |
| Compute cost              | Low    | Low          | High (needs embedding pass) |
| Predictable behavior      | ✅ Yes | ✅ Yes       | ⚠️ Variable                 |

Simple chunking alone produces chunks that are too small for meaningful context. Semantic chunking requires an additional embedding pass during ingestion and produces unpredictable chunk sizes.

Hierarchical gives us the best of both: small chunks for precise retrieval, parent chunks for richer context when needed.

### What we gave up

Semantic boundary detection. Our chunker splits at character count, not topic boundaries. A chunk can split mid-sentence or mid-concept. Semantic chunking would fix this but adds significant ingestion complexity and cost.

### When to revisit

When eval scores plateau and inspection shows chunks splitting at bad boundaries.

---

## 3. Embeddings: Local vs API

### What we chose

Local: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)

### What we rejected

OpenAI `text-embedding-3-small` or `text-embedding-3-large` via API

### Why

|                          | all-MiniLM-L6-v2               | OpenAI text-embedding-3-small |
| ------------------------ | ------------------------------ | ----------------------------- |
| Cost                     | Free                           | ~$0.02 per 1M tokens          |
| Latency                  | ~10s first load, then fast     | API round-trip per batch      |
| Privacy                  | Data never leaves your machine | Data sent to OpenAI           |
| Quality (MTEB benchmark) | Good                           | Excellent                     |
| Offline use              | ✅ Yes                         | ❌ No                         |
| Dimension                | 384                            | 1536                          |

For a learning project where cost and privacy matter, local embeddings are the right default. The quality difference is real but not large enough to matter until you have a proper eval pipeline measuring it.

### What we gave up

Raw embedding quality. OpenAI's models score higher on retrieval benchmarks. For production systems handling critical documents, the quality improvement may justify the cost and API dependency.

### When to revisit

When eval scores show retrieval recall is the bottleneck and not chunking or prompt quality.

---

## 4. Vector Store: ChromaDB vs Alternatives

### What we chose

ChromaDB (persistent, local)

### What we rejected

- FAISS (in-memory)
- Qdrant (self-hosted or cloud)
- Pinecone (cloud)
- Weaviate (self-hosted or cloud)

### Why

|                            | ChromaDB        | FAISS     | Qdrant             | Pinecone   |
| -------------------------- | --------------- | --------- | ------------------ | ---------- |
| Setup complexity           | Low             | Low       | Medium             | Low        |
| Persistence                | ✅ Built-in     | ❌ Manual | ✅ Yes             | ✅ Yes     |
| Metadata filtering         | ✅ Yes          | ❌ No     | ✅ Excellent       | ✅ Yes     |
| Hybrid search (BM25+dense) | ❌ Not built-in | ❌ No     | ✅ Yes             | ⚠️ Limited |
| Scale                      | Medium          | High      | High               | Very High  |
| Cost                       | Free            | Free      | Free (self-hosted) | Paid       |
| Collection isolation       | ✅ Native       | ❌ Manual | ✅ Yes             | ✅ Yes     |

ChromaDB is the simplest path to a persistent, metadata-aware vector store with native collection isolation. For Phase 1 learning, it removes infrastructure complexity without sacrificing correctness.

### What we gave up

Hybrid search. ChromaDB doesn't natively support BM25 + dense vector fusion. Qdrant does this out of the box. When we add hybrid search in Phase 2, we'll evaluate migrating to Qdrant.

### When to revisit

Phase 2 — when adding hybrid search (BM25 + dense + Reciprocal Rank Fusion).

---

## 5. LLM: Groq vs OpenAI vs Local (Ollama)

### What we chose

Groq (`llama-3.3-70b-versatile`)

### What we rejected

- OpenAI GPT-4o
- Anthropic Claude
- Local Ollama (llama3, mistral)

### Why

|             | Groq (llama-3.3-70b) | GPT-4o              | Ollama (local)          |
| ----------- | -------------------- | ------------------- | ----------------------- |
| Speed       | Extremely fast (~1s) | Medium (~3-5s)      | Slow on CPU             |
| Cost        | Free tier generous   | ~$5 per 1M tokens   | Free                    |
| Quality     | Excellent            | Best in class       | Good (depends on model) |
| Privacy     | Data sent to Groq    | Data sent to OpenAI | ✅ Local                |
| Reliability | High                 | High                | Depends on hardware     |

Groq's free tier provides fast, high-quality inference that makes the development loop tight. Waiting 5 seconds per query during debugging is painful. Groq's ~1 second responses make iteration fast.

### What we gave up

Peak quality. GPT-4o and Claude 3.5 Sonnet outperform llama-3.3-70b on complex reasoning. For a learning project, the quality difference is acceptable. For production systems requiring highest accuracy, this is worth revisiting.

### When to revisit

When eval scores show generation quality (not retrieval) is the bottleneck.

---

## 6. Reranker: Removed vs Kept

### What we chose

No reranker — direct similarity search results go to LLM

### What we rejected

`CrossEncoderReranker` with `cross-encoder/ms-marco-MiniLM-L-6-v2`

### Why

The reranker implementation was broken. It used `ContextualCompressionRetriever` which did independent retrieval, ignoring the initial similarity search results entirely. It was adding latency and complexity while providing zero benefit.

The correct reranker implementation would:

1. Fetch k=30 candidates via similarity search
2. Score all candidates with cross-encoder
3. Return top-n by cross-encoder score

We removed it because a broken complex system is worse than a simple working one. With k=15 and a good prompt, retrieval quality is sufficient for Phase 1.

### What we gave up

Better retrieval precision. Cross-encoders significantly outperform bi-encoders (what we use) on relevance scoring. This is one of the highest-leverage improvements available.

### When to revisit

Phase 2 — implement correctly as: `similarity_search(k=30)` → `cross_encoder.predict(pairs)` → `sort by score` → `take top 6`.

---

## 7. Document Isolation: Per-Collection vs Metadata Filtering

### What we chose

Per-document ChromaDB collections (one collection per `document_id`)

### What we rejected

Single collection with `document_id` as metadata filter

### Why

|                           | Per-collection          | Metadata filtering              |
| ------------------------- | ----------------------- | ------------------------------- |
| Query isolation           | ✅ Perfect              | ⚠️ Filter must be correct       |
| Implementation complexity | Low                     | Low                             |
| Scale (1000s of docs)     | ⚠️ Many collections     | ✅ Better                       |
| Cross-document search     | ❌ Requires aggregation | ✅ Natural                      |
| Accidental contamination  | ✅ Impossible           | ⚠️ Possible if filter forgotten |

For Phase 1 with a small number of documents, per-collection isolation is safer and simpler. Accidental cross-document contamination is impossible — you literally cannot query another document's collection without its UUID.

### What we gave up

Cross-document search and scalability. If you want to ask "find all documents that mention X", per-collection architecture requires querying each collection separately and aggregating results. At thousands of documents, this becomes slow.

### When to revisit

When adding cross-document search or when the number of collections becomes a management burden.

---

## 8. API Design: Synchronous vs Async Ingestion

### What we chose

Synchronous ingestion — upload blocks until ingestion completes, returns result

### What we rejected

Async ingestion — upload returns immediately with a job ID, client polls for status

### Why

The schemas were built for async (we have `UploadAcceptedResponse`, `status_url`, `IngestionStatus` enum). But the routes implement synchronous ingestion. This mismatch was a conscious tradeoff.

For Phase 1 with small documents (under 50MB), synchronous ingestion completes in under 30 seconds. Adding a job queue (Celery, Redis, background tasks) adds significant infrastructure complexity with little benefit at this scale.

### What we gave up

Non-blocking uploads for large documents. A 200-page PDF takes 10-15 seconds to ingest. During this time the HTTP connection is held open. For a production portal with many concurrent users, this is a problem.

### When to revisit

Phase 2 — when adding support for very large documents or when concurrent upload load becomes a concern. The schemas are already designed for it.

---

## Summary Table

| Decision      | Chose              | Rejected        | Main Tradeoff                      |
| ------------- | ------------------ | --------------- | ---------------------------------- |
| PDF Loader    | PyMuPDF            | Unstructured    | Speed vs table quality             |
| Chunking      | Hierarchical       | Semantic        | Predictability vs boundary quality |
| Embeddings    | Local MiniLM       | OpenAI API      | Cost/privacy vs raw quality        |
| Vector Store  | ChromaDB           | Qdrant/Pinecone | Simplicity vs hybrid search        |
| LLM           | Groq llama-3.3-70b | GPT-4o          | Speed/cost vs peak quality         |
| Reranker      | None (removed)     | CrossEncoder    | Simplicity vs retrieval precision  |
| Doc Isolation | Per-collection     | Metadata filter | Safety vs scalability              |
| Ingestion     | Synchronous        | Async + polling | Simplicity vs concurrency          |

Every rejected option is on a "when to revisit" list — not thrown away, just deferred until the simpler version proves insufficient.

That's the engineering discipline: make the simplest thing work first, measure whether it's good enough, then add complexity only when the data demands it.


---

## 9. Chunking: RecursiveCharacterTextSplitter vs NLTK Sentence-Aware

### What we chose (updated 2026-05-14)

NLTK `sent_tokenize` + sentence-aware grouping for child chunks. Parent chunks still use `RecursiveCharacterTextSplitter`.

### What we rejected (previous default)

`RecursiveCharacterTextSplitter` with `separators=["\n\n", "\n", ". ", " ", ""]` for child chunks.

### Why we changed

| | RecursiveCharacterTextSplitter | NLTK Sentence-Aware |
|---|---|---|
| Sentence boundary guarantee | ❌ No — falls back to char truncation | ✅ Yes — pre-split by sentences |
| Academic paper compatibility | ❌ Fails on `.\n` patterns | ✅ Handles all punctuation |
| Implementation complexity | Low | Medium (requires NLTK dependency) |
| Chunk size predictability | Exact | Approximate (grouped by sentences) |
| Overlap quality | Character-based, may split sentences | Sentence-based, preserves boundaries |

The old splitter cut the three-limitations sentence at "knowledge f" (mid-word) and "their predictions" (mid-sentence). This caused q09 hallucination. NLTK guarantees whole sentences, eliminating this failure mode.

### What we gave up

Exact chunk sizes. Sentence-aware chunks vary in size (e.g., 850-1100 chars instead of exactly 1000). This is acceptable because the variance is small and the boundary guarantee is more important.

### When to revisit

If chunk size variance causes embedding quality issues (very long sentences creating oversized chunks). Could switch to spaCy for better sentence segmentation, or add a max-sentence-length split.

---

## 10. Retrieval: Child-Only vs Parent-Only vs Hybrid

### What we chose

Child-only retrieval (`chunk_type="child"`) with k=10.

### What we tested and rejected

- **Parent-only retrieval:** Fixed q09 boundary split but triggered parametric memory override (q03 hallucinated fake paper sections). Coherent 2000-char excerpts look like "the actual paper" to the LLM, causing it to answer from training data.
- **Child→parent hybrid:** Architecturally correct (small chunks for precision, parents for context) but `store.get(ids=...)` crashed silently in our LangChain-Chroma version. Requires debugging Chroma API interaction.

### Why child-only is the current default

| | Child-only | Parent-only | Child→parent hybrid |
|---|---|---|---|
| Retrieval precision | High | Low | High |
| Context completeness | Low | High | High |
| Parametric memory risk | Low | High | Medium |
| Implementation complexity | Simple | Simple | Medium |
| Current reliability | ✅ Working | ⚠️ Triggers hallucinations | ❌ Crashes |

### What we gave up

Context completeness. Child chunks (600-1000 chars) may not contain full paragraphs or tables. The LLM sometimes lacks surrounding context to fully answer.

### When to revisit

Week 3 — implement child→parent hybrid correctly with defensive fallback. Or add a re-ranker that promotes chunks with more complete context.

---

## 11. Prompt Engineering: Current vs Hardened

### What we chose (current)

System prompt with basic constraints (no chunk tags, refuse if no answer, no filler).

### What we need (pending)

Explicit anti-hallucination and anti-leakage constraints:

```python
("system", """You are an expert technical research assistant.
Your task is to answer the user's question using ONLY the provided context.

CRITICAL RULES:
1. NEVER mention the word "Chunk" or include raw "[Chunk X | Page Y]" tags.
2. NEVER output raw mathematical notation, LaTeX, or code fragments from the context unless directly asked.
3. If the context does not contain the answer, say EXACTLY: "I cannot answer this based on the provided document." Then STOP. Do not add any other text.
4. Do NOT use your general knowledge. If the answer is not in the Context, you MUST refuse.
5. Do not include introductory filler like "Based on the context..."
""")