# Doclyze — Project State

## What Is Built & Working

- Full ingestion pipeline: PDF → PyMuPDF → filter noise → hierarchical chunker → ChromaDB
- Document isolation: each upload gets its own ChromaDB collection (keyed by document_id)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, free)
- RAG chain: similarity search (k=15) → Groq llama-3.3-70b → answer
- FastAPI: /api/upload, /api/query, /api/summarize all working
- Swagger UI: localhost:8000/docs
- GitHub: github.com/Rizwan2320/Doclyze
- Tested on: Wikipedia PDF ✅, RAG paper (2-column academic PDF) ✅

## Key Technical Decisions

- PyMuPDF (fitz) for PDF loading — Unstructured fast strategy missed content in 2-col PDFs
- Filter chunks < 50 chars — removes browser-print/header noise
- Hierarchical chunks: small (600) + parent (2000)
- No reranker (removed broken ContextualCompressionRetriever)
- LangSmith disabled (LANGSMITH_TRACING=false)
- document_id used as ChromaDB collection_name for isolation

## What Remains for Doclyze

1. Evaluation harness — 5 test Q&A pairs, scored objectively
2. Write README
3. Deploy to Hugging Face Spaces

## Known Issues

- /api/query requires document_id — no default collection fallback
- sources: [] not populated in QueryResponse
- No deduplication on re-upload

## Stack

Python 3.12, uv, FastAPI, LangChain, ChromaDB, PyMuPDF, Groq, Loguru
Windows, VS Code

## Deployed

- Live URL: https://rizwan444-doclyze.hf.space
- Swagger UI: https://rizwan444-doclyze.hf.space/docs
- Platform: Hugging Face Spaces (Docker)


```markdown
## Updated State — 2026-05-14

### What Changed Since Last Update

| Change | Date | Impact |
|---|---|---|
| Sentence-aware chunking (NLTK) | 2026-05-14 | Fixed q09 boundary-split hallucination |
| Gold dataset v2 generation | 2026-05-14 | Chunk IDs now align with actual DB state |
| Eval harness validated | 2026-05-14 | Metrics now measure real behavior, not UUID drift |
| Parent-only retrieval tested | 2026-05-14 | Rejected — triggers parametric memory override |
| Child→parent hybrid attempted | 2026-05-14 | Rejected — crashes on `store.get(ids=...)` |

### Current Eval Results (10-question baseline)

| ID | Category | Chunk P | Chunk R | Answer Correct? | Notes |
|---|---|---|---|---|---|
| q01 | factual | 0.00 | 0.000 | ❌ Incomplete | Embedding model failure — Week 2 |
| q02 | factual | 0.20 | 0.250 | ❌ Incomplete | Embedding model failure — Week 2 |
| q03 | factual | 0.10 | 0.100 | ⚠️ Terse | Correct but minimal |
| q04 | multi-section | 0.30 | 0.188 | ✅ Correct | MS-MARCO + Jeopardy accurate |
| q05 | multi-section | 0.40 | 0.200 | ⚠️ Partial | Double-counted 110M params |
| q06 | multi-section | 0.50 | 0.313 | ✅ Correct | All hot-swap numbers present |
| q07 | adversarial | 0.00 | 0.000 | ❌ Contaminated | Math garbage before refusal |
| q08 | adversarial | 0.00 | 0.000 | ✅ Refused | Clean refusal |
| q09 | failure-mode | 0.20 | 0.286 | ✅ Verbatim correct | Boundary split FIXED |
| q10 | failure-mode | 0.30 | 0.429 | ✅ Verbatim correct | Abstract intact |

### Known Code Issues (Not Yet Fixed)

1. `schemas.py` line 1: `ffrom __future__` — syntax error, will crash on import
2. `settings.py` `CHUNK_SIZE=800` is ignored by `chunker.py` which hardcodes 600
3. `chunker.py` uploaded version is stale (still shows old RecursiveCharacterTextSplitter)
4. `run_eval.py` adversarial metric measures retriever, not LLM behavior
5. Latency regressed: q04=16.5s, q06=13.8s (was ~3.7s in v2 baseline)

### What "Working" Actually Means Now

- ✅ Ingestion pipeline functional
- ✅ Eval harness produces trustworthy numbers
- ✅ Sentence-aware chunking fixes boundary splits (validated on q09)
- ✅ LLM refuses adversarial questions correctly (q08)
- ⚠️ Embedding model too weak for fine-grained factual retrieval (q01, q02)
- ⚠️ Prompt insufficiently hardened against garbage leakage (q07)
- ⚠️ Latency unacceptable for production (>10s on multi-section questions)

### Week 1 Completion Status

| Requirement | Status | Notes |
|---|---|---|
| Naive RAG baseline | ✅ Done | Working end-to-end |
| Eval harness | ✅ Done | 10 questions, chunk metrics, trustworthy |
| Sentence-aware chunking | ✅ Done | NLTK implementation, validated |
| Section hierarchy metadata | ❌ Not started | Would improve q01/q02 recall |
| Tables as JSON | ❌ Not started | PyMuPDF dumps tables as plain text |
| Figure descriptions | ❌ Not started | No image detection in pipeline |
| 80-question RAGAS baseline | ❌ Not started | Curriculum requires 80, we have 10 |
| Eval dashboard | ❌ Not started | JSON files only, no UI |

### Decision: Week 1 vs Week 2

**Current position:** Day 5 of Week 1 (not Day 6, not Week 2).

**Path A (recommended):** Complete Week 1 infrastructure before touching Week 2.
- Next: Section hierarchy metadata (30 min implementation, improves factual retrieval)
- Then: Harden prompt (10 min, fixes q07 garbage leakage)
- Then: Fix eval metric bug (5 min, correct adversarial scoring)
- Then: Decide on 80-question dataset or move to Week 2

**Path B (not recommended):** Skip to Week 2 hybrid search because "embedding model is the bottleneck."
- Risk: Without section metadata, hybrid search still won't know that "BART" lives in Section 2.3
- Risk: Without hardened prompt, better retrieval just means more garbage reaches the LLM

### Stack (Updated)

Python 3.12, uv, FastAPI, LangChain, ChromaDB, PyMuPDF, Groq, Loguru, **NLTK** (new)

### Git Hygiene Note

The `chunker.py` in this repo may be stale. Verify the running server has the NLTK sentence-aware implementation before claiming Week 1 is complete. The definitive test: query the debug endpoint for the three-limitations sentence and confirm it is not split.