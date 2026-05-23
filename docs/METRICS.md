# Doclyze Metrics & Baselines

# Eval Baseline — Week 1 Naive RAG

**Date:** 2026-04-18  
**Model:** llama-3.3-70b-versatile (Groq)  
**Embeddings:** all-MiniLM-L6-v2  
**Vector DB:** ChromaDB  
**Chunking:** Hierarchical (small=600, parent=2000)  
**k:** 10

## Page Recall by Category

| Category      | Avg Page Recall | Notes                                               |
| ------------- | --------------- | --------------------------------------------------- |
| factual       | 1.00            | Right pages retrieved                               |
| multi-section | 0.83            | q04 missed page 8                                   |
| adversarial   | N/A             | Retrieving chunks despite no answer existing        |
| failure-mode  | 1.00            | Pages retrieved but chunking may still lose content |

## Latency

| Min    | Max     | Avg     |
| ------ | ------- | ------- |
| 2530ms | 17759ms | ~8285ms |

## Known Issues

- Adversarial questions retrieve with scores 0.32-0.42 — likely hallucinating
- Page recall is a weak metric — right page ≠ right chunk at top position
- Latency variance is high — no caching
- Hierarchical chunking exists in ingestion but retriever ignores parent/child relationship

## What This Baseline Does NOT Measure Yet

- Answer quality (is the actual answer correct?)
- Faithfulness (is the answer grounded in retrieved chunks?)
- RAGAS metrics (context precision, context recall, answer relevancy)

## v1 vs v2 Comparison (2026-04-20)

**Changes made:**

- Prompt: system/human split, explicit no-chunk-tags rule
- Retriever: child chunks only (chunk_type filter)

**Results:**

- Latency: avg 8285ms → 3736ms (-55%) ✅
- Adversarial refusals: cleaner wording ✅
- q01: still broken (extraction failure, not prompt failure) ❌
- q09: third limitation hallucinated (chunk boundary issue) ❌
- q04: MS-MARCO miss unchanged (retrieval ranking issue) ❌

**Conclusion:** Prompt and deduplication fixes helped latency and
adversarial behavior. Core failures are extraction and retrieval
quality — not addressable by prompt engineering alone.



## Baseline: Week 1 Naive RAG (Post-Sentence-Aware Chunking)

**Date:** 2026-05-14  
**Commit:** [PENDING — commit NLTK chunker and tag]  
**Model:** llama-3.3-70b-versatile (Groq)  
**Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (384d)  
**Vector DB:** ChromaDB persistent  
**Chunking:** NLTK sentence-aware child (≈1000 chars, ≈200 overlap) + parent (2000 chars)  
**Retrieval:** Child-only, k=10, cosine similarity  
**Prompt:** Basic system/human with no-chunk-tags rule  

### Per-Question Results

| ID | Category | Chunk P | Chunk R | Correct? | Latency | Failure Mode |
|---|---|---|---|---|---|---|
| q01 | factual | 0.00 | 0.000 | ❌ | 725ms | FM-005: Embedding model miss |
| q02 | factual | 0.20 | 0.250 | ❌ | 418ms | FM-005: Embedding model miss |
| q03 | factual | 0.10 | 0.100 | ⚠️ | 516ms | Terse but correct |
| q04 | multi-section | 0.30 | 0.188 | ✅ | 16,547ms | — |
| q05 | multi-section | 0.40 | 0.200 | ⚠️ | 11,616ms | FM-007: Double-count hallucination |
| q06 | multi-section | 0.50 | 0.313 | ✅ | 13,790ms | — |
| q07 | adversarial | 0.00 | 0.000 | ❌ | 7,408ms | FM-006: Garbage before refusal |
| q08 | adversarial | 0.00 | 0.000 | ✅ | 4,355ms | Clean refusal |
| q09 | failure-mode | 0.20 | 0.286 | ✅ | 11,468ms | FM-002: FIXED — was boundary split |
| q10 | failure-mode | 0.30 | 0.429 | ✅ | 10,374ms | — |

### Category Averages

| Category | Avg Chunk P | Avg Chunk R | Correct Rate | Avg Latency |
|---|---|---|---|---|
| factual | 0.10 | 0.117 | 0/3 (0%) | 553ms |
| multi-section | 0.40 | 0.234 | 2/3 (67%) | 13,984ms |
| adversarial | 0.00 | 0.000 | 1/2 (50%)* | 5,882ms |
| failure-mode | 0.25 | 0.358 | 2/2 (100%) | 10,921ms |

*Metric bug: q07 marked incorrect due to FM-004. LLM actually refused but dumped garbage first.

### Key Signals

- **Boundary splits:** FIXED (q09, q10 both correct)
- **Embedding precision:** FAILED (q01, q02 zero recall) → justifies Week 2 hybrid search
- **Adversarial behavior:** MIXED (q08 clean, q07 contaminated) → prompt fix needed
- **Latency:** UNACCEPTABLE for multi-section (&gt;10s) → too much context sent to Groq

### What This Baseline Justifies

| Failure | Justified Next Step |
|---|---|
| q01, q02 chunk_recall = 0 | Week 2: Hybrid search (BM25 + dense) or better embeddings |
| q07 garbage leakage | Week 1: Harden prompt (10 min fix) |
| q05 double-count | Week 1: Harden prompt (may improve) |
| q04, q06 latency &gt;10s | Week 3: CRAG/self-reflection to reduce context volume |

### Measurement Quality

- ✅ Gold dataset aligned with DB (via `fix_gold_dataset.py`)
- ⚠️ Page-fallback inflates expected sets (q02=10 expected, q03=12 expected)
- ❌ Adversarial metric bug (FM-004) — fix before next run
- ❌ No RAGAS metrics yet (faithfulness, answer relevancy)

### Next Baseline

After prompt hardening + metric fix, re-run and record as Baseline 1.1. After Week 2 hybrid search, record as Baseline 2.0.




## Baseline 1.3-clean
**Date:** 2026-05-23
**Commit:** [tag this]
**Model:** llama-3.3-70b-versatile (Groq)
**Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (384d)
**Vector DB:** ChromaDB — clean collection, reference pages 10-16 removed
**Chunking:** NLTK sentence-aware child (~1000 chars) + parent (2000 chars)
**Retrieval:** Child-only, k=10, cosine similarity
**Gold dataset:** 10 questions, manually curated expected chunk IDs (manual_patch_v2)

### Per-Question Results
| ID  | Category      | Chunk P | Chunk R | hit@1 | pos  | Latency  |
|-----|---------------|---------|---------|-------|------|----------|
| q01 | factual       | 0.00    | 0.00    | 0     | None | 433ms    |
| q02 | factual       | 0.00    | 0.00    | 0     | None | 326ms    |
| q03 | factual       | 0.10    | 1.00    | 0     | 3    | 257ms    |
| q04 | multi-section | 0.20    | 0.67    | 1     | 1    | 1528ms   |
| q05 | multi-section | 0.20    | 1.00    | 1     | 1    | 6507ms   |
| q06 | multi-section | 0.10    | 0.50    | 0     | 2    | 14673ms  |
| q07 | adversarial   | —       | —       | —     | —    | 12397ms  |
| q08 | adversarial   | —       | —       | —     | —    | 12518ms  |
| q09 | failure-mode  | 0.00    | 0.00    | 0     | None | 12296ms  |
| q10 | failure-mode  | 0.10    | 1.00    | 1     | 1    | 12407ms  |

### Key Signals
- q01, q02, q09: zero recall — MiniLM embedding ceiling confirmed
- q03, q05, q10: perfect recall — retrieval works when semantics match
- q04, q06: partial recall — multi-section queries partially working
- q07, q08: clean refusal — adversarial behavior correct
- q01: retrieval failed but answer likely correct via parametric memory (FM-003 risk)
- q02: math leakage (FM-006) — wrong chunks → LLM echoed notation
- q09: false refusal (FM-008) — retrieval failed → prompt refused

### What This Baseline Justifies
- Embedding bakeoff (bge-small-en-v1.5) is the justified next step
- q01, q02, q09 zero recall cannot be fixed by prompt or chunking changes
- RAGAS needed to catch FM-003 (parametric memory override) on q01

### Note on Previous Baselines
Baselines 1.0 and 1.2 are NOT comparable to this baseline.
They were measured against a polluted collection (reference chunks pages 10-16 included)
and inflated gold dataset (page-fallback expected sets).
This is the first baseline with a trustworthy ruler.