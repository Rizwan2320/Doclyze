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
