# FAILURE_MODES.md

## FM-001 — Abstract/Introduction retrieval miss

**Date:** 2025-04-16  
**Query:** "What is the abstract of this paper?"  
**Expected:** Chunks from pages 1-3 (abstract, introduction)  
**Actual:** 10 chunks from references and appendix (pages 6-18)  
**Root cause hypothesis:** Dense retrieval (all-MiniLM-L6-v2) ranks
reference-section chunks above abstract chunks for this query type.
Possibly because "abstract" as a word appears in unrelated contexts
(e.g. "abstract reasoning", section headers in other parts).  
**Impact:** RAG correctly refuses to answer but user gets no value.  
**Mitigation (not yet built):**

- Metadata filtering by page range / section_type
- Section hierarchy metadata (chunk knows it's from "Abstract" section)
- Better chunking that preserves section boundaries
  **Status:** Open — will be validated by eval harness

## FM-001 — PyMuPDF extraction failure on 2-column academic PDF

**Confirmed:** Math notation extracted as garbled text (LaTeX symbols
appearing raw in answers). Multi-column layout causes reading order issues.
**Evidence:** "sim(q, z) T where q is the query" appearing in answer output.
**Planned fix:** Compare Docling extraction on same document in Week 1 Day 1.



## FM-002 — Sentence boundary split in child chunks

**Date:** 2026-05-14  
**Query:** "What three limitations does the paper identify in its introduction?" (q09)  
**Expected:** Full sentence: "They cannot easily expand or revise their memory, can't straightforwardly provide insight into their predictions, and may produce 'hallucinations'."  
**Actual (before fix):** Chunk ended at "insight into" — LLM hallucinated "their knowledge" as third limitation  
**Root cause:** `RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", ". ", " ", ""]` failed on academic paper text where periods are followed by `\n` not ` `. Fell back to character-level truncation, cutting mid-word ("knowledge f") and mid-sentence.  
**Fix applied:** NLTK `sent_tokenize` pre-split + sentence-aware grouping into ~1000-char chunks with ~200-char sentence-based overlap. Guarantees no sentence is ever split.  
**Validation:** q09 chunk_recall improved from 0.25 → 0.286 (still low due to page-fallback expected sets, but answer is now verbatim correct).  
**Status:** Resolved — sentence-aware chunking is now default.

## FM-003 — Parametric memory override with parent chunks

**Date:** 2026-05-14  
**Query:** q03 "How many documents does Wikipedia get split into?"  
**Expected:** "21 million documents. Each Wikipedia article is split into disjoint 100-word chunks."  
**Actual (with parent-only retrieval):** LLM generated fake sections "4.1 Open-Domain Question Answering... 5 Conclusion" from training memory, ignoring retrieved context  
**Root cause:** Parent chunks (2000 chars) provided coherent enough context that Llama-3.3-70b recognized the RAG paper from training data and switched from "read context" to "I know this paper" mode.  
**Lesson:** More context ≠ better grounding. Coherent large excerpts trigger parametric memory. Child chunks (fragmented, 600 chars) accidentally prevented this because they were too incoherent to trigger recognition.  
**Status:** Documented — child-only retrieval maintained as default. Hybrid child→parent architecture deferred to Week 3.

## FM-004 — Adversarial metric measures retriever, not system

**Date:** 2026-05-14  
**Query:** q07, q08 (questions with no answer in document)  
**Expected:** `correct_refusal = true`  
**Actual:** `correct_refusal = false` because retriever returned 10 chunks despite no relevant chunks existing  
**Root cause:** `run_eval.py` checks `len(retrieved_ids) == 0` instead of checking whether the LLM actually refused. The LLM correctly says "I cannot answer..." but the metric blames the retriever.  
**Impact:** False failure signals. System behavior is correct; measurement is wrong.  
**Fix pending:** Change metric to inspect `actual_answer` for refusal language.  
**Status:** Open — fix in `run_eval.py` before next eval run.

## FM-005 — Embedding model misses fine-grained factual alignment

**Date:** 2026-05-14  
**Query:** q01 "What two model variants does RAG introduce?"  
**Expected:** Chunk from Section 2.1 containing "RAG-Sequence" and "RAG-Token" definitions  
**Actual:** chunk_precision=0.00, chunk_recall=0.00. Retriever returned zero expected chunks.  
**Root cause:** `all-MiniLM-L6-v2` is a general-purpose sentence encoder optimized for broad semantic similarity ("sports" ≈ "athletics"). It fails at fine-grained factual discrimination ("RAG-Sequence model" vs "RAG-Token model" vs mathematical notation from unrelated sections).  
**Justification for Week 2:** This is a measured, reproducible embedding model failure. Hybrid search (BM25 + dense) is justified because sparse retrieval matches exact terms like "RAG-Sequence" that dense embeddings miss.  
**Status:** Deferred to Week 2 — will be addressed with hybrid search and/or better embedding model.

## FM-006 — Garbage leakage before adversarial refusal

**Date:** 2026-05-14  
**Query:** q07 "What is RAG's average inference latency in milliseconds?"  
**Expected:** Clean refusal: "I cannot answer this based on the provided document."  
**Actual:** LLM output: "z∈Z sim(x, z) where sim(x, z) is a similarity function... [math garbage]... I cannot answer this based on the provided document."  
**Root cause:** Retrieved chunks contained dense mathematical notation from Methods section. LLM echoed the notation before catching itself and refusing. Prompt insufficiently constrains against echoing raw context fragments.  
**Fix pending:** Add explicit prompt rule: "NEVER output raw mathematical notation, LaTeX, or code fragments from the context unless directly asked."  
**Status:** Open — prompt engineering fix, 10 minutes.

## FM-007 — Double-count hallucination in parameter counting

**Date:** 2026-05-14  
**Query:** q05 "What is the total number of trainable parameters?"  
**Expected:** "626M trainable parameters: 110M from BERT-base query encoder + 406M from BART-large"  
**Actual:** "110M parameters from the BERT-base query and document encoder, and 406M parameters from BART-large, and 110M parameters from the BERT-base document encoder" — double-counted 110M  
**Root cause:** LLM parametric memory confusion. Paper says BERT query+doc = 110M each but doc encoder is frozen (not trained). LLM invented an extra 110M from vague memory of BERT architecture.  
**Severity:** Low — answer is directionally correct but numerically inflated.  
**Status:** Documented — may improve with stronger "use ONLY context" prompt constraints.


## FM-010 — Reference Section Contamination in Vector Store

**Date:** 2026-05-23
**Symptom:** q01, q02 returned chunk_precision=0.00, chunk_recall=0.00 
across multiple baselines despite correct content existing in the paper.

**Root cause:** PyMuPDF extracted all 19 pages including reference pages 
10–16. These bibliography chunks contain domain vocabulary identical to 
content — BART, DPR, RAG, retrieval, seq2seq — so MiniLM scored them as 
semantically similar to queries. Content chunks on pages 3–6 were outranked 
by reference noise.

**Why it wasn't caught earlier:** Chunk P/R only measures whether expected 
IDs appeared in results. It doesn't show WHAT was retrieved instead. No 
observability into retrieval content = silent contamination.

**Fix:** Added _is_reference_chunk() filter in chunker.py. Two rules:
- Rule A: >25% of lines start with [N] pattern
- Rule B: 3+ lines start with [N] AND academic URLs present
Pages 10–16 removed. Pages 1–9 and 17–19 preserved.

**Validated:** 19 pages → 12 pages, 81 child chunks → 59 child chunks.
Pages 5, 6 (results tables) confirmed present after fix.

**Lesson:** Filter noise at ingestion time, not at query time. 
Garbage in the vector store cannot be compensated by better retrieval.
**Status:** ✅ Resolved