# ADR-001: Sentence-Aware Chunking with NLTK

**Status:** Accepted  
**Date:** 2026-05-14  
**Deciders:** Rizwan (implementer), Mentor (reviewer)  
**Consulted:** Curriculum Week 1 Day 5 requirements

## Context

The naive RAG baseline used `RecursiveCharacterTextSplitter` with `separators=["\n\n", "\n", ". ", " ", ""]` for child chunks. Eval question q09 ("What three limitations...") consistently hallucinated the third limitation because the sentence was split across chunks:

- Chunk A ended: "...can't straightforwardly provide insight into"
- Chunk B started: "their predictions, and may produce hallucinations"

The LLM, given only Chunk A, invented "their knowledge" as the third limitation.

## Decision

Replace `RecursiveCharacterTextSplitter` for child chunks with a two-step process:
1. **Pre-split:** NLTK `sent_tokenize()` splits text into whole sentences
2. **Group:** Custom `_group_sentences_into_chunks()` packs sentences into ~1000-char chunks with ~200-char sentence-based overlap

Parent chunks continue using `RecursiveCharacterTextSplitter` (2000 chars) because they are larger and less sensitive to boundary issues.

## Consequences

### Positive
- Guaranteed no mid-sentence splits
- q09 now returns verbatim correct answer
- Overlap preserves sentence boundaries (not character boundaries)

### Negative
- Adds NLTK dependency (~15MB download for punkt tokenizer)
- Chunk sizes are approximate, not exact (varies by sentence length)
- Slightly more complex chunker logic

### Neutral
- Parent chunking unchanged
- Embedding model unchanged
- Retrieval filter unchanged (still child-only)

## Validation

| Metric | Before | After | Delta |
|---|---|---|---|
| q09 chunk_recall | 0.25 | 0.286 | +0.036 |
| q09 answer correct | ❌ Hallucinated | ✅ Verbatim | Fixed |
| q10 answer correct | ✅ Correct | ✅ Correct | Unchanged |

Note: chunk_recall is still low because expected sets are inflated by page-fallback in gold dataset generation. The important signal is answer correctness, not chunk ID overlap.

## Alternatives Considered

| Alternative | Rejected Because |
|---|---|
| Larger chunks (1000/200) without sentence awareness | Still cut mid-sentence if no period matched separator |
| Parent-only retrieval | Triggered parametric memory override (q03 hallucinated fake sections) |
| Child→parent hybrid | `store.get(ids=...)` crashed in LangChain-Chroma version |
| spaCy instead of NLTK | Heavier dependency; NLTK sufficient for this use case |

## Links

- `FAILURE_MODES.md` FM-002
- `TRADEOFFS.md` Section 9 (Chunking)
- Curriculum Week 1 Day 5