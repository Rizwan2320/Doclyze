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
