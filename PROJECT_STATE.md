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
