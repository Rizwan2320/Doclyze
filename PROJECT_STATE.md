# Doclyze — Project State

## What Is Built & Working

- Full ingestion pipeline: PDF → Unstructured (fast strategy) → filter noise → hierarchical chunker → ChromaDB
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, free)
- RAG chain: similarity search (k=10) → Groq llama-3.3-70b-versatile → answer
- FastAPI server with /api/upload, /api/query, /api/summarize endpoints
- Swagger UI working at localhost:8000/docs
- GitHub repo live: github.com/Rizwan2320/Doclyze

## Key Technical Decisions Made

- Strategy=fast in loader (hi_res took 25 mins, unusable)
- Filter chunks < 50 chars (removes browser-print noise)
- No reranker yet (removed broken ContextualCompressionRetriever)
- Hierarchical chunks: small (600 chars) + parent (2000 chars)
- LangSmith disabled (LANGSMITH_TRACING=false in .env)

## What Remains for Doclyze

1. Test on academic PDF (the RAG paper)
2. Add evaluation — 5 test questions with expected answers
3. Write proper README
4. Deploy to Hugging Face Spaces

## Known Issues

- ingest_file() uses temp file — filename loses original name in metadata
- sources: [] in QueryResponse (not populated yet)
- No deduplication (uploading same file twice doubles chunks)

## Stack

Python 3.12, uv, FastAPI, LangChain, ChromaDB, Groq, Unstructured, Loguru
Windows machine, VS Code
