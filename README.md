---
title: Doclyze
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Doclyze — AI Document Analysis Portal

Upload any PDF and ask questions about it. Powered by RAG (Retrieval-Augmented Generation).

## Demo

> Upload a PDF → Get a document_id → Query it

```bash
POST /api/upload  → returns document_id
POST /api/query   → { "query": "...", "document_id": "..." }
POST /api/summarize
```

## Evaluation

Tested on the RAG paper (Lewis et al., 2020) — 5-question benchmark:
**Score: 4/5 (80%)**

## Architecture

PDF → PyMuPDF → Chunker → ChromaDB
Query → Embedding Search (k=15) → Groq llama-3.3-70b → Answe

## Stack

- **Backend:** FastAPI, Python 3.12
- **RAG:** LangChain, ChromaDB, sentence-transformers
- **LLM:** Groq (llama-3.3-70b-versatile)
- **PDF Parsing:** PyMuPDF

## Run Locally

```bash
git clone https://github.com/Rizwan2320/Doclyze
cd Doclyze
uv sync
cp .env.example .env  # add your GROQ_API_KEY
uv run uvicorn src.main:app --reload
```

Open http://localhost:8000/docs

## What I Learned

- Diagnosing RAG failures by inspecting actual chunk content
- Document isolation using per-document ChromaDB collections
- PyMuPDF vs Unstructured for different PDF types
- Production FastAPI patterns with Pydantic schemas
