from pathlib import Path
from uuid import uuid4
import tempfile
import time
from typing import List, Union

from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger

from src.api.schemas import (
    UploadAcceptedResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    compute_sha256,
    SupportedFileType,
    IngestionStatus,
)
from src.ingestion.service import ingestion_service
from src.rag.chain import rag_chain

router = APIRouter(prefix="/api", tags=["documents"])


@router.post("/upload", response_model=UploadAcceptedResponse)
async def upload_document(file: UploadFile = File(...)):
    file_ext = Path(file.filename).suffix.lower()
    allowed = {e.value for e in SupportedFileType}
    if file_ext not in allowed:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {allowed}"
        )

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 50MB.")

    document_id = uuid4()
    collection_name = str(document_id)
    sha256 = compute_sha256(content)

    logger.info(f"Upload: {file.filename} | id={document_id}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Ingest the file and create the vector collection
        chunks = ingestion_service.ingest_file(
            tmp_path, 
            collection_name=collection_name, 
            original_filename=file.filename
        )

    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        # Ensure cleanup even if ingestion fails
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    return UploadAcceptedResponse(
        document_id=document_id,
        filename=file.filename,
        content_sha256=sha256,
        status=IngestionStatus.SUCCESS,
        status_url=f"/api/status/{document_id}",
        message=f"Ingested {len(chunks)} chunks successfully.",
    )


@router.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not request.document_id:
        raise HTTPException(status_code=400, detail="document_id is required.")

    collection_name = str(request.document_id)
    logger.info(f"Query: '{request.query}' | collection={collection_name}")

    try:
        # Measure latency for the RAG operation
        start = time.time()
        answer, docs = rag_chain.query(
            request.query,
            k=request.top_k,
            collection_name=collection_name
        )
        latency_ms = (time.time() - start) * 1000

    except Exception as e:
        logger.error(f"RAG Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Defensive mapping of retrieved documents to SourceChunk schemas
    sources = []
    for i, doc in enumerate(docs):
        # Handle Scores
        score = doc.metadata.get("relevance_score") or getattr(doc, "score", 0.0)
        
        # Handle Page Numbers (support multiple metadata keys)
        page = (
            doc.metadata.get("page_number") or 
            doc.metadata.get("page") or 
            doc.metadata.get("page_label")
        )

        sources.append(
            SourceChunk(
                document_id=request.document_id,
                chunk_id=doc.metadata.get("chunk_id"),
                chunk_index=i,
                text=doc.page_content[:1000],
                score=float(score) if score is not None else 0.0,
                page_number=int(page) if page is not None else None,
                source_file=doc.metadata.get("source", "unknown"),
            )
        )

    return QueryResponse(
        answer=answer, 
        sources=sources, 
        latency_ms=latency_ms
    )


@router.post("/summarize")
async def summarize_document(document_id: str):
    """Generates a high-level summary using larger parent chunks."""
    try:
        summary = rag_chain.summarize_document(collection_name=document_id)
    except Exception as e:
        logger.error(f"Summarization failed for {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"summary": summary}


@router.get("/debug/chunks/{document_id}")
async def debug_chunks(document_id: str, limit: int = 500, offset: int = 0):
    """Return chunks in a collection with pagination for ground-truth dataset building."""
    from src.vectorstore.chroma import vector_store
    
    store = vector_store._get_vectorstore(document_id)
    
    # Get total count first
    all_results = store.get()
    total = len(all_results.get("ids", [])) if all_results else 0
    
    # Paginate
    end = min(offset + limit, total)
    
    chunks = []
    for i in range(offset, end):
        try:
            meta = all_results["metadatas"][i]
            chunks.append({
                "chunk_id": meta.get("chunk_id"),
                "chunk_type": meta.get("chunk_type"),
                "page_number": meta.get("page_number"),
                "source": meta.get("source"),
                "text_preview": all_results["documents"][i][:300] if all_results.get("documents") else "",
            })
        except (IndexError, KeyError, TypeError) as e:
            logger.warning(f"Skipping malformed chunk at index {i}: {e}")
            continue
            
    return {
        "total": total,
        "returned": len(chunks),
        "offset": offset,
        "limit": limit,
        "chunks": chunks
    }