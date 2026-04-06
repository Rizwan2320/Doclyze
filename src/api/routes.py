from pathlib import Path
from uuid import uuid4
import tempfile

from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger

from src.api.schemas import UploadAcceptedResponse, QueryRequest, QueryResponse, compute_sha256, SupportedFileType, IngestionStats, IngestionStatus
from src.ingestion.service import ingestion_service
from src.rag.chain import rag_chain

router = APIRouter(prefix="/api", tags=["documents"])


@router.post("/upload", response_model=UploadAcceptedResponse)
async def upload_document(file: UploadFile = File(...)):
    # Validate extension
    file_ext = Path(file.filename).suffix.lower()
    allowed = {e.value for e in SupportedFileType}
    if file_ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed}")

    # Read bytes
    content = await file.read()

    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 50MB.")

    document_id = uuid4()
    sha256 = compute_sha256(content)

    logger.info(f"Upload received: {file.filename} | {len(content)} bytes | id={document_id}")

    # Save to temp file and ingest
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        chunks = ingestion_service.ingest_file(tmp_path)

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    finally:
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

    logger.info(f"Query received: '{request.query}'")

    try:
        answer = rag_chain.query(request.query, k=request.top_k)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return QueryResponse(answer=answer, sources=[])


@router.post("/summarize")
async def summarize_document():
    try:
        summary = rag_chain.summarize_document()
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"summary": summary}