from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger
from pathlib import Path
from uuid import uuid4

from src.api.schemas import (
    UploadAcceptedResponse,
    IngestionStatus,
    SupportedFileType,
    compute_sha256,
)
from src.ingestion.service import ingestion_service

router = APIRouter(prefix="/api", tags=["documents"])


@router.post("/upload", response_model=UploadAcceptedResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document and process it synchronously.
    
    This is the simplest and most reliable version for development.
    We will switch to async + background tasks later when the system is more stable.
    """
    # ── Basic validation ─────────────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in [ext.value for ext in SupportedFileType]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed types: {[e.value for e in SupportedFileType]}"
        )

    # Read file content
    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(content) > 50 * 1024 * 1024:  # 50 MB limit
        raise HTTPException(status_code=400, detail="File too large. Maximum allowed size is 50MB.")

    # Generate document_id and hash for integrity + deduplication
    document_id = uuid4()
    content_sha256 = compute_sha256(content)

    logger.info(f"Upload received → {file.filename} | size={len(content)} bytes | document_id={document_id}")

    # ── Process ingestion synchronously ─────────────────────────────────────
    try:
        chunks = ingestion_service.ingest_from_bytes(
            content=content,
            filename=file.filename,
            document_id=document_id,
        )

        logger.info(f"Successfully ingested {file.filename} → {len(chunks)} chunks created")

        return UploadAcceptedResponse(
            document_id=document_id,
            filename=file.filename,
            content_sha256=content_sha256,
            status=IngestionStatus.SUCCESS,
            status_url=f"/api/status/{document_id}",
            message=f"Document uploaded and ingested successfully. Created {len(chunks)} chunks.",
        )

    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )


# Placeholder endpoints (we'll implement these later)
@router.get("/status/{document_id}")
async def get_ingestion_status(document_id: uuid4):
    """Future endpoint to check ingestion status (polling)."""
    raise HTTPException(status_code=501, detail="Status polling not implemented yet")


@router.post("/query")
async def query_documents(request):
    """Future RAG query endpoint."""
    raise HTTPException(status_code=501, detail="Query endpoint not implemented yet")