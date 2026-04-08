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
    file_ext = Path(file.filename).suffix.lower()
    allowed = {e.value for e in SupportedFileType}
    if file_ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed}")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 50MB.")

    document_id = uuid4()
    collection_name = str(document_id)  # each doc gets its own collection
    sha256 = compute_sha256(content)

    logger.info(f"Upload: {file.filename} | id={document_id}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        chunks = ingestion_service.ingest_file(tmp_path, collection_name=collection_name)

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
    if not request.document_id:
        raise HTTPException(status_code=400, detail="document_id is required.")

    collection_name = str(request.document_id)
    logger.info(f"Query: '{request.query}' | collection={collection_name}")

    try:
        answer = rag_chain.query(
            request.query,
            k=request.top_k,
            collection_name=collection_name
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(answer=answer, sources=[])


@router.post("/summarize")
async def summarize_document(document_id: str):
    try:
        summary = rag_chain.summarize_document(collection_name=document_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"summary": summary}