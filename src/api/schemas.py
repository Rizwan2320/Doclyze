from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────
class IngestionStatus(str, Enum):
    """Lifecycle of a document ingestion job."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class SupportedFileType(str, Enum):
    """Explicit allow-list of supported file extensions."""
    PDF = ".pdf"
    DOCX = ".docx"
    TXT = ".txt"
    MD = ".md"


# ── Sub-models ────────────────────────────────────────────────────────────────
class SourceChunk(BaseModel):
    """
    A single retrieved chunk returned with RAG answers.
    Refactored for defensive engineering to prevent API crashes.
    """
    document_id: Optional[Union[UUID, str]] = None
    chunk_id: Optional[str] = None
    chunk_index: Optional[int] = Field(default=0, ge=0)
    text: str = Field(default="[Content missing]", min_length=1)
    
    # RELAXED: Score is now optional. We remove le=1.0 because 
    # some distance metrics (like L2) can exceed 1.0.
    score: Optional[float] = Field(default=0.0)
    
    # SAFE: Page numbers often fail on poorly formatted PDFs.
    page_number: Optional[int] = Field(default=None, ge=1)
    
    source_file: Optional[str] = Field(default="Unknown Source")


class IngestionStats(BaseModel):
    """Metrics from a completed ingestion run."""
    num_parsed_elements: int = Field(..., ge=0)
    num_chunks: int = Field(..., ge=0)
    num_tokens_approx: Optional[int] = Field(default=None, ge=0)
    duration_ms: Optional[float] = Field(default=None, ge=0.0)


# ── Request Models ────────────────────────────────────────────────────────────
class UploadRequest(BaseModel):
    """Metadata for file upload (used internally)."""
    filename: str
    content_length: int = Field(..., gt=0, le=50_000_000)  # 50MB max


# ── Response Models ───────────────────────────────────────────────────────────
class UploadAcceptedResponse(BaseModel):
    """Immediate response when a file is accepted for background processing."""
    document_id: UUID = Field(..., description="Use this to poll status")
    filename: str
    content_sha256: str
    status: IngestionStatus = IngestionStatus.PENDING
    status_url: str = Field(..., description="GET /api/status/{document_id}")
    accepted_at: datetime = Field(default_factory=datetime.utcnow)
    message: str = "Document accepted for ingestion."


class IngestionStatusResponse(BaseModel):
    """Response when polling ingestion status."""
    document_id: UUID
    filename: str
    status: IngestionStatus
    stats: Optional[IngestionStats] = None
    error_detail: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    @model_validator(mode="after")
    def validate_status_invariants(self) -> "IngestionStatusResponse":
        if self.status == IngestionStatus.FAILED and not self.error_detail:
            raise ValueError("error_detail is required when status is FAILED")
        if self.status == IngestionStatus.SUCCESS and not self.stats:
            raise ValueError("stats are required when status is SUCCESS")
        return self


class QueryRequest(BaseModel):
    """RAG query request."""
    query: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)
    document_id: Optional[UUID] = None

    @field_validator("query")
    @classmethod
    def strip_and_validate_query(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("query cannot be empty or whitespace only")
        return cleaned


class QueryResponse(BaseModel):
    """Final RAG response."""
    answer: str
    sources: List[SourceChunk] = Field(default_factory=list)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    latency_ms: Optional[float] = None


# ── Utility ───────────────────────────────────────────────────────────────────
def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hash for file deduplication and integrity."""
    return hashlib.sha256(data).hexdigest()