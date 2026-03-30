from pathlib import Path
from typing import List
from uuid import UUID

from langchain_core.documents import Document
from loguru import logger

from src.config.settings import settings
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import chunker


class IngestionService:
    """
    Main orchestrator for document ingestion pipeline.
    Responsible for coordinating loading, chunking, and future embedding steps.

    """

    def __init__(self):
        self.loader = DocumentLoader()
        self.chunker = chunker

    def ingest_file(self, file_path: str | Path) -> List[Document]:
        """Synchronous ingestion from local file path (for testing)."""
        logger.info(f"🚀 Starting ingestion for file: {Path(file_path).name}")

        raw_docs = self.loader.load_file(file_path)
        chunks = self.chunker.chunk_documents(raw_docs)

        logger.info(f"✅ Ingestion completed: {len(raw_docs)} elements → {len(chunks)} chunks")
        return chunks

    def ingest_from_bytes(self, content: bytes, filename: str, document_id: UUID | None = None) -> List[Document]:
        """
        Ingestion from uploaded bytes (used by FastAPI).
        This will be called from background task.
        """
        logger.info(f"🚀 Starting background ingestion for: {filename} | size={len(content)} bytes")

        try:
            # Load from bytes
            raw_docs = self.loader.load_from_bytes(content, filename)

            # Chunk
            chunks = self.chunker.chunk_documents(raw_docs)

            logger.info(f"✅ Background ingestion completed: {len(raw_docs)} elements → {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"❌ Ingestion failed for {filename}: {e}")
            raise


# Global singleton
ingestion_service = IngestionService()