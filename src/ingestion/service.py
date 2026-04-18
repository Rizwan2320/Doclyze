from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from loguru import logger

from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import chunker
from src.vectorstore.chroma import vector_store


class IngestionService:

    def __init__(self):
        self.loader = DocumentLoader()
        self.chunker = chunker

    def ingest_file(
        self, file_path: str | Path, collection_name: str = "default",  original_filename: Optional[str] = None
    ) -> List[Document]:
        """Ingest a file and store its chunks in the vector store."""
        logger.info(f"Starting ingestion: {Path(file_path).name}")

        raw_documents = self.loader.load_file(file_path, original_filename=original_filename)
        chunks = self.chunker.chunk_documents(raw_documents)

        vector_store.add_documents(chunks, collection_name=collection_name)

        logger.info(f"Saved {len(chunks)} chunks to collection '{collection_name}'")
        return chunks


# Singleton / global instance
ingestion_service = IngestionService()