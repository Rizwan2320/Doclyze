from pathlib import Path
from typing import List

from langchain_core.documents import Document
from loguru import logger

from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import chunker
from src.vectorstore.chroma import vector_store


class IngestionService:

    def __init__(self):
        self.loader = DocumentLoader()
        self.chunker = chunker

    def ingest_file(self, file_path: str | Path) -> List[Document]:
        """Load → chunk → save to vector store."""
        logger.info(f"Starting ingestion: {Path(file_path).name}")

        # Step 1: Load
        raw_documents = self.loader.load_file(file_path)

        # Step 2: Chunk
        chunks = self.chunker.chunk_documents(raw_documents)

        # Step 3: Save to ChromaDB  ← this was missing
        vector_store.add_documents(chunks)
        logger.info(f"Saved {len(chunks)} chunks to ChromaDB")

        return chunks


ingestion_service = IngestionService()