from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger

from src.config.settings import settings


class DocumentChunker:
    """
    Production-grade hybrid document chunker.
    
    Engineering Decisions:
    - Uses RecursiveCharacterTextSplitter as primary (structure-aware)
    - Configurable chunk_size and chunk_overlap from settings
    - Preserves all metadata from loader
    - Keeps separators logical (paragraphs → sentences → words)
    - Ready for future semantic chunking enhancement
    """

    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE          # e.g. 800
        self.chunk_overlap = settings.CHUNK_OVERLAP    # e.g. 150

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],   # Logical hierarchy
            is_separator_regex=False,
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks while preserving metadata."""
        logger.info(f"Chunking {len(documents)} documents | Chunk size={self.chunk_size}, Overlap={self.chunk_overlap}")

        chunked_docs = self.text_splitter.split_documents(documents)

        # Re-attach and enrich metadata
        for i, doc in enumerate(chunked_docs):
            doc.metadata.update({
                "chunk_id": i,
                "total_chunks": len(chunked_docs),
                "chunk_size": len(doc.page_content),
            })

        logger.info(f"✅ Created {len(chunked_docs)} chunks from {len(documents)} original elements")
        return chunked_docs


# Singleton for easy use
chunker = DocumentChunker()