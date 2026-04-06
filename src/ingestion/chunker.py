from typing import List, Dict
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger

from src.config.settings import settings


class DocumentChunker:
    """
    Hierarchical chunker for production RAG.
    Creates both small chunks (for retrieval) and larger parent chunks (for summarization).
    """

    def __init__(self):
        self.small_chunk_size = 600
        self.small_overlap = 100
        self.parent_chunk_size = 2000
        self.parent_overlap = 200

        self.small_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.small_chunk_size,
            chunk_overlap=self.small_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Create hierarchical chunks."""
        if not documents:
            return []

        # Filter out noise — elements shorter than 50 chars are headers/UI junk
        documents = [
            doc for doc in documents
            if len(doc.page_content.strip()) >= 50
        ]
        logger.info(f"After filtering: {len(documents)} meaningful elements")

        logger.info(f"Creating hierarchical chunks from {len(documents)} elements")

        # Create small chunks for retrieval
        small_chunks = self.small_splitter.split_documents(documents)

        # Create parent chunks for summarization
        parent_chunks = self.parent_splitter.split_documents(documents)

        # Add parent_id to small chunks for hierarchical retrieval later
        parent_map: Dict[str, Document] = {str(uuid4()): p for p in parent_chunks}

        for small in small_chunks:
            small.metadata["parent_id"] = list(parent_map.keys())[0]

        all_chunks = small_chunks + parent_chunks

        logger.info(f"✅ Created {len(small_chunks)} small chunks + {len(parent_chunks)} parent chunks")
        return all_chunks


# Singleton
chunker = DocumentChunker()