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
        """Create hierarchical chunks with proper parent-child relationships."""
        if not documents:
            return []

        # Filter out noise — elements shorter than 50 chars are often headers/UI junk
        documents = [
            doc for doc in documents
            if len(doc.page_content.strip()) >= 50
        ]
        logger.info(f"After filtering: {len(documents)} meaningful elements")

        logger.info(f"Creating hierarchical chunks from {len(documents)} elements")

        # Create parent chunks for summarization/context
        parent_chunks = self.parent_splitter.split_documents(documents)

        # Add unique ID and chunk_type to each parent chunk
        for parent in parent_chunks:
            if "id" not in parent.metadata:
                parent.metadata["id"] = str(uuid4())
            parent.metadata["chunk_type"] = "parent"

        # Create small chunks for retrieval, linked to their parent
        small_chunks: List[Document] = []

        for parent in parent_chunks:
            # Split this parent into smaller child chunks
            children_texts = self.small_splitter.split_text(parent.page_content)

            for child_text in children_texts:
                child_doc = Document(
                    page_content=child_text,
                    metadata={
                        **parent.metadata,           # inherit parent metadata
                        "parent_id": parent.metadata["id"],
                        "chunk_type": "child",       # explicitly set as child for retrieval
                    }
                )
                small_chunks.append(child_doc)

        all_chunks = small_chunks + parent_chunks

        logger.info(
            f"✅ Created {len(small_chunks)} small chunks + "
            f"{len(parent_chunks)} parent chunks"
        )

        return all_chunks


# Singleton
chunker = DocumentChunker()