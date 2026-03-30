from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger

from src.config.settings import settings
from src.embeddings.model import embedding_model


class VectorStore:
    """
    Production-grade wrapper around Chroma vector database.
    
    Engineering Decisions:
    - Persistent storage (data survives server restarts)
    - Uses the same embedding model as the rest of the system
    - Collection per document type or single collection with metadata filtering (we'll start simple)
    - Ready for hybrid search and metadata filtering later
    """

    def __init__(self):
        self.persist_directory = Path(settings.CHROMA_PERSIST_DIR)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.vectorstore = None  # Lazy initialization

    def _get_vectorstore(self):
        """Lazy load the Chroma instance."""
        if self.vectorstore is None:
            logger.info(f"Initializing Chroma vector store at: {self.persist_directory}")
            self.vectorstore = Chroma(
                collection_name="doclyze_documents",
                embedding_function=embedding_model.embeddings,
                persist_directory=str(self.persist_directory),
            )
            logger.info("✅ Chroma vector store initialized")
        return self.vectorstore

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents (with embeddings) to the vector store."""
        if not documents:
            logger.warning("No documents to add to vector store")
            return []

        logger.info(f"Adding {len(documents)} documents to vector store...")

        vectorstore = self._get_vectorstore()
        ids = vectorstore.add_documents(documents)

        logger.info(f"✅ Successfully added {len(ids)} vectors to Chroma")
        return ids

    def similarity_search(self, query: str, k: int = 10) -> List[Document]:
        """Search for similar documents."""
        vectorstore = self._get_vectorstore()
        results = vectorstore.similarity_search(query, k=k)
        return results


# Global singleton
vector_store = VectorStore()