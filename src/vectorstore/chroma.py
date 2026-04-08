from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger
from typing import List

from src.config.settings import settings
from src.embeddings.model import embeddings


class VectorStore:

    def __init__(self):
        self.persist_dir = settings.CHROMA_PERSIST_DIR
        self.embeddings = embeddings
        self._stores: dict = {}  # cache open collections

    def _get_vectorstore(self, collection_name: str = "default") -> Chroma:
        if collection_name not in self._stores:
            logger.info(f"Initializing Chroma collection: {collection_name}")
            self._stores[collection_name] = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )
            logger.info(f"✅ Chroma collection '{collection_name}' ready")
        return self._stores[collection_name]

    def add_documents(self, documents: List[Document], collection_name: str = "default"):
        store = self._get_vectorstore(collection_name)
        logger.info(f"Adding {len(documents)} documents to collection '{collection_name}'")
        store.add_documents(documents)
        logger.info(f"✅ Successfully added {len(documents)} vectors")

    def similarity_search(self, query: str, k: int = 10, collection_name: str = "default"):
        store = self._get_vectorstore(collection_name)
        return store.similarity_search(query, k=k)


vector_store = VectorStore()

