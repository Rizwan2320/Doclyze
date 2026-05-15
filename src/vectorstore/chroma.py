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

    def similarity_search(
        self, query: str, k: int = 10, collection_name: str = "default"
    ):
        store = self._get_vectorstore(collection_name)
        
        try:
            results = store.similarity_search_with_relevance_scores(
                query,
                k=k,
                filter={"chunk_type": {"$eq": "child"}}
            )
        except Exception as e:
            logger.error(f"Similarity search failed for '{query[:50]}...': {e}")
            return []

        docs = []
        for i, item in enumerate(results):
            try:
                # Handle both tuple and list formats defensively
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    doc, score = item[0], item[1]
                else:
                    logger.warning(f"Unexpected result format at index {i}: {type(item)}")
                    continue
                
                if doc is None:
                    logger.warning(f"Null document at index {i}")
                    continue
                    
                doc.metadata["relevance_score"] = round(float(score), 4)
                docs.append(doc)
                
            except Exception as e:
                logger.warning(f"Failed to process result at index {i}: {e}")
                continue
            
        logger.info(f"Retrieved {len(docs)} documents for query")
        return docs


# Global instance
vector_store = VectorStore()