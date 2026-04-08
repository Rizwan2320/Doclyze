from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from src.config.settings import settings


class EmbeddingModel:
    """
    Production-ready embedding service.
    
    Engineering Decisions:
    - Uses local HuggingFace model (no API cost during development)
    - Lazy initialization (model loads only when first used)
    - Batching support for efficiency
    - Easy to swap model later (just change settings.EMBEDDING_MODEL_NAME)
    """

    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.dimension = settings.EMBEDDING_DIMENSION
        self._embeddings = None   # Lazy load

    @property
    def embeddings(self):
        """Lazy initialization of the embedding model."""
        if self._embeddings is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},   # Change to "cuda" if you have GPU
            )
            logger.info(f"✅ Embedding model loaded successfully (dim={self.dimension})")
        return self._embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts (chunks)."""
        logger.info(f"Embedding {len(texts)} documents...")
        embeddings = self.embeddings.embed_documents(texts)
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        return self.embeddings.embed_query(text)


# Global singleton
embedding_model = EmbeddingModel()

embeddings = embedding_model.embeddings