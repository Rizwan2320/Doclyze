import nltk
from typing import List
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger


# Download punkt tokenizer if not present (only runs once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class DocumentChunker:
    """
    Hierarchical chunker with sentence-aware child chunks.
    Guarantees no sentence is ever split mid-sentence.
    """

    def __init__(self):
        self.small_chunk_size = 1000   # Increased to fit full sentences
        self.small_overlap = 200       # Overlap in characters
        self.parent_chunk_size = 2000
        self.parent_overlap = 200

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK's Punkt tokenizer."""
        return nltk.sent_tokenize(text)

    def _group_sentences_into_chunks(
        self, 
        sentences: List[str], 
        target_size: int, 
        overlap_size: int
    ) -> List[str]:
        """
        Group sentences into chunks of approximately target_size characters.
        Never splits a sentence. Overlap by roughly overlap_size characters.
        """
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence exceeds target and we have content, finalize chunk
            if current_size + sentence_len > target_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Build overlap: keep sentences from the end that fit within overlap_size
                overlap_chunk = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= overlap_size:
                        overlap_chunk.insert(0, s)
                        overlap_len += len(s) + 1  # +1 for space
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_size = overlap_len

            current_chunk.append(sentence)
            current_size += sentence_len + 1  # +1 for space

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Create hierarchical chunks with sentence-aware child chunks."""
        if not documents:
            return []

        # Filter out noise
        documents = [
            doc for doc in documents
            if len(doc.page_content.strip()) >= 50
        ]
        logger.info(f"After filtering: {len(documents)} meaningful elements")

        # Create parent chunks using RecursiveCharacterTextSplitter
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_overlap,
            separators=["\n\n", "\n", ". ", ".\n", ".", " ", ""],
        )
        parent_chunks = parent_splitter.split_documents(documents)

        # Assign parent IDs and chunk_type
        for parent in parent_chunks:
            parent.metadata["chunk_id"] = str(uuid4())
            parent.metadata["chunk_type"] = "parent"

        # Create sentence-aware child chunks
        small_chunks: List[Document] = []

        for parent in parent_chunks:
            sentences = self._split_into_sentences(parent.page_content)
            child_texts = self._group_sentences_into_chunks(
                sentences, 
                target_size=self.small_chunk_size, 
                overlap_size=self.small_overlap
            )

            for child_text in child_texts:
                child_doc = Document(
                    page_content=child_text,
                    metadata={
                        **parent.metadata,
                        "chunk_id": str(uuid4()),
                        "chunk_type": "child",
                        "parent_id": parent.metadata["chunk_id"],
                    }
                )
                small_chunks.append(child_doc)

        all_chunks = small_chunks + parent_chunks

        logger.info(
            f"Created {len(small_chunks)} child chunks + "
            f"{len(parent_chunks)} parent chunks"
        )

        return all_chunks


# Singleton
chunker = DocumentChunker()