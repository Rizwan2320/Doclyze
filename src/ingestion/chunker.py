import nltk
from typing import List
from uuid import uuid4
import re

from src.config.settings import settings
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
        self.small_chunk_size = settings.CHUNK_SIZE
        self.small_overlap = settings.CHUNK_OVERLAP
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
            
            if current_size + sentence_len > target_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Build overlap
                overlap_chunk = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= overlap_size:
                        overlap_chunk.insert(0, s)
                        overlap_len += len(s) + 1
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_size = overlap_len

            current_chunk.append(sentence)
            current_size += sentence_len + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _is_reference_chunk(self, text: str) -> bool:
        """
        Detect reference/bibliography chunks.

        DESIGN PRINCIPLE: only fire on strong structural signals.
        Inline citations ([26] mid-sentence) must NOT trigger this.
        Only lines that START with [N] are reference entries.

        Two rules, both required to be conservative:
        Rule A: >25% of lines start with [N] — page is mostly bibliography
        Rule B: 3+ lines start with [N] AND URLs present — confirmed ref block
        """
        if not text or len(text.strip()) < 30:
            return False

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return False

        # Count lines that START with a reference entry [N]
        ref_line_count = sum(
            1 for line in lines
            if re.match(r'^\[\d+\]', line)
        )
        ref_ratio = ref_line_count / len(lines)

        # URL signal — only used as secondary confirmation
        has_urls = bool(re.search(
            r'arxiv\.org|doi\.org|aclweb\.org|openreview\.net',
            text
        ))

        # Rule A: majority of lines are reference entries
        if ref_ratio > 0.25:
            return True

        # Rule B: multiple reference entries + URLs = confirmed bibliography
        if ref_line_count >= 3 and has_urls:
            return True

        return False

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Create hierarchical chunks with sentence-aware child chunks."""
        if not documents:
            return []

        # Filter out noise and reference chunks
        documents = [
            doc for doc in documents
            if len(doc.page_content.strip()) >= 50
            and not self._is_reference_chunk(doc.page_content)
        ]

        logger.info(f"After noise filter: {len(documents)} pages remain (reference chunks removed)")

        # Use each page as a parent chunk
        parent_chunks = documents

        # Assign parent IDs and chunk_type
        for parent in parent_chunks:
            parent.metadata["chunk_id"] = str(uuid4())
            parent.metadata["chunk_type"] = "parent"

        # Create sentence-aware child chunks from each page
        small_chunks: List[Document] = []

        for parent in parent_chunks:
            sentences = self._split_into_sentences(parent.page_content)
            child_texts = self._group_sentences_into_chunks(
                sentences, 
                target_size=self.small_chunk_size, 
                overlap_size=self.small_overlap
            )

            for child_text in child_texts:
                cleaned_text = child_text.strip()
                if not cleaned_text:
                    continue
                    
                child_doc = Document(
                    page_content=cleaned_text,
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