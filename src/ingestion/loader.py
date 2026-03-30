from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document
from loguru import logger
from unstructured.partition.auto import partition

from src.config.settings import settings


class DocumentLoader:
    """
    Production-grade Document Loader.
    
    Engineering Decisions:
    - Uses strategy='fast' by default to avoid Tesseract dependency on Windows
    - 'hi_res' can be enabled later when Tesseract is installed for better table/layout detection
    - Explicitly disables OCR to prevent TesseractNotFoundError
    - Targeted warning suppression for clean logs
    - Rich metadata for future retrieval/reranking
    """

    def __init__(self):
        # Start with 'fast' to avoid system dependencies during early development
        self.strategy: str = "fast"           # Change to "hi_res" after installing Tesseract
        self.infer_table_structure: bool = True
        self.include_page_breaks: bool = True
        self.ocr_languages: List[str] = ["eng"]

    def load_file(self, file_path: str | Path) -> List[Document]:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        logger.info(f"📄 Loading document: {file_path.name} | Strategy={self.strategy} | OCR=disabled")

        # Suppress noisy warnings
        """
        warnings.filterwarnings("ignore", category=UserWarning, module="unstructured")
        warnings.filterwarnings("ignore", category=UserWarning, module="pdfminer")
        
        """
        try:
            elements = partition(
                filename=str(file_path),
                strategy=self.strategy,
                infer_table_structure=self.infer_table_structure,
                include_page_breaks=self.include_page_breaks,
                languages=self.ocr_languages,
                # Critical: Disable OCR to avoid Tesseract dependency for now
                skip_infer_table_types=[],   # Keep table detection
            )

            documents: List[Document] = []
            for idx, element in enumerate(elements):
                metadata = {
                    "source": file_path.name,
                    "filetype": file_path.suffix.lower(),
                    "element_type": type(element).__name__,
                    "page_number": getattr(element, "page_number", None),
                    "element_id": getattr(element, "id", None),
                    "chunk_idx": idx,
                }

                # Clean text
                content = str(element).strip()
                if content:  # Skip completely empty elements
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)

            logger.info(f"✅ Successfully parsed {len(documents)} elements from {file_path.name}")
            return documents

        except Exception as e:
            logger.error(f"❌ Failed to load {file_path.name}: {e}")
            raise


        # Optional: early size validation (can also be done in router)
    def load_from_bytes(self, file_bytes: bytes, filename: str) -> List[Document]:
        """Load document from bytes safely and robustly (for FastAPI uploads)."""
        import tempfile
        import uuid

        # Sanitize filename (prevent path traversal)
        safe_name = Path(filename).name

        # Use a unique id to avoid collisions
        unique_id = uuid.uuid4().hex

        logger.info(f"Loading from bytes: {safe_name}")

        # --- FIX: These lines below were not indented correctly ---
        # Optional: early size validation
        if len(file_bytes) > 50 * 1024 * 1024:  # 50MB
            raise ValueError(f"File too large: {len(file_bytes)} bytes")

        tmp_path: Path | None = None

        try:
            # Create a secure temporary file
            with tempfile.NamedTemporaryFile(
                delete=False,
                prefix=f"upload_{unique_id}_",
                suffix=Path(safe_name).suffix,
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)

                # Write bytes safely
                tmp_file.write(file_bytes)
                tmp_file.flush()

            logger.debug(f"Temporary file created at: {tmp_path}")

            # Reuse existing logic
            documents = self.load_file(tmp_path)
            return documents

        except Exception as e:
            logger.exception(f"Failed to process uploaded file: {safe_name}")
            raise

        finally:
            # Ensure cleanup
            try:
                if tmp_path is not None and tmp_path.exists():
                    tmp_path.unlink()
                    logger.debug(f"Temporary file deleted: {tmp_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to delete temp file: {cleanup_error}")


                