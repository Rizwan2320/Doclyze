from pathlib import Path
from typing import List, Optional

import fitz  # pymupdf
from langchain_core.documents import Document
from loguru import logger


class DocumentLoader:

    def load_file(self, file_path: str | Path, original_filename: Optional[str] = None) -> List[Document]:
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self._load_pdf(file_path, original_filename=original_filename)
        elif suffix in {".txt", ".md"}:
            return self._load_text(file_path, original_filename=original_filename)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _load_pdf(self, file_path: Path, original_filename: Optional[str] = None) -> List[Document]:
        logger.info(f"📄 Loading PDF: {file_path.name}")
        doc = fitz.open(str(file_path))
        documents = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": original_filename or file_path.name,
                        "page_number": page_num + 1,
                        "filetype": ".pdf",
                    }
                ))

        doc.close()
        logger.info(f"✅ Extracted {len(documents)} pages from {original_filename or file_path.name}")
        return documents

    def _load_text(self, file_path: Path, original_filename: Optional[str] = None) -> List[Document]:
        text = file_path.read_text(encoding="utf-8")
        logger.info(f"📄 Loaded text file: {original_filename or file_path.name}")
        return [Document(
            page_content=text,
            metadata={
                "source": original_filename or file_path.name,
                "filetype": file_path.suffix,
            }
        )]


# Singleton
loader = DocumentLoader()