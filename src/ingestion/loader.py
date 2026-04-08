from pathlib import Path
from typing import List

import fitz  # pymupdf
from langchain_core.documents import Document
from loguru import logger


class DocumentLoader:

    def load_file(self, file_path: str | Path) -> List[Document]:
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self._load_pdf(file_path)
        elif suffix in {".txt", ".md"}:
            return self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _load_pdf(self, file_path: Path) -> List[Document]:
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
                        "source": file_path.name,
                        "page_number": page_num + 1,
                        "filetype": ".pdf",
                    }
                ))

        doc.close()
        logger.info(f"✅ Extracted {len(documents)} pages from {file_path.name}")
        return documents

    def _load_text(self, file_path: Path) -> List[Document]:
        text = file_path.read_text(encoding="utf-8")
        logger.info(f"📄 Loaded text file: {file_path.name}")
        return [Document(
            page_content=text,
            metadata={
                "source": file_path.name,
                "filetype": file_path.suffix,
            }
        )]


# Singleton
loader = DocumentLoader()