from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import chunker

loader = DocumentLoader()
docs = loader.load_file(r'C:\Users\swat\Desktop\doclyze\README (3).pdf')

chunks = chunker.chunk_documents(docs)

print(f"Loaded {len(docs)} elements → Created {len(chunks)} chunks")
print("First chunk preview:", chunks[0].page_content[:300])
print("Metadata example:", chunks[0].metadata)