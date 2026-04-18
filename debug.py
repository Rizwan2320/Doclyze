from src.vectorstore.chroma import vector_store

docs = vector_store.similarity_search(
    'abstract summary main contribution',
    k=5,
    collection_name='ba3c84c1-3164-4126-ae19-f12c6e9ec6fc'
)
for i, doc in enumerate(docs, 1):
    page = doc.metadata.get('page_number', 'N/A')
    print(f'--- Chunk {i} | Page {page} ---')
    print(doc.page_content[:300])
    print()
