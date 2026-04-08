
from src.ingestion.service import ingestion_service
from src.rag.chain import rag_chain
from src.vectorstore.chroma import vector_store
from chromadb import PersistentClient
from src.config.settings import settings
import uuid

# 5 questions about the RAG paper with expected keywords in answers
EVAL_SET = [
    {
        "question": "What does RAG stand for?",
        "expected_keywords": ["retrieval", "augmented", "generation"],
    },
    {
        "question": "What knowledge source does RAG use?",
        "expected_keywords": ["wikipedia"],
    },
    {
        "question": "What are the two memory types in RAG?",
        "expected_keywords": ["parametric", "non-parametric"],
    },
    {
        "question": "What problem does RAG solve?",
        "expected_keywords": ["hallucin", "knowledge", "factual"],
    },
    {
        "question": "What generator does RAG use?",
        "expected_keywords": ["bart"],
    },
]

def score_answer(answer: str, keywords: list) -> bool:
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in keywords)

def run_eval(pdf_path: str):
    print("=== Doclyze Evaluation ===\n")

    # Ingest
    collection_name = str(uuid.uuid4())
    ingestion_service.ingest_file(pdf_path, collection_name=collection_name)
    print(f"Ingested into collection: {collection_name}\n")

    # Run eval
    passed = 0
    for i, item in enumerate(EVAL_SET, 1):
        answer = rag_chain.query(item["question"], collection_name=collection_name)
        result = score_answer(answer, item["expected_keywords"])
        passed += result
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"Q{i}: {item['question']}")
        print(f"A:  {answer[:200]}")
        print(f"    {status}\n")

    score = (passed / len(EVAL_SET)) * 100
    print(f"=== Score: {passed}/{len(EVAL_SET)} ({score:.0f}%) ===")

if __name__ == "__main__":
    run_eval("2005.11401v4 (1).pdf")