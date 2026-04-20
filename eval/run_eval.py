"""
Baseline eval runner for Doclyze.
Step 1: Queries the system for each gold question and saves raw results.
Step 2: RAGAS scoring added in next iteration.

Run from project root:
    python eval/run_eval.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

import requests
from loguru import logger

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = "http://localhost:8000"
GOLD_DATASET_PATH = Path("eval/gold_dataset.json")
RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def query_doclyze(question: str, document_id: str, top_k: int = 10) -> dict:
    """Send one question to the /api/query endpoint."""
    payload = {
        "query": question,
        "document_id": document_id,
        "top_k": top_k,
    }
    response = requests.post(f"{BASE_URL}/api/query", json=payload)
    response.raise_for_status()
    return response.json()


def run_eval():
    # Load gold dataset
    gold = json.loads(GOLD_DATASET_PATH.read_text())
    logger.info(f"Loaded {len(gold)} gold questions")

    results = []

    for item in gold:
        qid = item["id"]
        question = item["question"]
        document_id = item["document_id"]
        expected_answer = item["expected_answer"]
        expected_pages = item["source_pages"]
        category = item["category"]

        logger.info(f"Running {qid} [{category}]: {question[:60]}...")

        try:
            start = time.time()
            response = query_doclyze(question, document_id)
            latency_ms = (time.time() - start) * 1000

            actual_answer = response["answer"]
            sources = response["sources"]
            retrieved_pages = [s["page_number"] for s in sources]
            retrieved_scores = [s["score"] for s in sources]

            # Simple page hit check: did we retrieve the expected pages?
            page_hits = [p for p in expected_pages if p in retrieved_pages]
            page_recall = len(page_hits) / len(expected_pages) if expected_pages else None

            result = {
                "id": qid,
                "category": category,
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": actual_answer,
                "expected_pages": expected_pages,
                "retrieved_pages": retrieved_pages[:5],  # top 5 only
                "page_recall": page_recall,
                "top_score": max(retrieved_scores) if retrieved_scores else 0,
                "avg_score": round(sum(retrieved_scores) / len(retrieved_scores), 4) if retrieved_scores else 0,
                "latency_ms": round(latency_ms, 1),
                "num_sources": len(sources),
            }

        except Exception as e:
            logger.error(f"Failed {qid}: {e}")
            result = {
                "id": qid,
                "category": category,
                "question": question,
                "error": str(e),
            }

        results.append(result)
        logger.info(f"  ✓ page_recall={result.get('page_recall')} | latency={result.get('latency_ms')}ms")

    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"baseline_{timestamp}.json"
    output_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved to {output_path}")

    # Print summary table
    print("\n" + "="*70)
    print(f"{'ID':<6} {'Category':<15} {'Page Recall':<14} {'Top Score':<12} {'Latency':>8}")
    print("-"*70)
    for r in results:
        if "error" not in r:
            pr = f"{r['page_recall']:.2f}" if r['page_recall'] is not None else "N/A"
            print(f"{r['id']:<6} {r['category']:<15} {pr:<14} {r['top_score']:<12} {r['latency_ms']:>7}ms")
    print("="*70)

    # Category summary
    print("\nBy category:")
    for cat in ["factual", "multi-section", "adversarial", "failure-mode"]:
        cat_results = [r for r in results if r.get("category") == cat and "error" not in r]
        if cat_results:
            recalls = [r["page_recall"] for r in cat_results if r["page_recall"] is not None]
            avg_recall = sum(recalls) / len(recalls) if recalls else 0
            print(f"  {cat:<15}: avg page_recall={avg_recall:.2f} ({len(cat_results)} questions)")


if __name__ == "__main__":
    run_eval()