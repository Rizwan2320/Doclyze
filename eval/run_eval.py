"""
Baseline eval runner for Doclyze.
Collects answers, chunk IDs, and full retrieved contexts for later RAGAS scoring in Colab.
Run from project root:
    python eval/run_eval.py
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import requests
from loguru import logger

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = "http://localhost:8000"
GOLD_DATASET_PATH = Path("eval/gold_dataset_v2.json")
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


def compute_chunk_metrics(
    retrieved_ids: List[str],
    expected_ids: List[str],
    actual_answer: str = "",  # NEW: pass the LLM's answer for proper refusal detection
) -> dict:
    """
    Strict chunk-ID-based precision and recall.
    For adversarial questions, expected_ids is empty.
    
    FIXED (2026-05-14): correct_refusal now measures SYSTEM behavior (retriever + LLM),
    not just retriever behavior. The LLM can correctly refuse even if chunks were retrieved.
    """
    retrieved_set = set(retrieved_ids)
    expected_set = set(expected_ids)
    
    # Adversarial question: no chunks SHOULD contain the answer
    if len(expected_set) == 0:
        # System-level refusal check: did the LLM actually refuse?
        refusal_phrases = [
            "cannot answer this based on the provided document",
            "i cannot answer",
            "not found in the document",
            "no information",
        ]
        answer_lower = actual_answer.lower().strip()
        llm_refused = any(phrase in answer_lower for phrase in refusal_phrases)
        
        # Also check for garbage contamination (FM-006)
        if llm_refused:
            # Find where refusal phrase starts
            refusal_start = len(answer_lower)
            for phrase in refusal_phrases:
                pos = answer_lower.find(phrase)
                if pos != -1 and pos < refusal_start:
                    refusal_start = pos
            
            # If there's substantial text before refusal (>150 chars), mark as contaminated
            if refusal_start > 150:
                return {
                    "chunk_precision": 0.0,
                    "chunk_recall": 0.0,
                    "correct_refusal": False,  # Contaminated refusal (FM-006)
                    "refusal_contaminated": True,
                    "refusal_note": "LLM refused but dumped context first",
                }
            
            # Clean refusal
            return {
                "chunk_precision": 0.0,
                "chunk_recall": 0.0,
                "correct_refusal": True,
                "refusal_contaminated": False,
            }
        else:
            # LLM did not refuse at all — system failure
            return {
                "chunk_precision": 0.0,
                "chunk_recall": 0.0,
                "correct_refusal": False,
                "refusal_contaminated": False,
            }
    
    # Normal question: compute precision/recall
    intersection = retrieved_set & expected_set
    precision = len(intersection) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(intersection) / len(expected_set) if expected_set else 0.0
    
    return {
        "chunk_precision": round(precision, 4),
        "chunk_recall": round(recall, 4),
        "correct_refusal": None,
        "refusal_contaminated": False,
    }


def run_eval():
    gold_raw = json.loads(GOLD_DATASET_PATH.read_text())
    gold = gold_raw if isinstance(gold_raw, list) else gold_raw.get("questions", [])

    if not gold:
        logger.warning("No questions found in gold dataset. Exiting.")
        return

    logger.info(f"Loaded {len(gold)} gold questions")

    results = []
    for item in gold:
        # ── Safely extract fields with defaults ─────────────────────────────
        qid = item.get("id", "unknown")
        question = item.get("question", "")
        document_id = item.get("document_id", "")
        expected_answer = item.get("expected_answer", "")
        expected_chunk_ids = item.get("expected_chunk_ids", [])
        category = item.get("category", "unknown")
        top_k = item.get("top_k", 10)

        logger.info(f"Running {qid} [{category}]: {question[:60]}...")

        try:
            start = time.time()
            response = query_doclyze(question, str(document_id), top_k=top_k)
            latency_ms = (time.time() - start) * 1000

            # Safer access to response fields
            actual_answer = response.get("answer", "")
            sources = response.get("sources", [])

            # Extract retrieved chunk IDs (preserve chunk_id=0)
            retrieved_chunk_ids = [
                s.get("chunk_id") for s in sources if s.get("chunk_id") is not None
            ]

            # Safer access to source texts
            retrieved_texts = [s.get("text", "") for s in sources]

            # Chunk-ID metrics
            chunk_metrics = compute_chunk_metrics(
                retrieved_chunk_ids, expected_chunk_ids, actual_answer
            )

            result = {
                "id": qid,
                "category": category,
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": actual_answer,
                "contexts": retrieved_texts,  # FULL CHUNK TEXTS FOR COLAB RAGAS
                "expected_chunk_ids": expected_chunk_ids,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "chunk_precision": chunk_metrics["chunk_precision"],
                "chunk_recall": chunk_metrics["chunk_recall"],
                "correct_refusal": chunk_metrics["correct_refusal"],
                "refusal_contaminated": chunk_metrics.get("refusal_contaminated", False),
                "num_retrieved": len(sources),
                "latency_ms": round(latency_ms, 1),
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
        cp = result.get("chunk_precision")
        cr = result.get("chunk_recall")
        latency = result.get("latency_ms")
        logger.info(f" chunk_precision={cp} | chunk_recall={cr} | latency={latency}ms")

    # Save baseline results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"baseline_{timestamp}.json"
    output_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'ID':<6} {'Category':<15} {'Chunk P':<10} {'Chunk R':<10} {'Latency':>8}")
    print("-" * 80)
    for r in results:
        if "error" not in r:
            cp = f"{r['chunk_precision']:.2f}" if r.get("chunk_precision") is not None else "N/A"
            cr = f"{r['chunk_recall']:.2f}" if r.get("chunk_recall") is not None else "N/A"
            print(f"{r['id']:<6} {r['category']:<15} {cp:<10} {cr:<10} {r['latency_ms']:>7}ms")
    print("=" * 80)

    # Category summary (dynamic — not hardcoded)
    print("\nBy category:")
    categories = sorted(set(
        r.get("category") for r in results if "error" not in r
    ))
    for cat in categories:
        cat_results = [
            r for r in results
            if r.get("category") == cat and "error" not in r
        ]
        if cat_results:
            precisions = [r["chunk_precision"] for r in cat_results if r.get("chunk_precision") is not None]
            recalls = [r["chunk_recall"] for r in cat_results if r.get("chunk_recall") is not None]
            avg_p = sum(precisions) / len(precisions) if precisions else 0
            avg_r = sum(recalls) / len(recalls) if recalls else 0
            print(f" {cat:<15}: precision={avg_p:.2f}, recall={avg_r:.2f} ({len(cat_results)} questions)")

    # Failure highlights
    print("\nFailure highlights:")
    for r in results:
        if "error" in r:
            continue
        cp = r.get("chunk_precision")
        cr = r.get("chunk_recall")
        # Only flag if metrics exist and are imperfect
        if cp is not None and cr is not None and (cp < 1.0 or cr < 1.0):
            print(f" {r['id']} [{r['category']}]: precision={cp}, recall={cr}")
            print(f" expected: {r.get('expected_chunk_ids', [])}")
            print(f" retrieved: {r.get('retrieved_chunk_ids', [])}")


if __name__ == "__main__":
    run_eval()