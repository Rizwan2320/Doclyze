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
from typing import List
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
    actual_answer: str = "",
) -> dict:
    """
    Strict chunk-ID-based precision and recall.
    For adversarial questions, expected_ids is empty.
    """
    retrieved_set = set(retrieved_ids)
    expected_set = set(expected_ids)
    
    if len(expected_set) == 0:
        refusal_phrases = [
            "cannot answer this based on the provided document",
            "i cannot answer",
            "not found in the document",
            "no information",
        ]
        answer_lower = actual_answer.lower().strip()
        llm_refused = any(phrase in answer_lower for phrase in refusal_phrases)
        
        if llm_refused:
            refusal_start = len(answer_lower)
            for phrase in refusal_phrases:
                pos = answer_lower.find(phrase)
                if pos != -1 and pos < refusal_start:
                    refusal_start = pos
            
            if refusal_start > 150:
                return {
                    "chunk_precision": 0.0,
                    "chunk_recall": 0.0,
                    "correct_refusal": False,
                    "refusal_contaminated": True,
                    "refusal_note": "LLM refused but dumped context first",
                }
            
            return {
                "chunk_precision": 0.0,
                "chunk_recall": 0.0,
                "correct_refusal": True,
                "refusal_contaminated": False,
            }
        else:
            return {
                "chunk_precision": 0.0,
                "chunk_recall": 0.0,
                "correct_refusal": False,
                "refusal_contaminated": False,
            }
    
    intersection = retrieved_set & expected_set
    precision = len(intersection) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(intersection) / len(expected_set) if expected_set else 0.0
    
    return {
        "chunk_precision": round(precision, 4),
        "chunk_recall": round(recall, 4),
        "correct_refusal": None,
        "refusal_contaminated": False,
    }


def compute_position_metrics(
    retrieved_ids: List[str],
    expected_ids: List[str],
    ks: List[int] = [1, 3, 5, 10]
) -> dict:
    """
    Measures WHERE relevant chunks appear in the ranked retrieval list.

    WHY THIS MATTERS:
    chunk_recall tells you IF relevant chunks were retrieved.
    Position metrics tell you HOW EARLY they appeared.

    A relevant chunk at position 1 vs position 9 is a huge difference —
    the LLM reads context top-to-bottom and earlier chunks have more
    influence on the generated answer.

    hit@k = 1 if ANY expected chunk appears in the top-k results.
    position_of_first_hit = rank of the first relevant chunk (1-indexed).
    None means no relevant chunk was retrieved at all.
    """
    expected_set = set(expected_ids)

    if not expected_set:
        return {f"hit@{k}": None for k in ks} | {"position_of_first_hit": None}

    hits = {f"hit@{k}": 0 for k in ks}
    position_of_first_hit = None

    for i, rid in enumerate(retrieved_ids):
        if rid in expected_set:
            if position_of_first_hit is None:
                position_of_first_hit = i + 1
            for k in ks:
                if i < k:
                    hits[f"hit@{k}"] = 1

    return {**hits, "position_of_first_hit": position_of_first_hit}


def run_eval():
    gold_raw = json.loads(GOLD_DATASET_PATH.read_text())
    gold = gold_raw if isinstance(gold_raw, list) else gold_raw.get("questions", [])

    if not gold:
        logger.warning("No questions found in gold dataset. Exiting.")
        return

    logger.info(f"Loaded {len(gold)} gold questions")

    results = []
    for item in gold:
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

            actual_answer = response.get("answer", "")
            sources = response.get("sources", [])

            retrieved_chunk_ids = [
                s.get("chunk_id") for s in sources if s.get("chunk_id") is not None
            ]
            retrieved_texts = [s.get("text", "") for s in sources]

            chunk_metrics = compute_chunk_metrics(
                retrieved_chunk_ids, expected_chunk_ids, actual_answer
            )

            position_metrics = compute_position_metrics(
                retrieved_chunk_ids, expected_chunk_ids
            )

            result = {
                "id": qid,
                "category": category,
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": actual_answer,
                "contexts": retrieved_texts,
                "expected_chunk_ids": expected_chunk_ids,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "chunk_precision": chunk_metrics["chunk_precision"],
                "chunk_recall": chunk_metrics["chunk_recall"],
                "correct_refusal": chunk_metrics["correct_refusal"],
                "refusal_contaminated": chunk_metrics.get("refusal_contaminated", False),
                "num_retrieved": len(sources),
                "latency_ms": round(latency_ms, 1),
                **position_metrics,
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

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"baseline_{timestamp}.json"
    output_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved to {output_path}")

    # Print summary table (as per instruction)
    print("\n" + "=" * 80)
    print(f"{'ID':<6} {'Category':<15} {'Chunk P':<10} {'Chunk R':<10} {'hit@1':<6} {'pos':<6} {'Latency':>8}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            continue
        cp = f"{r['chunk_precision']:.2f}" if r.get("chunk_precision") is not None else "N/A"
        cr = f"{r['chunk_recall']:.2f}" if r.get("chunk_recall") is not None else "N/A"
        h1 = str(r.get('hit@1', 'N/A'))
        pos = str(r.get('position_of_first_hit', 'N/A'))
        print(f"{r['id']:<6} {r['category']:<15} {cp:<10} {cr:<10} {h1:<6} {pos:<6} {r['latency_ms']:>7}ms")
    print("=" * 80)

    # ... rest of your summary code (category summary, failure highlights) remains the same


if __name__ == "__main__":
    run_eval()