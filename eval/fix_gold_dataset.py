"""
Regenerate gold dataset chunk IDs from the actual vector DB.
Run this AFTER indexing your document and BEFORE running eval.
Usage:
    python eval/fix_gold_dataset.py
"""
import json
import re
from pathlib import Path
from typing import List, Dict
import requests
from loguru import logger

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = "http://localhost:8000"
GOLD_INPUT = Path("eval/gold_dataset.json")
GOLD_OUTPUT = Path("eval/gold_dataset_v2.json")
# Minimum text length to use as a search phrase (avoid stopwords)
MIN_PHRASE_LEN = 12


def fetch_chunks(document_id: str) -> List[Dict]:
    """Fetch all chunks from the debug endpoint."""
    url = f"{BASE_URL}/api/debug/chunks/{document_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("chunks", [])
    except Exception as e:
        logger.error(f"Failed to fetch chunks for {document_id}: {e}")
        return []


def extract_search_phrases(expected_answer: str, max_phrases: int = 3) -> List[str]:
    """
    Extract key phrases from the expected answer to search for in chunks.
    We pick the longest sentences/fragments — they are least ambiguous.
    """
    # Split into sentences
    sentences = re.split(r'[.!?]\s+', expected_answer)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= MIN_PHRASE_LEN]
    
    # Sort by length (longest first) — longer phrases are more unique
    sentences.sort(key=len, reverse=True)
    
    return sentences[:max_phrases]


def find_matching_chunks(chunks: List[Dict], phrases: List[str]) -> List[str]:
    """
    Find chunk IDs whose text contains any of the search phrases.
    Uses case-insensitive substring matching.
    """
    matched_ids = []
    for chunk in chunks:
        text = chunk.get("text_preview", "").lower()
        chunk_id = chunk.get("chunk_id")
        # Skip chunks with no ID to avoid appending None
        if chunk_id is None:
            continue
        for phrase in phrases:
            if phrase.lower() in text:
                matched_ids.append(chunk_id)
                break  # One match per chunk is enough
    return matched_ids


def find_chunks_by_page(chunks: List[Dict], target_pages: List[int]) -> List[str]:
    """Fallback: find chunks from specific pages."""
    matched = []
    for chunk in chunks:
        page = chunk.get("page_number")
        chunk_id = chunk.get("chunk_id")
        if page is None or chunk_id is None:
            continue
        if page in target_pages:
            matched.append(chunk_id)
    return matched


def main():
    if not GOLD_INPUT.exists():
        logger.error(f"Gold dataset not found: {GOLD_INPUT}")
        return
    
    # Load current gold dataset
    gold_raw = json.loads(GOLD_INPUT.read_text(encoding="utf-8"))
    questions = gold_raw if isinstance(gold_raw, list) else gold_raw.get("questions", [])
    
    if not questions:
        logger.warning("No questions found in gold dataset. Exiting.")
        return
    
    logger.info(f"Loaded {len(questions)} questions from {GOLD_INPUT}")
    
    # Cache chunks per document to avoid repeated API calls
    doc_chunks: Dict[str, List[Dict]] = {}
    updated_questions = []
    stats = {"matched": 0, "fallback_to_page": 0, "unmatched": 0}
    
    for q in questions:
        # ── Safely extract fields with defaults ─────────────────────────────
        qid = q.get("id", "unknown")
        doc_id = q.get("document_id", "")
        expected_answer = q.get("expected_answer", "")
        target_pages = q.get("source_pages", [])
        old_chunk_ids = q.get("expected_chunk_ids", [])
        
        if not doc_id:
            logger.warning(f" {qid}: Skipping — no document_id provided")
            stats["unmatched"] += 1
            continue
        
        logger.info(f"Running {qid}: {str(expected_answer)[:60]}...")
        
        # Fetch chunks for this document (cached)
        if doc_id not in doc_chunks:
            logger.info(f"Fetching chunks for document {doc_id}...")
            doc_chunks[doc_id] = fetch_chunks(doc_id)
            logger.info(f" -> Found {len(doc_chunks[doc_id])} chunks")
        chunks = doc_chunks[doc_id]
        
        # Strategy 1: Search by text phrases from expected answer
        phrases = extract_search_phrases(expected_answer)
        matched_ids = find_matching_chunks(chunks, phrases)
        
        # Strategy 2: If text search fails, fall back to page numbers
        used_fallback = False
        if not matched_ids and target_pages:
            logger.warning(f" {qid}: Text match failed, trying page fallback...")
            matched_ids = find_chunks_by_page(chunks, target_pages)
            if matched_ids:
                used_fallback = True
                stats["fallback_to_page"] += 1
        
        if matched_ids:
            stats["matched"] += 1
        else:
            stats["unmatched"] += 1
            logger.warning(f" {qid}: NO MATCH FOUND — manual review needed")
        
        # Build updated question
        updated_q = {
            **q,
            "expected_chunk_ids": matched_ids,
            "expected_retrieved_chunk_count": len(matched_ids),
            "_old_chunk_ids": old_chunk_ids,  # keep for audit
            "_match_method": (
                "text" if matched_ids and not used_fallback 
                else ("page" if matched_ids else "none")
            ),
            "_search_phrases": phrases,
        }
        updated_questions.append(updated_q)
        logger.info(
            f" {qid}: {len(matched_ids)} chunks matched "
            f"(old had {len(old_chunk_ids)})"
        )
    
    # Save updated gold dataset
    GOLD_OUTPUT.write_text(
        json.dumps(updated_questions, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    logger.info(f"\nSaved updated gold dataset to {GOLD_OUTPUT}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("FIX SUMMARY")
    print("-" * 60)
    print(f"Total questions processed: {len(updated_questions)}")
    print(f"Matched by text: {stats['matched'] - stats['fallback_to_page']}")
    print(f"Matched by page: {stats['fallback_to_page']}")
    print(f"UNMATCHED (manual): {stats['unmatched']}")
    print("=" * 60)
    
    # Print verification table for first 5 questions
    print("\nVERIFICATION TABLE (first 5 questions):")
    print(f"{'ID':<6} {'Old IDs':<12} {'New IDs':<12} {'Method':<10}")
    print("-" * 50)
    for q in updated_questions[:5]:
        old = len(q["_old_chunk_ids"])
        new = len(q["expected_chunk_ids"])
        method = q["_match_method"]
        print(f"{q['id']:<6} {old:<12} {new:<12} {method:<10}")


if __name__ == "__main__":
    main()