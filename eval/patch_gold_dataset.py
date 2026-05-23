# eval/patch_gold_dataset.py
import json
from pathlib import Path

GOLD_PATH = Path("eval/gold_dataset_v2.json")

# Best chunk IDs after careful review
PATCHES = {
    "q01": [
        "e0eb115b-a71a-4463-93f1-54f6e15b8fc2",   # RAG-Token definition
        "f0830cda-bf7c-44c5-981f-15b81ab3412d",   # RAG-Sequence definition
    ],
    "q02": [
        "9f6a821f-7110-483b-983e-f8a271b9344c",   # BART pre-trained with denoising objective
    ],
    "q03": [
        "7b6beee8-feea-4d1f-a7e4-7b70b3ffebcf",   # Wikipedia dump → 21M documents (past preview cutoff)
    ],
    "q04": [
        "8a25ee07-d47b-43a9-8b2d-ea934696f5c6",   # Table 1 QA scores
        "717cc58d-3388-4a4b-afa3-c3cf08318ac3",   # Table 2 MS-MARCO/Jeopardy numbers
        "af4aaec5-ad42-43c7-9317-39d5bb973c6e",   # Human eval 42.7% factuality
    ],
    "q05": [
        "f072aec3-7664-4805-8b67-3afe474213b9",   # 626M parameters breakdown
        "23c80664-f8e4-4b35-b354-d0108962982d",   # T5-11B 11B params comparison
    ],
    "q06": [
        "5e050358-0782-40b3-bb8f-92e18cff1e22",   # 70% correct with 2016 index
        "abecdbc5-7c30-4ea5-8308-331379bf2e4d",   # hot-swap without retraining
    ],
    "q07": [],   # adversarial — correct
    "q08": [],   # adversarial — correct
    "q09": [
        "51f4fbba-4ab6-44cc-b88f-c4599120c313",   # three limitations sentence
    ],
    "q10": [
        "a4efb778-dfca-4b19-9079-71b44a8357a7",   # more specific, diverse, factual
    ],
}


def main():
    if not GOLD_PATH.exists():
        print(f"❌ Error: File not found → {GOLD_PATH}")
        print("Please check the path.")
        return

    data = json.loads(GOLD_PATH.read_text())

    updated_count = 0
    for q in data:
        qid = q["id"]
        if qid in PATCHES:
            old_count = len(q.get("expected_chunk_ids", []))
            new_ids = PATCHES[qid]

            q["expected_chunk_ids"] = new_ids
            q["expected_retrieved_chunk_count"] = len(new_ids)
            q["_match_method"] = "manual_patch_v2"

            print(f"✓ Updated {qid}: {old_count} → {len(new_ids)} chunks")
            updated_count += 1

    # Save updated file
    GOLD_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    
    print(f"\n🎉 Done! Updated {updated_count} questions.")
    print(f"   File saved: {GOLD_PATH}")


if __name__ == "__main__":
    main()