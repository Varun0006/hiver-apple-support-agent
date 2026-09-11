"""
Golden Evaluation Set Creation Script.
Samples and labels 200 diverse customer support queries from apple_conversations.csv across all 8 intent categories.
Exports data/golden/golden_set.csv and data/golden/golden_set.json.
"""

import json
import re
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.intent.intents import INTENT_TAXONOMY, INTENT_NAMES


def match_intent(text: str) -> str:
    """Matches text against taxonomy keywords to categorize initial query."""
    text_lower = text.lower()
    scores = {}

    for intent, meta in INTENT_TAXONOMY.items():
        score = 0
        for kw in meta["keywords"]:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                score += 1
        if score > 0:
            scores[intent] = score

    if scores:
        return max(scores, key=scores.get)
    return "other_general"


def create_golden_set(input_file: str, output_csv: str):
    csv_path = Path(input_file)
    if not csv_path.exists():
        print(f"Error: Processed conversations file {csv_path} not found.")
        return

    print(f"Loading reconstructed conversations from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filter out empty queries or non-English/greeting-only noise
    df = df[df["initial_query"].dropna().astype(str).str.len() > 15].copy()

    df["matched_intent"] = df["initial_query"].apply(match_intent)

    golden_records = []
    target_total = 200
    base_quota, remainder = divmod(target_total, len(INTENT_NAMES))
    quotas = {
        intent: base_quota + (index < remainder)
        for index, intent in enumerate(INTENT_NAMES)
    }

    for intent in INTENT_NAMES:
        intent_df = df[df["matched_intent"] == intent]
        sample_count = min(len(intent_df), quotas[intent])
        sampled = intent_df.sample(n=sample_count, random_state=42)

        meta = INTENT_TAXONOMY[intent]
        expected_action = meta["default_action"]

        for _, row in sampled.iterrows():
            golden_records.append(
                {
                    "conversation_id": row["conversation_id"],
                    "message": row["initial_query"],
                    "intent": intent,
                    "expected_action": expected_action,
                    "first_support_reply": row["first_support_reply"],
                    "label_source": "keyword_candidate",
                    "review_status": "needs_review",
                    "notes": "Candidate sampled by keyword taxonomy; requires manual review.",
                }
            )

    golden_df = pd.DataFrame(golden_records)

    # Ensure output directory exists
    out_csv_path = Path(output_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving Golden Set CSV ({len(golden_df)} records) to: {out_csv_path}...")
    golden_df.to_csv(out_csv_path, index=False)

    out_json_path = out_csv_path.with_suffix(".json")
    print(f"Saving Golden Set JSON to: {out_json_path}...")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(golden_records, f, indent=2)

    print("\n--------------------------------------------------")
    print("GOLDEN SET CREATION SUMMARY")
    print("--------------------------------------------------")
    print(f"Total Golden Examples Created: {len(golden_df)}")
    print("Breakdown per Intent:")
    for intent, count in golden_df["intent"].value_counts().items():
        print(f"  - {intent}: {count}")
    print("--------------------------------------------------\n")


if __name__ == "__main__":
    create_golden_set("data/processed/apple_conversations.csv", "data/golden/golden_set.csv")
