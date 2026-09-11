"""
Intent Taxonomy Exploration Script.
Analyzes frequent n-grams and topic clusters in customer initial queries to validate intent categories.
"""

import sys
from pathlib import Path
import pandas as pd
from collections import Counter
import re

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def explore_intents(input_file: str):
    csv_path = Path(input_file)
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return

    print(f"Loading conversations from {csv_path}...")
    df = pd.read_csv(csv_path)
    queries = df["initial_query"].dropna().astype(str).tolist()
    print(f"Total customer initial queries: {len(queries):,}\n")

    # Simple word frequencies
    words = []
    for q in queries:
        cleaned = re.sub(r"[^\w\s]", "", q.lower())
        words.extend([w for w in cleaned.split() if len(w) > 3])

    counter = Counter(words)
    print("--- Top 20 Most Frequent Keywords ---")
    for word, count in counter.most_common(20):
        print(f"  {word}: {count:,}")


if __name__ == "__main__":
    explore_intents("data/processed/apple_conversations.csv")
