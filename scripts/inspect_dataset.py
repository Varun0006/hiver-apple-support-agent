"""
Dataset Inspection Script.
Analyzes twcs dataset (or sample.csv) to quantify AppleSupport tweets, customer inbound messages, and conversation threads.
"""

import argparse
from pathlib import Path
import pandas as pd


def inspect_dataset(csv_path: Path):
    print(f"====================================")
    print(f"INSPECTING DATASET: {csv_path}")
    print(f"====================================")

    if not csv_path.exists():
        print(f"Error: File {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    total_tweets = len(df)
    print(f"Total Tweets in Dataset: {total_tweets:,}")

    # Check columns
    print(f"Columns: {list(df.columns)}")

    # Filter AppleSupport
    apple_outbound = df[df["author_id"].astype(str).str.lower() == "applesupport"]
    apple_inbound = df[df["text"].astype(str).str.lower().str.contains("@applesupport")]

    print(f"\n--- AppleSupport Statistics ---")
    print(f"AppleSupport Outbound Replies: {len(apple_outbound):,}")
    print(f"AppleSupport Inbound Mentions: {len(apple_inbound):,}")

    combined_apple_ids = set(apple_outbound["tweet_id"]).union(set(apple_inbound["tweet_id"]))
    print(f"Total Unique Apple-related Tweets: {len(combined_apple_ids):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect TWCS dataset for AppleSupport analysis.")
    parser.add_argument(
        "--file",
        type=str,
        default="sample.csv",
        help="Path to CSV dataset (default: sample.csv or twcs/twcs.csv)",
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists() and Path("twcs/twcs.csv").exists():
        file_path = Path("twcs/twcs.csv")

    inspect_dataset(file_path)
