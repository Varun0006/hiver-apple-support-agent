"""
Stage 1 Data Preparation Script.
Filters AppleSupport data from twcs.csv, cleans tweets, reconstructs conversation threads, and saves outputs.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.build_conversations import build_conversation_threads, conversations_to_dataframe
from src.data.load_data import extract_apple_support_tweets, load_raw_twcs


def prepare_dataset(input_file: str, output_file: str):
    start_time = time.time()
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"Error: Input file {input_path} not found.")
        return

    print("==================================================")
    print(f"STAGE 1: DATA PREPARATION & CONVERSATION BUILDING")
    print("==================================================")
    print(f"Reading raw dataset: {input_path}...")

    df_raw = load_raw_twcs(input_path)
    print(f"Total raw tweets loaded: {len(df_raw):,}")

    print("Filtering AppleSupport tweets and thread context...")
    df_apple = extract_apple_support_tweets(df_raw)
    print(f"AppleSupport relevant tweets extracted: {len(df_apple):,}")

    print("Reconstructing multi-turn conversation threads...")
    conversations = build_conversation_threads(df_apple)
    print(f"Total conversation threads reconstructed: {len(conversations):,}")

    # Convert to DataFrame and JSON
    df_conv = conversations_to_dataframe(conversations)

    # Calculate summary metrics
    with_reply = df_conv[df_conv["has_support_reply"] == True]
    avg_turns = df_conv["num_turns"].mean() if len(df_conv) > 0 else 0

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving processed conversations CSV to: {output_path}...")
    df_conv.to_csv(output_path, index=False)

    json_output_path = output_path.with_suffix(".json")
    print(f"Saving full conversation details JSON to: {json_output_path}...")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2)

    elapsed = time.time() - start_time
    print("\n--------------------------------------------------")
    print("DATA PREPARATION SUMMARY")
    print("--------------------------------------------------")
    print(f"Total Conversations:                  {len(df_conv):,}")
    print(f"Conversations with AppleSupport Reply:{len(with_reply):,}")
    print(f"Average Turns per Conversation:       {avg_turns:.2f}")
    print(f"Time Taken:                           {elapsed:.2f} seconds")
    print("--------------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare AppleSupport dataset and reconstruct conversation threads.")
    parser.add_argument(
        "--input",
        type=str,
        default="sample.csv",
        help="Input CSV file path (default: sample.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/apple_conversations.csv",
        help="Output CSV file path (default: data/processed/apple_conversations.csv)",
    )
    args = parser.parse_args()

    prepare_dataset(args.input, args.output)
