"""Validate golden-set structure and manual-review status."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.intent.intents import INTENT_NAMES


REQUIRED_COLUMNS = {
    "conversation_id",
    "message",
    "intent",
    "expected_action",
    "label_source",
    "review_status",
}


def validate_golden_set(path: str, require_reviewed: bool = False) -> None:
    """Validate size, schema, labels, and optional manual-review completion."""
    golden_path = Path(path)
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden set not found: {golden_path}")

    df = pd.read_csv(golden_path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if not 150 <= len(df) <= 250:
        raise ValueError(f"Golden set must contain 150–250 rows; found {len(df)}.")
    if df["conversation_id"].duplicated().any():
        raise ValueError("conversation_id values must be unique.")
    if not df["intent"].isin(INTENT_NAMES).all():
        raise ValueError("Golden set contains an intent outside the taxonomy.")
    if not df["expected_action"].isin(["AUTO_HANDLE", "ESCALATE"]).all():
        raise ValueError("expected_action must be AUTO_HANDLE or ESCALATE.")

    if require_reviewed:
        pending = int((df["review_status"] != "reviewed").sum())
        if pending:
            raise ValueError(
                f"{pending} rows still require manual review. "
                "Set review_status=reviewed only after checking intent and action."
            )

    print(f"Golden set valid: {len(df)} rows, {df['intent'].nunique()} intents.")
    if not require_reviewed:
        pending = int((df["review_status"] != "reviewed").sum())
        print(f"Manual review pending: {pending} rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/golden/golden_set.csv")
    parser.add_argument("--require-reviewed", action="store_true")
    args = parser.parse_args()
    try:
        validate_golden_set(args.file, args.require_reviewed)
    except (FileNotFoundError, ValueError) as error:
        print(f"Golden set invalid: {error}")
        raise SystemExit(1)