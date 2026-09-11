"""One-command runner for the Hiver Support Agent project."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run_step(label: str, *args: str) -> None:
    print(f"\n=== {label} ===")
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare available data, create the candidate golden set, and run the agent."
    )
    parser.add_argument(
        "--input",
        help="Optional raw TWCS CSV. If supplied, rebuild processed conversations first.",
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Run evaluation only.",
    )
    args = parser.parse_args()

    processed_csv = ROOT / "data" / "processed" / "apple_conversations.csv"
    golden_csv = ROOT / "data" / "golden" / "golden_set.csv"

    if args.input:
        run_step(
            "Preparing conversations",
            "scripts/prepare_data.py",
            "--input",
            args.input,
            "--output",
            str(processed_csv),
        )
    elif not processed_csv.exists():
        raise SystemExit(
            "Processed data is missing. Run with --input path\\to\\twcs.csv."
        )

    if not golden_csv.exists() or args.input:
        run_step("Creating candidate golden set", "scripts/create_golden_set.py")

    run_step("Validating candidate golden set", "scripts/validate_golden_set.py")
    run_step("Running evaluation", "scripts/run_evaluation.py")

    if not args.skip_smoke_test:
        run_step("Running smoke test", "main.py")

    print("\nProject run complete.")
    print("Results: experiments/evaluation_summary.json")
    print("Note: candidate golden labels and human ratings require manual review.")


if __name__ == "__main__":
    main()