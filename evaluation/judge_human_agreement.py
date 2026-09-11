"""
Human vs LLM Judge Agreement Analysis.
Computes percentage agreement, Pearson correlation, and Spearman correlation
between human ratings and LLM judge ratings across the 5 quality dimensions.

Usage:
  - Humans rate 30–50 replies on the same 1–5 scale.
  - Save human ratings to data/golden/human_ratings.csv
  - Run this script to produce agreement metrics and save to experiments/agreement_results.csv
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DIMENSIONS = ["correctness", "groundedness", "helpfulness", "tone", "hallucination"]


def validate_human_ratings(
    ratings: pd.DataFrame,
    expected_conversation_ids: List[str] | None = None,
) -> None:
    """Validate human ratings before calculating agreement statistics.

    Raises:
        ValueError: If the file is incomplete, inconsistent, or contains
            scores outside the documented 1–5 scale.
    """
    required_columns = {"conversation_id", *DIMENSIONS}
    missing_columns = sorted(required_columns - set(ratings.columns))
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    if ratings.empty:
        raise ValueError("The human ratings file contains no rows.")

    if ratings["conversation_id"].duplicated().any():
        raise ValueError("conversation_id values must be unique.")

    numeric_scores = ratings[DIMENSIONS].apply(pd.to_numeric, errors="coerce")
    invalid_scores = (
        numeric_scores.isna()
        | ~numeric_scores.ge(1)
        | ~numeric_scores.le(5)
        | numeric_scores.mod(1).ne(0)
    )
    if invalid_scores.any().any():
        invalid_dimensions = sorted(
            invalid_scores.columns[invalid_scores.any()].tolist()
        )
        raise ValueError(
            "Every human rating must be a number from 1 to 5. "
            f"Invalid or incomplete dimensions: {', '.join(invalid_dimensions)}."
        )

    if expected_conversation_ids is not None:
        expected_ids = set(expected_conversation_ids)
        rated_ids = set(ratings["conversation_id"].astype(str))
        unknown_ids = sorted(rated_ids - expected_ids)
        if unknown_ids:
            raise ValueError(
                "Human ratings contain conversation IDs missing from the judge results: "
                + ", ".join(unknown_ids[:5])
            )


def calculate_agreement(
    human_scores: List[float],
    llm_scores: List[float],
    tolerance: int = 1,
) -> Dict[str, float]:
    """Compute agreement statistics between human and LLM scores on a single dimension.

    Args:
        human_scores: List of human ratings (1–5).
        llm_scores:   List of LLM judge ratings (1–5).
        tolerance:    Counts as "agree" if |human - llm| <= tolerance.

    Returns:
        Dict with percentage_agreement, pearson_r, spearman_r, mean_abs_error.
    """
    if len(human_scores) != len(llm_scores) or len(human_scores) == 0:
        raise ValueError("human_scores and llm_scores must be non-empty and same length.")

    h = np.array(human_scores, dtype=float)
    l = np.array(llm_scores, dtype=float)

    pct_agree = float(np.mean(np.abs(h - l) <= tolerance))
    mae = float(np.mean(np.abs(h - l)))

    if np.std(h) < 1e-9 or np.std(l) < 1e-9:
        pearson = 0.0
        spearman = 0.0
    else:
        pearson, _ = pearsonr(h, l)
        spearman, _ = spearmanr(h, l)

    return {
        "percentage_agreement": round(pct_agree * 100, 2),
        "pearson_r": round(float(pearson), 4),
        "spearman_r": round(float(spearman), 4),
        "mean_abs_error": round(mae, 4),
        "n_samples": len(human_scores),
    }


def run_agreement_analysis(
    llm_results_csv: str = "experiments/judged_results.csv",
    human_ratings_csv: str = "data/golden/human_ratings.csv",
    output_csv: str = "experiments/agreement_results.csv",
) -> pd.DataFrame:
    """Load LLM judge results and human ratings, compute agreement per dimension.

    The human_ratings_csv must have columns:
      conversation_id + one column per dimension (correctness, groundedness, etc.)

    The llm_results_csv must have:
      conversation_id + judge_<dimension> columns.
    """
    llm_path = Path(llm_results_csv)
    human_path = Path(human_ratings_csv)

    if not llm_path.exists():
        print(f"LLM results not found: {llm_path}")
        print("Run scripts/run_evaluation.py first to generate judged results.")
        return pd.DataFrame()

    if not human_path.exists():
        print(f"Human ratings not found: {human_path}")
        print("Please provide human ratings at: data/golden/human_ratings.csv")
        print("Required columns: conversation_id, correctness, groundedness, helpfulness, tone, hallucination")
        _create_human_ratings_template(llm_path, human_path)
        return pd.DataFrame()

    df_llm = pd.read_csv(llm_path)
    df_human = pd.read_csv(human_path)

    try:
        validate_human_ratings(
            df_human,
            expected_conversation_ids=df_llm["conversation_id"].astype(str).tolist(),
        )
    except ValueError as error:
        print(f"Human ratings are not ready: {error}")
        error_text = str(error)
        if "conversation IDs missing" in error_text:
            print(
                "These ratings belong to a different golden/judge run. "
                "Back up data/golden/human_ratings.csv, generate a fresh template "
                "from the current judged results, rate those rows, then rerun this script."
            )
        else:
            print("Fill every rating with an integer from 1 to 5, then rerun this script.")
        return pd.DataFrame()

    # Merge on conversation_id
    merged = df_llm.merge(df_human, on="conversation_id", suffixes=("_llm", "_human"))
    print(f"Matched {len(merged)} records with both human and LLM ratings.\n")

    rows = []
    for dim in DIMENSIONS:
        llm_col = f"judge_{dim}"
        human_col = dim

        if llm_col not in merged.columns or human_col not in merged.columns:
            print(f"  Skipping dimension '{dim}' — missing columns.")
            continue

        stats = calculate_agreement(
            human_scores=merged[human_col].tolist(),
            llm_scores=merged[llm_col].tolist(),
        )
        row = {"dimension": dim, **stats}
        rows.append(row)

        print(
            f"  {dim:<14}: "
            f"Agreement={stats['percentage_agreement']:.1f}%  "
            f"Pearson={stats['pearson_r']:.3f}  "
            f"Spearman={stats['spearman_r']:.3f}  "
            f"MAE={stats['mean_abs_error']:.3f}"
        )

    df_results = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    print(f"\nAgreement results saved to: {output_csv}")
    return df_results


def _create_human_ratings_template(llm_results_path: Path, output_path: Path):
    """Create a template CSV for human raters to fill in."""
    df_llm = pd.read_csv(llm_results_path)
    # Take up to 50 samples for human rating
    sample = df_llm.head(50)[["conversation_id", "message", "reply"]].copy()
    for dim in DIMENSIONS:
        sample[dim] = ""  # blank for human to fill in

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)
    print(f"\nCreated human rating template at: {output_path}")
    print("Fill in the 1-5 scores for each dimension and re-run this script.")


if __name__ == "__main__":
    print("=" * 52)
    print("HUMAN vs LLM JUDGE AGREEMENT ANALYSIS")
    print("=" * 52)
    run_agreement_analysis()
