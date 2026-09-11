"""
Aggregate reply quality scores across dimensions (Correctness, Groundedness, Helpfulness, Tone, Hallucination).
"""

from typing import Dict, List


def aggregate_reply_scores(scores: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute per-dimension average scores across a list of judge evaluations.

    Args:
        scores: List of judge score dicts (from LLMJudge.evaluate_reply).

    Returns:
        Dict mapping each dimension to its average score.
    """
    dims = ["correctness", "groundedness", "helpfulness", "tone", "hallucination"]
    if not scores:
        return {d: 0.0 for d in dims}

    totals = {d: 0.0 for d in dims}
    n = len(scores)
    for s in scores:
        for d in dims:
            totals[d] += float(s.get(d, 0.0))

    return {d: round(totals[d] / n, 3) for d in dims}
