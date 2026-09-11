"""
Evaluation metrics for Intent Classification (Accuracy, Precision, Recall, Macro-F1).
"""

from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


def compute_intent_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Computes comprehensive metrics for predicted vs true intents.

    Args:
        y_true: Ground truth intent labels.
        y_pred: Predicted intent labels.

    Returns:
        Dict containing Accuracy, Macro-F1, Precision, Recall, and per-class metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "classification_report": report,
    }
