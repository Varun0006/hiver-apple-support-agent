"""
Evaluation metrics for Escalation Policy (Precision, Recall, F1 for ESCALATE class).
"""

from typing import Dict, List, Any
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def compute_escalation_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Compute precision, recall, and F1 for escalation decisions.

    Args:
        y_true: Ground truth decisions — "ESCALATE" or "AUTO_HANDLE".
        y_pred: Predicted decisions — "ESCALATE" or "AUTO_HANDLE".

    Returns:
        Dict with precision, recall, f1 for ESCALATE class plus confusion matrix.
    """
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=["ESCALATE"],
        average="binary",
        pos_label="ESCALATE",
        zero_division=0,
    )

    try:
        cm = confusion_matrix(y_true, y_pred, labels=["ESCALATE", "AUTO_HANDLE"])
        tn, fp, fn, tp = cm[1, 1], cm[1, 0], cm[0, 1], cm[0, 0]
    except Exception:
        tn = fp = fn = tp = 0

    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }
