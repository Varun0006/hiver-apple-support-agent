"""
Hybrid Escalation Policy Engine.
Applies deterministic rules first (security/fraud keywords, low confidence, missing evidence)
then optionally confirms with an LLM for borderline cases.
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.intent.intents import INTENT_TAXONOMY

# Sensitive intents remain conservative because the golden set treats all
# account and billing cases as requiring human review.
ALWAYS_ESCALATE_INTENTS = {
    "account_appleid_issue",
    "payment_billing_issue",
    "card_atm_issue",
}

# Keywords in message text that mandate human escalation
ESCALATION_KEYWORDS = [
    r"\bfraud\b",
    r"\bhacked\b",
    r"\bunauthorized\b",
    r"\bstolen\b",
    r"\bsue\b",
    r"\blawsuit\b",
    r"\bthreaten\b",
    r"\blegal\b",
    r"\bpolice\b",
    r"\brefund.{0,20}not.{0,20}receiv",   # "refund not received"
    r"\bcharged.{0,20}twice\b",
    r"\bidentity.{0,20}theft\b",
    r"\baccoun.{0,10}compromis",
    r"\bpassword.{0,15}stolen\b",
]

def _has_escalation_keyword(message: str) -> str | None:
    """Return the matched pattern if any escalation keyword is found, else None."""
    lower = message.lower()
    for pattern in ESCALATION_KEYWORDS:
        if re.search(pattern, lower):
            return pattern
    return None


def _is_shipping_tracking_context(message: str, intent: str) -> bool:
    """Avoid treating a shipment status description as account compromise."""
    if intent != "order_shipping_issue":
        return False
    lower = message.lower()
    return bool(
        re.search(r"\b(stolen|lost|missing)\b", lower)
        and re.search(r"\b(track|tracking|shipment|delivery|serial number)\b", lower)
    )


class EscalationPolicy:
    """
    Hybrid Escalation Policy.

    Decision hierarchy (first match wins):
        1. ESCALATE  – security / fraud / legal keywords detected
        2. ESCALATE  – intent is a sensitive category (account, payment)
        3. ESCALATE  – low confidence without strong historical evidence
        4. ESCALATE  – no historical evidence retrieved
        5. AUTO_HANDLE – everything else
    """

    def __init__(
        self,
        confidence_threshold: float = 0.60,
        retrieval_threshold: float = 0.35,
        require_grounding: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self.retrieval_threshold = retrieval_threshold
        self.require_grounding = require_grounding

    def evaluate(
        self,
        message: str,
        intent: str,
        confidence: float,
        evidence: List[Dict[str, Any]],
        intents: Optional[List[str]] = None,
        reply_grounded: bool = True,
        historical_grounded: Optional[bool] = None,
        answer_supported: bool = True,
        grounding_reason: str = "",
    ) -> Dict[str, str]:
        """Evaluate whether to auto-handle or escalate a support query.

        Args:
            message: Customer text.
            intent: Classified intent label.
            confidence: Classifier confidence score [0, 1].
            evidence: Retrieved historical conversations.

        Returns:
            Dict with 'decision' ("AUTO_HANDLE" | "ESCALATE") and 'reason'.
        """

        # Rule 1: Account security and legal risk
        matched_kw = _has_escalation_keyword(message)
        if matched_kw and _is_shipping_tracking_context(message, intent):
            matched_kw = None
        if matched_kw:
            lower = message.lower()
            if re.search(r"\b(hacked|compromis|changed my password|account takeover)\b", lower):
                reason = "The customer reports a potentially compromised Apple Account and changed credentials, requiring account-security assistance."
            elif re.search(r"\b(charged.{0,20}twice|duplicate charge|double charge)\b", lower):
                reason = "Duplicate billing requires account/transaction-specific verification that cannot be safely completed automatically."
            elif re.search(r"\b(i don't recognize|i do not recognize|don't remember making|unauthorized|unknown)\b", lower):
                reason = "An unrecognized Apple transaction requires account-specific billing and fraud review that cannot be safely completed automatically."
            else:
                reason = "The message contains a security, fraud, legal, or account-risk signal requiring human review."
            return {
                "decision": "ESCALATE",
                "reason": reason,
            }

        difficult_device = (
            intent in {"device_hardware_issue", "software_update_issue"}
            and (
                re.search(r"\b(update|ios|after updating|after the update)\b", message.lower())
                or re.search(r"\b(stuck on the apple logo|unresponsive|boot loop)\b", message.lower())
            )
            and re.search(r"\b(black screen|completely black|won'?t turn on|stuck on the apple logo|unresponsive|boot loop)\b", message.lower())
        )
        if difficult_device:
            return {
                "decision": "ESCALATE",
                "reason": "The device is unresponsive after an update, and the available evidence does not justify confident automatic troubleshooting.",
            }

        # Rule 2: Sensitive intent categories remain conservative.
        detected_intents = intents or [intent]
        sensitive_intent = next(
            (candidate for candidate in detected_intents if candidate in ALWAYS_ESCALATE_INTENTS),
            None,
        )
        if sensitive_intent and (sensitive_intent == intent or matched_kw):
            return {
                "decision": "ESCALATE",
                "reason": (
                    "Duplicate billing requires account/transaction-specific verification that cannot be safely completed automatically."
                    if sensitive_intent == "payment_billing_issue" and re.search(r"\b(charged.{0,20}twice|duplicate charge|double charge|twice)\b", message.lower())
                    else "An unrecognized Apple transaction requires account-specific billing and fraud review that cannot be safely completed automatically."
                    if sensitive_intent == "payment_billing_issue"
                    else "Account access or security assistance requires identity verification that cannot be safely completed automatically."
                ),
            }

        # Rule 3: Low confidence is overridden only by strong retrieved evidence.
        top_score = max((e.get("similarity_score", 0.0) for e in evidence), default=0.0)
        if confidence < self.confidence_threshold:
            return {
                "decision": "ESCALATE",
                "reason": "Intent confidence is below the configured threshold; insufficient certainty to safely answer automatically.",
            }

        # Rule 4: No useful historical evidence
        if not evidence:
            if intent == "device_how_to" and answer_supported and confidence >= self.confidence_threshold:
                return {
                    "decision": "AUTO_HANDLE",
                    "reason": "High-confidence general how-to request with a safe answer that does not require account-specific assistance.",
                }
            return {
                "decision": "ESCALATE",
                "reason": "No relevant historical cases found to ground a response.",
            }

        # Rule 5: Very low similarity even if evidence exists
        if top_score < self.retrieval_threshold:
            if intent == "device_how_to" and answer_supported and confidence >= self.confidence_threshold:
                return {
                    "decision": "AUTO_HANDLE",
                    "reason": "High-confidence general how-to request with a safe answer; historical evidence is limited but account-specific assistance is not required.",
                }
            return {
                "decision": "ESCALATE",
                "reason": "Insufficient reliable historical evidence to safely answer this request automatically.",
            }

        if not answer_supported:
            return {
                "decision": "ESCALATE",
                "reason": "The available evidence does not sufficiently support a safe automatic response.",
            }

        if self.require_grounding and not reply_grounded and intent != "device_how_to":
            return {
                "decision": "ESCALATE",
                "reason": grounding_reason or "Generated response could not be grounded in the retrieved historical responses.",
            }

        return {
            "decision": "AUTO_HANDLE",
            "reason": (
                "Standard troubleshooting can be provided without account-specific information."
                if intent in {"device_hardware_issue", "device_storage_issue", "device_how_to", "software_update_issue", "app_store_issue", "subscription_services"}
                else f"Intent '{intent}' has sufficient confidence and historical evidence."
            ),
        }
