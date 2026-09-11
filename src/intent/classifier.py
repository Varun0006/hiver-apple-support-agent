"""
LLM-based Intent Classifier using Google Gemini with structured JSON output.
Falls back to keyword-based classification if the LLM call fails.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.intent.intents import INTENT_NAMES, INTENT_TAXONOMY


def _build_taxonomy_prompt_block() -> str:
    lines = []
    for name, meta in INTENT_TAXONOMY.items():
        lines.append(f'  - "{name}": {meta["description"]}')
    return "\n".join(lines)


def _keyword_classify(message: str) -> Dict[str, Any]:
    """Simple keyword scoring fallback when LLM is unavailable."""
    text = message.lower()
    if re.search(r"\b(how to|how do i|how can i)\b", text) and re.search(
        r"\b(screenshot|screen shot|screen capture|delete|remove|uninstall|record|wi[- ]?fi|dark mode|update|storage|app)\b",
        text,
    ):
        return {
            "intent": "device_how_to",
            "intents": ["device_how_to"],
            "confidence": 0.88,
            "reasoning": "Explicit device how-to request",
        }
    if re.search(
        r"\b(i don't recognize|i do not recognize|don't remember making|do not remember making|unauthorized|why did apple charge|apple transaction)\b",
        text,
    ) and re.search(r"\b(charge|charged|purchase|transaction|payment)\b", text):
        return {
            "intent": "payment_billing_issue",
            "intents": ["payment_billing_issue"],
            "confidence": 0.88,
            "reasoning": "Unrecognized or disputed Apple purchase/payment signal",
        }
    scores: Dict[str, int] = {}
    for intent, meta in INTENT_TAXONOMY.items():
        score = sum(
            1
            for kw in meta["keywords"]
            if re.search(r"\b" + re.escape(kw) + r"\b", text)
        )
        if score > 0:
            scores[intent] = score

    if scores:
        ranked = sorted(scores, key=lambda name: (-scores[name], INTENT_NAMES.index(name)))
        best = ranked[0]
        best_score = scores[best]
        detected_intents = [
            intent for intent in ranked if scores[intent] >= max(1, best_score - 1)
        ][:3]
        # Normalize score to rough confidence
        conf = min(0.55 + best_score * 0.08, 0.88)
        return {
            "intent": best,
            "intents": detected_intents,
            "confidence": round(conf, 2),
            "reasoning": f"Keyword match ({best_score} hits)",
        }

    return {
        "intent": "other_general",
        "intents": ["other_general"],
        "confidence": 0.30,
        "reasoning": "No keyword matches found; defaulting to other_general",
    }


class IntentClassifier:
    """Classifies customer messages into the intent taxonomy using Gemini with structured output."""

    SYSTEM_PROMPT = """You are an intent classification system for Apple customer support tweets.

Your task: given a customer message, return ONLY a valid JSON object (no other text) with:
- "intent": one of the allowed intent labels
- "intents": an ordered list of one or more allowed intent labels when the message contains multiple issues
- "confidence": float from 0.0 to 1.0
- "reasoning": one short sentence explaining your choice

Allowed intents and their meanings:
{taxonomy}

Rules:
- Return ONLY the JSON object, no markdown code fences or extra text.
- If the message is ambiguous, pick the best single match and lower confidence accordingly.
- Use the first item in "intents" as the primary "intent".
- For security, fraud, or account compromise signals, prefer account_appleid_issue or payment_billing_issue.
"""

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        self._client = None
        self._init_client()

    def _init_client(self):
        try:
            from dotenv import load_dotenv
            load_dotenv()
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if api_key:
                self._client = genai.Client(api_key=api_key)
        except Exception:
            self._client = None

    def classify(self, message: str) -> Dict[str, Any]:
        """Classify a customer message. Returns structured dict with intent, confidence, reasoning."""
        text = message.lower()
        keyword_result = _keyword_classify(message)
        if keyword_result["intent"] == "payment_billing_issue" and re.search(
            r"\b(i don't recognize|i do not recognize|don't remember making|do not remember making|unauthorized|why did apple charge|apple transaction)\b",
            text,
        ):
            return keyword_result
        if (
            keyword_result["intent"] in {"account_appleid_issue", "device_storage_issue", "device_hardware_issue", "device_how_to"}
            and (
                re.search(r"\b(recover|recovery|forgot|reset|apple account|apple id)\b", text)
                or re.search(r"\b(storage|free up space|low storage)\b", text)
                or re.search(r"\b(hot|overheating|overheat|won'?t turn on|no power|restart|restarting|reboots|rebooting)\b", text)
                or re.search(r"\b(screenshot|screen shot|screen capture|picture of my screen)\b", text)
                or keyword_result["intent"] == "device_how_to"
            )
        ):
            return keyword_result

        if self._client is None:
            return keyword_result

        taxonomy_block = _build_taxonomy_prompt_block()
        prompt = self.SYSTEM_PROMPT.format(taxonomy=taxonomy_block)
        full_prompt = f"{prompt}\n\nCustomer message:\n{message}"

        try:
            chat = self._client.chats.create(model=self.model_name)
            response = chat.send_message(full_prompt)
            raw = response.text.strip()
            # Strip markdown fences if present
            raw = re.sub(r"^```json?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            result = json.loads(raw)

            # Validate intent label
            if result.get("intent") not in INTENT_NAMES:
                result["intent"] = "other_general"
            raw_intents = result.get("intents", [result["intent"]])
            if not isinstance(raw_intents, list):
                raw_intents = [result["intent"]]
            result["intents"] = list(dict.fromkeys(
                intent for intent in raw_intents if intent in INTENT_NAMES
            ))[:3]
            if not result["intents"]:
                result["intents"] = [result["intent"]]
            result["intent"] = result["intents"][0]
            result["confidence"] = float(result.get("confidence", 0.5))
            result["reasoning"] = str(result.get("reasoning", ""))
            return result

        except Exception as e:
            # LLM call failed — fall back to keyword classification
            fallback = _keyword_classify(message)
            fallback["reasoning"] = f"LLM fallback ({type(e).__name__}): {fallback['reasoning']}"
            return fallback
