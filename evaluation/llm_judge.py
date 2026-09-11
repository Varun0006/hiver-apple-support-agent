"""
LLM-as-Judge evaluation harness.
Scores agent-generated replies on 5 dimensions using Gemini as an automated judge.
Falls back to heuristic scoring if no API key is configured.

Dimensions (all 1–5 scale):
  - correctness:   Does the reply correctly address the customer's problem?
  - groundedness:  Is the reply supported by historical evidence?
  - helpfulness:   Would this actually help the customer?
  - tone:          Is it appropriate, empathetic, and professional?
  - hallucination: Did the AI invent policies, refunds, or timelines? (5 = no hallucination, 1 = severe)
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class LLMJudge:
    """Evaluates agent replies using an LLM judge prompt against customer query and evidence."""

    JUDGE_PROMPT = """You are an expert evaluator of customer support responses.

You will be given:
1. A customer message
2. Historical evidence (similar resolved cases) that the agent had access to
3. The agent's generated reply

Score the reply on EACH of the following 5 dimensions from 1 to 5:

  correctness   (1=completely wrong, 5=fully correct)
    Does the reply correctly address the customer's stated problem?

  groundedness  (1=no grounding, 5=fully grounded)
    Is the reply content supported by the historical evidence provided?

  helpfulness   (1=useless, 5=very helpful)
    Would this reply actually help the customer resolve their issue?

  tone          (1=inappropriate, 5=perfect)
    Is the tone empathetic, professional, and appropriate for customer support?

  hallucination (1=severe hallucination, 5=no hallucination)
    Did the AI invent policies, timelines, refunds, or promises NOT in the evidence?
    Score 5 if the reply is fully faithful, 1 if it contains fabricated specifics.

Return ONLY a valid JSON object (no markdown, no extra text) with keys:
  correctness, groundedness, helpfulness, tone, hallucination (all integers 1-5)
  reasoning: one sentence explaining your scores

---
CUSTOMER MESSAGE:
{message}

HISTORICAL EVIDENCE:
{evidence}

AGENT REPLY:
{reply}
"""

    def __init__(self, judge_model: str = "gemini-2.0-flash"):
        self.judge_model = judge_model
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

    def _heuristic_score(self, message: str, evidence: List[Dict[str, Any]], reply: str) -> Dict[str, Any]:
        """Simple heuristic fallback scoring when LLM is unavailable."""
        reply_lower = reply.lower()

        # Tone: check for empathy markers
        empathy_markers = ["sorry", "understand", "happy to help", "appreciate", "thank"]
        tone = min(3 + sum(1 for m in empathy_markers if m in reply_lower), 5)

        # Helpfulness: check for actionable content
        action_markers = ["dm", "direct message", "settings", "restart", "update", "check"]
        helpfulness = min(2 + sum(1 for m in action_markers if m in reply_lower), 5)

        # Groundedness: if evidence is present and reply references context
        groundedness = 3 if evidence else 2

        # Hallucination: penalise if reply contains specific promises
        hallucination_risk = ["will refund", "guarantee", "within 24 hours", "within 48 hours",
                              "100%", "definitely", "we promise"]
        hallucination = 5 - sum(1 for h in hallucination_risk if h in reply_lower)
        hallucination = max(hallucination, 2)

        # Correctness: moderate default
        correctness = 3

        return {
            "correctness": correctness,
            "groundedness": groundedness,
            "helpfulness": helpfulness,
            "tone": tone,
            "hallucination": hallucination,
            "reasoning": "Heuristic scoring (no LLM judge available — set GEMINI_API_KEY)",
        }

    def evaluate_reply(
        self,
        message: str,
        evidence: List[Dict[str, Any]],
        reply: str,
    ) -> Dict[str, Any]:
        """Score a generated reply on 5 quality dimensions (1–5 scale).

        Args:
            message: Original customer query.
            evidence: Retrieved historical conversation examples used for generation.
            reply: Agent-generated support reply.

        Returns:
            Dict with integer scores (1–5) per dimension plus reasoning string.
        """
        if self._client is None:
            return self._heuristic_score(message, evidence, reply)

        evidence_text = "\n\n".join(
            f"Case {i+1}: Customer: {ex.get('initial_query','')[:200]}\n"
            f"  AppleSupport: {ex.get('first_support_reply','')[:200]}"
            for i, ex in enumerate(evidence[:3])
        ) or "No evidence retrieved."

        prompt = self.JUDGE_PROMPT.format(
            message=message,
            evidence=evidence_text,
            reply=reply,
        )

        try:
            chat = self._client.chats.create(model=self.judge_model)
            response = chat.send_message(prompt)
            raw = response.text.strip()
            raw = re.sub(r"^```json?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            result = json.loads(raw)

            # Clamp all scores to 1–5
            for dim in ["correctness", "groundedness", "helpfulness", "tone", "hallucination"]:
                result[dim] = max(1, min(5, int(result.get(dim, 3))))

            result.setdefault("reasoning", "")
            return result

        except Exception as e:
            fallback = self._heuristic_score(message, evidence, reply)
            fallback["reasoning"] = f"LLM judge error ({type(e).__name__}): {fallback['reasoning']}"
            return fallback

    def evaluate_batch(
        self,
        records: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Evaluate a list of agent outputs. Each record needs: message, evidence, reply.

        Args:
            records: List of agent output dicts.
            verbose: Print progress if True.

        Returns:
            List of records with judge scores appended.
        """
        results = []
        for i, rec in enumerate(records):
            if verbose and (i % 10 == 0):
                print(f"  Judging {i+1}/{len(records)}...")
            scores = self.evaluate_reply(
                message=rec.get("message", ""),
                evidence=rec.get("evidence", []),
                reply=rec.get("reply", ""),
            )
            results.append({**rec, **{f"judge_{k}": v for k, v in scores.items()}})
        return results
