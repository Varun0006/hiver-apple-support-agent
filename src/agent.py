"""
Hiver Support Agent — Unified Pipeline.
Orchestrates: Intent Classification → Historical Retrieval → Reply Generation → Escalation Decision.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.intent.classifier import IntentClassifier
from src.retrieval.retriever import HistoricalRetriever
from src.generation.reply_generator import ReplyGenerator
from src.escalation.policy import EscalationPolicy


class SupportAgent:
    """
    End-to-end Apple Support Agent pipeline.

    Flow:
        Customer message
            │
            ▼
        IntentClassifier  ──────────────────────────────────────────────────►  intent, confidence
            │
            ▼
        HistoricalRetriever  ──────────────────────────────────────────────►  top-k evidence cases
            │
            ├──────────────────────────────────────────────────────────────►  ReplyGenerator
            │                                                                       │
            │                                                                       ▼
            │                                                                  draft reply
            │
            └──────────────────────────────────────────────────────────────►  EscalationPolicy
                                                                                    │
                                                                                    ▼
                                                                          AUTO_HANDLE / ESCALATE
    """

    def __init__(
        self,
        llm_model: str = "gemini-2.0-flash",
        confidence_threshold: float = 0.60,
        retrieval_top_k: int = 5,
        retrieval_index_path: Optional[str] = None,
        conversations_csv: Optional[str] = None,
        retrieval_verbose: bool = True,
    ):
        self.classifier = IntentClassifier(model_name=llm_model)
        self.retriever = HistoricalRetriever(
            index_path=retrieval_index_path,
            conversations_csv=conversations_csv,
            verbose=retrieval_verbose,
        )
        self.generator = ReplyGenerator(model_name=llm_model)
        self.escalation = EscalationPolicy(confidence_threshold=confidence_threshold)
        self.top_k = retrieval_top_k
        self._index_ready = False

    def warm_up(self) -> None:
        """Pre-load retrieval index. Call once before processing many messages."""
        self.retriever.ensure_loaded()
        self._index_ready = True

    def run(self, message: str) -> Dict[str, Any]:
        """Run the full support agent pipeline for a single customer message.

        Args:
            message: Raw customer tweet / support message.

        Returns:
            Dict with:
                message        – original input
                intent         – predicted intent label
                confidence     – classifier confidence [0-1]
                reasoning      – classifier reasoning
                evidence       – list of retrieved historical conversation examples
                reply          – generated support reply
                reply_grounded – True if evidence was used in generation
                escalation     – dict with 'decision' and 'reason'
        """
        if not self._index_ready:
            self.warm_up()

        # Step 1: Classify intent
        classification = self.classifier.classify(message)
        intent = classification["intent"]
        intents = classification.get("intents", [intent])
        confidence = classification["confidence"]
        reasoning = classification.get("reasoning", "")

        # Step 2: Retrieve similar historical cases
        evidence = self.retriever.retrieve(message, top_k=self.top_k)

        # Step 3: Generate grounded reply
        generation = self.generator.generate(message, intent, evidence)
        reply = generation["reply"]
        reply_grounded = generation.get("grounded", False)
        historical_grounded = generation.get("historical_grounded", reply_grounded)
        answer_supported = generation.get("answer_supported", False)
        support_reason = generation.get("support_reason", "")
        grounding_reason = generation.get("grounding_reason", "")

        # Step 4: Escalation decision
        escalation_result = self.escalation.evaluate(
            message=message,
            intent=intent,
            intents=intents,
            confidence=confidence,
            evidence=evidence,
            reply_grounded=reply_grounded,
            historical_grounded=historical_grounded,
            answer_supported=answer_supported,
            grounding_reason=grounding_reason,
        )

        return {
            "message": message,
            "intent": intent,
            "intents": intents,
            "confidence": confidence,
            "reasoning": reasoning,
            "evidence": evidence,
            "reply": reply,
            "reply_grounded": reply_grounded,
            "historical_grounded": historical_grounded,
            "answer_supported": answer_supported,
            "support_reason": support_reason,
            "grounding_reason": grounding_reason,
            "generation_model": generation.get("model", ""),
            "escalation": escalation_result,
        }


def run_agent(message: str, **kwargs) -> Dict[str, Any]:
    """Convenience function — creates a fresh agent and runs a single message."""
    agent = SupportAgent(**kwargs)
    return agent.run(message)
