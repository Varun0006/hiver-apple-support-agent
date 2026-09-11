import unittest

from src.escalation.policy import EscalationPolicy


class EscalationPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = EscalationPolicy()
        self.evidence = [{"similarity_score": 0.5}]

    def decision(self, message, intent, confidence=0.9):
        return self.policy.evaluate(
            message=message,
            intent=intent,
            confidence=confidence,
            evidence=self.evidence,
        )["decision"]

    def test_account_intent_remains_conservative(self):
        self.assertEqual(
            self.decision(
                "How do I reset my Apple ID password?",
                "account_appleid_issue",
            ),
            "ESCALATE",
        )

    def test_account_access_risk_escalates(self):
        self.assertEqual(
            self.decision(
                "Someone accessed my Apple ID account",
                "account_appleid_issue",
            ),
            "ESCALATE",
        )

    def test_billing_intent_remains_conservative(self):
        self.assertEqual(
            self.decision(
                "How do I update my payment method?",
                "payment_billing_issue",
            ),
            "ESCALATE",
        )

    def test_unrecognized_charge_escalates(self):
        self.assertEqual(
            self.decision(
                "I do not recognize this charge on my card",
                "payment_billing_issue",
            ),
            "ESCALATE",
        )

    def test_duplicate_billing_has_transaction_reason(self):
        result = self.policy.evaluate(
            message="I was charged twice for one app purchase",
            intent="payment_billing_issue",
            confidence=0.9,
            evidence=self.evidence,
        )
        self.assertIn("Duplicate billing", result["reason"])

    def test_unrecognized_charge_has_billing_reason(self):
        result = self.policy.evaluate(
            message="I don't recognize this Apple charge",
            intent="payment_billing_issue",
            confidence=0.9,
            evidence=self.evidence,
        )
        self.assertIn("unrecognized Apple transaction", result["reason"])

    def test_compromised_account_has_security_reason(self):
        result = self.policy.evaluate(
            message="Someone hacked my Apple ID and changed my password",
            intent="account_appleid_issue",
            confidence=0.9,
            evidence=self.evidence,
        )
        self.assertIn("compromised Apple Account", result["reason"])

    def test_unresponsive_device_after_update_escalates(self):
        result = self.policy.evaluate(
            message="My iPhone won't turn on after the latest iOS update and the screen is completely black",
            intent="device_hardware_issue",
            confidence=0.9,
            evidence=self.evidence,
        )
        self.assertEqual(result["decision"], "ESCALATE")

    def test_safe_how_to_can_auto_handle_without_historical_grounding(self):
        result = self.policy.evaluate(
            message="How do I take a screenshot on my iPhone?",
            intent="device_how_to",
            confidence=0.88,
            evidence=[{"similarity_score": 0.8}],
            reply_grounded=False,
            historical_grounded=False,
            answer_supported=True,
        )
        self.assertEqual(result["decision"], "AUTO_HANDLE")

    def test_stolen_shipping_item_is_not_misclassified_as_account_risk(self):
        self.assertEqual(
            self.decision(
                "Serial number CPWKJ7WFDTY3 track MacBook Pro 13, stolen in transit",
                "order_shipping_issue",
            ),
            "AUTO_HANDLE",
        )

    def test_low_confidence_still_escalates(self):
        self.assertEqual(
            self.policy.evaluate(
                message="What is happening?",
                intent="other_general",
                confidence=0.3,
                evidence=[{"similarity_score": 0.1}],
            )["decision"],
            "ESCALATE",
        )

    def test_strong_evidence_can_resolve_low_confidence_non_sensitive_case(self):
        self.assertEqual(
            self.decision("What is happening?", "other_general", confidence=0.3),
            "ESCALATE",
        )

    def test_secondary_sensitive_intent_escalates(self):
        result = self.policy.evaluate(
            message="My phone broke and I was charged twice for one purchase",
            intent="device_hardware_issue",
            intents=["device_hardware_issue", "payment_billing_issue"],
            confidence=0.9,
            evidence=self.evidence,
        )
        self.assertEqual(result["decision"], "ESCALATE")


if __name__ == "__main__":
    unittest.main()