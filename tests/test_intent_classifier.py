import unittest

from src.intent.classifier import _keyword_classify
from src.generation.reply_generator import ReplyGenerator, validate_grounding


class IntentClassifierTests(unittest.TestCase):
    def test_keyword_classifier_preserves_primary_intent(self):
        result = _keyword_classify("My battery is draining quickly")
        self.assertEqual(result["intent"], result["intents"][0])
        self.assertEqual(result["intents"], ["device_hardware_issue"])

    def test_keyword_classifier_detects_multiple_issues(self):
        result = _keyword_classify("My phone screen broke and I was charged twice")
        self.assertEqual(result["intent"], "device_hardware_issue")
        self.assertIn("payment_billing_issue", result["intents"])

    def test_unknown_message_has_single_fallback_intent(self):
        result = _keyword_classify("hello there")
        self.assertEqual(result["intents"], ["other_general"])

    def test_card_atm_message_has_dedicated_intent(self):
        result = _keyword_classify("My debit card was declined at the ATM")
        self.assertEqual(result["intent"], "card_atm_issue")

    def test_account_recovery_has_account_intent(self):
        result = _keyword_classify("How do I recover my Apple Account?")
        self.assertEqual(result["intent"], "account_appleid_issue")

    def test_storage_question_has_storage_intent(self):
        result = _keyword_classify("My iPhone storage is full")
        self.assertEqual(result["intent"], "device_storage_issue")

    def test_hot_device_has_hardware_intent(self):
        result = _keyword_classify("My iPhone is getting very hot")
        self.assertEqual(result["intent"], "device_hardware_issue")

    def test_restarting_device_has_hardware_intent(self):
        result = _keyword_classify("My iPhone keeps restarting by itself")
        self.assertEqual(result["intent"], "device_hardware_issue")

    def test_screenshot_question_has_how_to_intent(self):
        result = _keyword_classify("How do I take a screenshot on my phone?")
        self.assertEqual(result["intent"], "device_how_to")

    def test_delete_app_reply_includes_steps(self):
        reply = ReplyGenerator._known_issue_reply(
            "How do I delete a app?",
            "device_how_to",
        )
        self.assertIn("Remove App", reply)
        self.assertIn("Delete App", reply)

    def test_basic_how_to_replies_are_actionable(self):
        cases = {
            "How do I turn on Wi-Fi?": "Settings",
            "How do I record my screen?": "Control Center",
            "How do I turn on dark mode?": "Display & Brightness",
            "How do I update my iPhone?": "Software Update",
        }
        for message, expected in cases.items():
            reply = ReplyGenerator._known_issue_reply(message, "device_how_to")
            self.assertIn(expected, reply, message)

    def test_contact_only_reply_is_not_grounded(self):
        result = validate_grounding(
            "Please contact Apple Support so we can help.",
            [{"initial_query": "I have a problem", "first_support_reply": "Please contact Apple Support."}],
        )
        self.assertFalse(result["grounded"])

    def test_unsupported_troubleshooting_action_is_not_grounded(self):
        result = validate_grounding(
            "Restart your iPhone and reset network settings.",
            [{"initial_query": "My App Store is not working", "first_support_reply": "Please contact Apple Support."}],
        )
        self.assertFalse(result["grounded"])

    def test_unrecognized_apple_charge_is_billing_intent(self):
        result = _keyword_classify("I don't recognize this Apple charge")
        self.assertEqual(result["intent"], "payment_billing_issue")

    def test_unrecognized_charge_reply_is_not_duplicate_billing_reply(self):
        reply = ReplyGenerator._known_issue_reply(
            "I don't recognize this Apple charge",
            "payment_billing_issue",
        )
        self.assertIn("do not recognize", reply)
        self.assertNotIn("duplicate", reply.lower())

    def test_password_reset_reply_includes_steps(self):
        reply = ReplyGenerator._known_issue_reply(
            "I forgot my Apple ID password. How can I reset it?",
            "account_appleid_issue",
        )
        self.assertIn("Sign-In & Security", reply)
        self.assertIn("iforgot.apple.com", reply)


if __name__ == "__main__":
    unittest.main()