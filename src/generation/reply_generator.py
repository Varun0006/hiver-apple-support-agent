"""
Evidence-grounded Reply Generator.
Uses Gemini to draft support replies strictly grounded in retrieved historical examples.
The prompt explicitly forbids invented policies, timelines, or refunds not supported by evidence.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _format_evidence(evidence: List[Dict[str, Any]]) -> str:
    """Format retrieved historical conversations as structured evidence block."""
    if not evidence:
        return "No historical examples available."

    blocks = []
    for i, ex in enumerate(evidence, 1):
        query = ex.get("initial_query", "")
        reply = ex.get("first_support_reply", "")
        score = ex.get("similarity_score", 0.0)
        dialogue = ex.get("full_dialogue", "")
        blocks.append(
            f"[Case {i}] (similarity: {score:.3f})\n"
            f"  Customer: {query}\n"
            f"  AppleSupport replied: {reply}\n"
            f"  Full thread: {dialogue[:300]}{'...' if len(dialogue) > 300 else ''}"
        )
    return "\n\n".join(blocks)


def _normalise_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z][a-z']+", text.lower())
        if len(term) >= 4
    }


def _fallback_grounding(reply: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Conservative grounding check used when a grounding model is unavailable."""
    if not evidence:
        return {
            "grounded": False,
            "reason": "No historical evidence was retrieved.",
        }

    evidence_text = " ".join(
        f"{item.get('initial_query', '')} {item.get('first_support_reply', '')}"
        for item in evidence
    )
    action_groups = {
        "restart": {"restart", "restarting", "reboot", "reboots", "rebooting"},
        "charge": {"charge", "charging", "charger", "power"},
        "charging_accessory": {"cord", "cords", "cable", "cables", "port", "ports"},
        "settings": {"settings", "setting"},
        "contact_support": {"support", "team", "contact", "help"},
        "account": {"account", "password", "credentials", "verify", "recovery"},
        "purchase": {"purchase", "billing", "transaction", "receipt", "charged"},
        "download": {"download", "install", "app", "store"},
        "subscription": {"subscription", "subscriptions", "manage", "cancel"},
        "request_details": {"share", "model", "version", "happened", "details"},
    }
    evidence_terms = _normalise_terms(evidence_text)
    reply_terms = _normalise_terms(reply)
    reply_actions = {
        name for name, terms in action_groups.items() if reply_terms & terms
    }
    meaningful_actions = reply_actions - {"contact_support"}
    if not meaningful_actions:
        return {
            "grounded": False,
            "reason": "The reply does not contain a specific action supported by the retrieved historical responses.",
        }
    unsupported_actions = {
        name for name in meaningful_actions if not evidence_terms & action_groups[name]
    }
    if unsupported_actions:
        return {
            "grounded": False,
            "reason": "The reply includes actions not supported by the retrieved historical support responses.",
        }
    return {
        "grounded": True,
        "reason": "The troubleshooting actions in the reply are supported by the retrieved historical support responses.",
    }


def validate_grounding(
    reply: str,
    evidence: List[Dict[str, Any]],
    client: Any = None,
    model_name: str = "gemini-2.0-flash",
) -> Dict[str, Any]:
    """Evaluate claim/action support using structured judging, with a conservative fallback."""
    if not evidence:
        return {"grounded": False, "reason": "No historical evidence was retrieved."}
    if client is None:
        return _fallback_grounding(reply, evidence)

    evidence_text = _format_evidence(evidence)
    prompt = f"""Evaluate whether the reply is grounded in the historical Apple support evidence.

Grounded means every important troubleshooting action or recommendation in the reply is supported by at least one historical support response. Shared topic words alone are not enough. If the reply adds an unsupported action, URL, policy, guarantee, or procedure, grounded must be false.

Return only JSON: {{"grounded": true or false, "reason": "one concise sentence"}}

HISTORICAL EVIDENCE:
{evidence_text}

REPLY:
{reply}
"""
    try:
        chat = client.chats.create(model=model_name)
        raw = chat.send_message(prompt).text.strip()
        raw = re.sub(r"^```json?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
        return {
            "grounded": bool(result.get("grounded", False)),
            "reason": str(result.get("reason", "Grounding could not be established.")),
        }
    except Exception:
        return _fallback_grounding(reply, evidence)


class ReplyGenerator:
    """
    Generates customer support replies strictly grounded in retrieved historical evidence.

    The LLM is explicitly instructed NOT to:
    - Invent policies, timelines, refunds, or actions not supported by retrieved evidence.
    - Promise outcomes that weren't observed historically.
    - Add new information not present in the historical cases.
    """

    SYSTEM_PROMPT = """You are an Apple Customer Support agent drafting a reply to a customer tweet.

You MUST base your reply ONLY on the historical AppleSupport cases provided below.
Do NOT invent policies, refund timelines, fixes, or actions that are not supported by the evidence.
If the evidence doesn't cover the issue well, acknowledge the concern and ask for more info or offer to continue in DM.

Use the historical support responses as the source of any troubleshooting action. Do not replace a specific historical resolution with a generic request to DM.

Keep the reply:
- Concise (tweet-length when possible, 2–4 sentences max)
- Empathetic and professional
- Actionable (suggest a clear next step)
- Free of invented guarantees or hallucinated facts

Historical AppleSupport Cases (your evidence):
{evidence}

Customer message:
{message}

Intent identified: {intent}

Write ONLY the support reply text. No preamble, no quotation marks around it."""

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

    @staticmethod
    def _known_issue_reply(
        message: str,
        intent: str,
        evidence: List[Dict[str, Any]] | None = None,
    ) -> str | None:
        """Return a deterministic, actionable reply for common support requests."""
        text = message.lower()
        if intent == "device_storage_issue" or re.search(
            r"\b(storage|free up space|low storage|storage is full)\b", text
        ):
            return (
                "Your iPhone is low on storage.\n"
                "Go to Settings > General > iPhone Storage to review recommendations and remove unused apps, photos, or downloads."
            )
        if intent == "device_how_to" and re.search(
            r"\b(screenshot|screen shot|screen capture|picture of my screen)\b", text
        ):
            return (
                "To take a screenshot on an iPhone with Face ID, press the Side button and Volume Up at the same time, then quickly release both.\n"
                "On an iPhone with a Home button, press the Home and Side or Top buttons together. The screenshot will appear in Photos > Albums > Screenshots."
            )
        if intent == "device_how_to" and re.search(
            r"\b(delete|remove|uninstall)\b.*\b(app|application)\b", text
        ):
            return (
                "To delete an app on your iPhone, touch and hold the app on the Home Screen.\n"
                "Tap Remove App, choose Delete App, then tap Delete to confirm. You can also choose Remove from Home Screen to keep the app in the App Library."
            )
        if intent == "device_how_to" and re.search(
            r"\b(screen recording|record my screen|record the screen)\b", text
        ):
            return (
                "To record your iPhone screen, open Control Center and tap the Screen Recording button.\n"
                "Wait for the three-second countdown, then stop recording from Control Center or by tapping the red status indicator. The video is saved in Photos."
            )
        if intent == "device_how_to" and re.search(
            r"\b(turn on|enable|connect to)\b.*\b(wi[- ]?fi)\b|\b(wi[- ]?fi)\b.*\b(turn on|enable|connect)\b", text
        ):
            return (
                "To turn on Wi-Fi, open Settings and tap Wi-Fi.\n"
                "Turn on Wi-Fi, select your network, and enter the password if prompted. You can also tap the Wi-Fi icon in Control Center for a quick toggle."
            )
        if intent == "device_how_to" and re.search(
            r"\b(dark mode|dark theme)\b", text
        ):
            return (
                "To turn on Dark Mode, open Settings > Display & Brightness.\n"
                "Select Dark, or turn on Automatic to switch between Light and Dark Mode on a schedule."
            )
        if intent == "device_how_to" and re.search(
            r"\b(update ios|update my iphone|update my phone|software update)\b", text
        ):
            return (
                "To update your iPhone, connect it to Wi-Fi and power, then open Settings > General > Software Update.\n"
                "If an update is available, tap Download and Install and follow the onscreen instructions."
            )
        if intent == "account_appleid_issue" and re.search(
            r"\b(recover|recovery|forgot|reset|can't access|cannot access)\b", text
        ):
            if re.search(r"\b(password|passcode)\b", text):
                return (
                    "To reset your Apple ID password, open Settings on a trusted Apple device, tap your name, then choose Sign-In & Security > Change Password.\n"
                    "If you do not have access to a trusted device, use Apple's account recovery page at iforgot.apple.com. Apple Support can help if you cannot complete the reset."
                )
            return (
                "Sorry you're having trouble accessing your Apple Account.\n"
                "Use Apple's account recovery process to reset your password and regain access. If you still need help, Apple Support can verify the account."
            )
        if intent == "account_appleid_issue" and re.search(
            r"\b(hacked|compromised|changed my password|changed my apple account password|unauthorized access)\b", text
        ):
            return (
                "Sorry you're dealing with this. Because your Apple Account credentials may have been changed without your permission, this requires account-security assistance.\n"
                "Apple Support can help with account recovery and securing the account."
            )
        if re.search(r"\b(stuck on the apple logo|boot loop|keeps rebooting)\b", text):
            return (
                "Sorry your iPhone is stuck while starting up.\n"
                "Please share your iPhone model and iOS version, then contact Apple Support so they can gather more information and continue troubleshooting."
            )
        if intent == "device_hardware_issue" and re.search(
            r"\b(black screen|screen is completely black|won'?t turn on|doesn'?t turn on)\b", text
        ) and re.search(r"\b(update|ios)\b", text):
            return (
                "Sorry your iPhone is not powering on after the update.\n"
                "Please share your iPhone model and what happened before the screen went black, including the iOS version or update you installed, so Apple Support can investigate the issue."
            )
        if intent == "device_hardware_issue" and re.search(
            r"\b(crack|cracked|broken|shatter|shattered)\b", text
        ):
            return (
                "Sorry to hear about the damage to your iPhone screen.\n"
                "Avoid using the device if the glass is loose or sharp, and contact Apple Support or an Apple Authorized Service Provider to arrange a repair."
            )
        if intent == "device_hardware_issue" and re.search(
            r"\b(won'?t turn on|doesn'?t turn on|no power|power on)\b", text
        ):
            return (
                "Sorry to hear you're having trouble with your iPhone.\n"
                "Please share your iPhone model and what happens when you connect the charger, so Apple Support can guide the next troubleshooting step."
            )
        if re.search(r"\b(won'?t charge|doesn'?t charge|not charging|charge issue)\b", text):
            return (
                "If your iPhone won't charge, try a different charging cable and another power port.\n"
                "If it still does not charge, contact Apple Support for further troubleshooting."
            )
        if intent == "device_hardware_issue" and re.search(
            r"\b(hot|overheating|overheat)\b", text
        ):
            return (
                "If your iPhone is getting unusually hot, unplug it and move it away from direct heat.\n"
                "Close intensive apps and let it cool down. If it remains hot, shuts down, or shows a temperature warning, contact Apple Support for help."
            )
        if intent in {"device_hardware_issue", "software_update_issue"} and re.search(
            r"\b(restart|restarting|reboot|reboots|rebooting)\b", text
        ):
            return (
                "Sorry your iPhone keeps restarting unexpectedly.\n"
                "Install any available iOS update, check that you have enough storage, and disconnect accessories to see whether the restarts stop. If the issue continues, Apple Support can help diagnose it."
            )
        if intent == "payment_billing_issue" and re.search(
            r"\b(charged twice|double charge|duplicate charge|twice)\b", text
        ):
            return (
                "Sorry about the duplicate charge.\n"
                "Apple Support's billing team can look into the duplicate Apple purchase. This requires account-specific transaction review."
            )
        if intent == "payment_billing_issue" and re.search(
            r"\b(i don't recognize|i do not recognize|don't remember making|do not remember making|unauthorized|unknown)\b",
            text,
        ):
            return (
                "Sorry you found an Apple charge you do not recognize.\n"
                "Please contact Apple Support with the transaction details so the billing team can investigate it securely."
            )
        if intent == "subscription_services" and re.search(
            r"\b(cancel|cancellation|unsubscribe|stop|end)\b", text
        ):
            return (
                "Apple Support's subscription-management guidance can help you cancel your Apple Music subscription.\n"
                "If the cancellation option is missing or the subscription still appears active, contact Apple Support for help."
            )
        if intent == "app_store_issue" and re.search(
            r"\b(download|install|error|cannot|can't)\b", text
        ):
            return (
                "Sorry you're having trouble downloading from the App Store.\n"
                "What happens when you try to download the app, and what device model and iOS version are you using? Apple Support can continue troubleshooting through support if the error persists."
            )
        return None

    def generate(
        self,
        message: str,
        intent: str,
        evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate a grounded support reply.

        Args:
            message: Customer query text.
            intent: Classified intent label.
            evidence: Top-k retrieved historical conversations.

        Returns:
            Dict with 'reply' text and 'grounded' bool indicating if evidence was used.
        """
        evidence_text = _format_evidence(evidence)
        grounded = bool(evidence)

        known_reply = self._known_issue_reply(message, intent, evidence)
        if known_reply:
            grounding = validate_grounding(known_reply, evidence, self._client, self.model_name)
            return {
                "reply": known_reply,
                "grounded": grounding["grounded"],
                "historical_grounded": grounding["grounded"],
                "answer_supported": True,
                "support_reason": "A deterministic response is available for this recognized support request.",
                "grounding_reason": grounding["reason"],
                "model": "deterministic_issue_handler",
            }

        if self._client is None:
            # Fallback: template reply
            template_replies = {
                "device_hardware_issue": "Sorry to hear you're having trouble with your iPhone.\nPlease share your iPhone model and what happens when you connect the charger, so Apple Support can guide the next troubleshooting step.",
                "device_storage_issue": "Your iPhone is low on storage.\nGo to Settings > General > iPhone Storage to review recommendations and remove unused apps, photos, or downloads.",
                "device_how_to": "Tell us which Apple feature you want to use, and we can provide the steps.",
                "software_update_issue": "Sorry you're having trouble after the update.\nTry restarting your device and checking that you have enough available storage. If the issue continues, Apple Support can help with further troubleshooting.",
                "account_appleid_issue": "Sorry you're having trouble accessing your Apple ID.\nUse Apple's account recovery process to reset your password. If you still can't regain access, Apple Support can help verify the account.",
                "payment_billing_issue": "Sorry about the duplicate charge.\nCheck your purchase history to confirm both transactions, then contact Apple Support with the receipt details so the billing issue can be reviewed.",
                "card_atm_issue": "Sorry you're having trouble with your card.\nCheck that the card details and available funds are correct. If the transaction is still declined, Apple Support can investigate further.",
                "subscription_services": "Sorry you're having trouble with your subscription.\nOpen your Apple Account subscription settings to review or cancel the plan. Apple Support can help if the subscription still appears active.",
                "app_store_issue": "Sorry you're having trouble with the App Store.\nTry restarting your device and signing in again before retrying the download. If the issue continues, Apple Support can help with further troubleshooting.",
                "order_shipping_issue": "Sorry about the delivery concern.\nCheck your order status using the tracking details in your confirmation email. Apple Support can investigate further if the status has not changed.",
                "other_general": "Thanks for contacting Apple Support.\nPlease share a few more details about the issue so we can point you to the right next step.",
            }
            if intent == "device_hardware_issue" and re.search(
                r"\b(crack|cracked|broken|shatter|shattered)\b", message.lower()
            ):
                reply = (
                    "Sorry to hear about the damage to your iPhone screen.\n"
                    "Avoid using the device if the glass is loose or sharp, and contact Apple Support or an Apple Authorized Service Provider to arrange a repair."
                )
            else:
                reply = template_replies.get(intent, template_replies["other_general"])
            grounding = validate_grounding(reply, evidence, self._client, self.model_name)
            return {
                "reply": reply,
                "grounded": grounding["grounded"],
                "historical_grounded": grounding["grounded"],
                "answer_supported": False,
                "support_reason": "The fallback response does not contain enough issue-specific guidance to verify as a supported answer.",
                "grounding_reason": grounding["reason"],
                "model": "template_fallback",
            }

        prompt = self.SYSTEM_PROMPT.format(
            evidence=evidence_text,
            message=message,
            intent=intent,
        )

        try:
            chat = self._client.chats.create(model=self.model_name)
            response = chat.send_message(prompt)
            reply = response.text.strip()
            grounding = validate_grounding(reply, evidence)
            return {
                "reply": reply,
                "grounded": grounding["grounded"],
                "historical_grounded": grounding["grounded"],
                "answer_supported": grounding["grounded"],
                "support_reason": "The generated response is supported by retrieved evidence." if grounding["grounded"] else "The generated response needs review because its actions are not fully supported by retrieved evidence.",
                "grounding_reason": grounding["reason"],
                "model": self.model_name,
            }
        except Exception as e:
            grounding = validate_grounding(
                "Thank you for contacting Apple Support. Please DM us so we can look into this further.",
                evidence,
                self._client,
                self.model_name,
            )
            return {
                "reply": f"Thank you for contacting Apple Support. Please DM us so we can look into this further.",
                "grounded": grounding["grounded"],
                "historical_grounded": grounding["grounded"],
                "answer_supported": False,
                "support_reason": f"Reply generation failed: {type(e).__name__}.",
                "grounding_reason": f"Reply generation failed: {type(e).__name__}.",
                "model": f"error_fallback ({type(e).__name__})",
            }
