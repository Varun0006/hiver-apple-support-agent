"""
Main entrypoint for the Hiver AppleSupport AI Agent.
Warms up the retrieval index and runs a quick smoke test.
"""

import sys
import json
import argparse
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agent import SupportAgent


def print_result(result):
    intent_names = {
        "device_hardware_issue": "Device Troubleshooting",
        "device_storage_issue": "Device Storage",
        "device_how_to": "Device How-To",
        "software_update_issue": "Software Update",
        "account_appleid_issue": "Apple ID and Account Access",
        "payment_billing_issue": "Billing and Purchases",
        "card_atm_issue": "Card and ATM Issues",
        "subscription_services": "Subscriptions",
        "app_store_issue": "App Store",
        "order_shipping_issue": "Orders and Delivery",
        "other_general": "General Support",
    }
    decision = result["escalation"]["decision"]
    reason = result["escalation"]["reason"]
    display_intent = intent_names.get(result["intent"], result["intent"])
    if result["intent"] == "payment_billing_issue" and re.search(
        r"\b(charged twice|duplicate charge|double charge|charged me twice)\b",
        result["message"].lower(),
    ):
        display_intent = "Apple Purchase Billing - Duplicate Charge"
    if result["intent"] == "device_hardware_issue" and decision == "AUTO_HANDLE":
        reason = "Basic troubleshooting can be provided without account-specific information."

    print("=" * 50)
    print("CUSTOMER MESSAGE")
    print(result["message"])
    print("\nINTENT")
    print(f"{display_intent} (confidence={result['confidence']:.2f})")
    print("\nRETRIEVAL")
    print(f"Evidence count: {len(result['evidence'])}")
    top_similarity = max((item.get("similarity_score", 0.0) for item in result["evidence"]), default=0.0)
    print(f"Top similarity: {top_similarity:.3f}")
    print("\nDECISION")
    print(decision.replace("_", "-"))
    print("\nREASON")
    print(reason)
    print("\nREPLY")
    print(result["reply"])
    print("\nGROUNDING")
    print(f"Historical grounding: {result.get('historical_grounded', False)}")
    print("\nGROUNDING REASON")
    print(result.get("grounding_reason", "Not available."))
    print("\nANSWER SUPPORTED")
    print(str(result.get("answer_supported", False)))
    print("\nSUPPORT REASON")
    print(result.get("support_reason", "Not available."))
    print("\nEVIDENCE")
    if result["evidence"]:
        for index, item in enumerate(result["evidence"][:3], 1):
            print(f"{index}. Customer: {item.get('initial_query', '')[:180]}")
            print(f"   Support: {item.get('first_support_reply', '')[:180]}")
    else:
        print("No historical evidence retrieved.")
    print("=" * 50)


def run_interactive(agent):
    print("Type a customer message and press Enter. Type 'exit' to quit.")
    while True:
        try:
            message = input("\nCustomer: ").strip()
        except EOFError:
            print()
            break
        if message.lower() in {"exit", "quit"}:
            break
        if not message:
            continue
        print()
        print_result(agent.run(message))


def run_smoke_test(agent):
    print("=" * 52)
    print("  HIVER APPLESUPPORT AGENT — PIPELINE SMOKE TEST")
    print("=" * 52)

    test_messages = [
        "My iPhone won't turn on after the latest iOS update. The screen is completely black.",
        "I was charged twice for an app I only bought once. This is not acceptable!",
        "I don't recognize this Apple charge.",
        "How do I cancel my Apple Music subscription?",
        "Someone hacked my Apple ID and changed my password.",
        "Someone changed my Apple Account password.",
        "The App Store keeps showing an error when I try to download any app.",
        "My iPhone won't charge. What should I try?",
        "How do I take a screenshot on my iPhone?",
        "How do I delete an app?",
        "I have a problem with my phone.",
        "Something is wrong with my account.",
        "It stopped working.",
        "My iPhone is stuck on the Apple logo.",
    ]

    for msg in test_messages:
        print("\n" + "-" * 52)
        print(f"Message : {msg}")
        result = agent.run(msg)
        print(f"Intent  : {result['intent']} (confidence={result['confidence']:.2f})")
        print(f"Decision: {result['escalation']['decision']} — {result['escalation']['reason']}")
        top_similarity = max(
            (item.get("similarity_score", 0.0) for item in result["evidence"]),
            default=0.0,
        )
        print(f"Reply   : {result['reply']}")
        print(f"Historical grounding: {result.get('historical_grounded', result['reply_grounded'])} — {result['grounding_reason']}")
        print(f"Answer supported: {result.get('answer_supported', False)} — {result.get('support_reason', '')}")
        print(f"Evidence: {len(result['evidence'])} case(s)")
        print(f"Top similarity: {top_similarity:.3f}")

    print("\n" + "=" * 52)
    print("Smoke test complete.")


def main():
    parser = argparse.ArgumentParser(description="Run the Apple Support AI agent.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--message", help="Answer one customer message.")
    mode.add_argument(
        "--interactive",
        action="store_true",
        help="Accept customer messages until 'exit' or EOF.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of historical evidence cases to retrieve (default: 3).",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    agent = SupportAgent(
        confidence_threshold=0.60,
        retrieval_top_k=args.top_k,
        retrieval_verbose=False,
    )
    agent.warm_up()

    if args.message:
        print_result(agent.run(args.message))
    elif args.interactive:
        run_interactive(agent)
    else:
        run_smoke_test(agent)


if __name__ == "__main__":
    main()
