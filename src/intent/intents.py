"""
Intent Taxonomy Definitions and Metadata for AppleSupport.
"""

from typing import Dict, Any, List

INTENT_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "device_hardware_issue": {
        "name": "device_hardware_issue",
        "description": "Physical hardware issues: battery drain, screen glitches, charging port failures, speaker/mic problems, overheating, or camera defects.",
        "keywords": ["battery", "screen", "charge", "charger", "charging", "heat", "hot", "overheating", "overheat", "speaker", "mic", "camera", "button", "display", "drain", "turn on", "turned on", "power on", "won't turn on", "wont turn on", "restart", "restarting", "reboot", "reboots", "rebooting"],
        "default_action": "AUTO_HANDLE",
    },
    "software_update_issue": {
        "name": "software_update_issue",
        "description": "iOS update errors, freezing, lagging, system crashes, storage full during update, or bugs after updating.",
        "keywords": ["update", "ios", "version", "freeze", "freezing", "lag", "lagging", "slow", "stuck", "reboot", "loop", "crash", "downloading"],
        "default_action": "AUTO_HANDLE",
    },
    "device_storage_issue": {
        "name": "device_storage_issue",
        "description": "iPhone or iPad storage is full, low-storage warnings, and freeing space on the device.",
        "keywords": ["storage", "storage is full", "space", "free up space", "low storage", "full"],
        "default_action": "AUTO_HANDLE",
    },
    "device_how_to": {
        "name": "device_how_to",
        "description": "How-to questions about using Apple device features, including screenshots, settings, and common device actions.",
        "keywords": ["screenshot", "screen shot", "screen capture", "how to", "how do i", "take a picture of my screen", "delete an app", "delete a app", "delete app", "remove app", "uninstall app", "screen recording", "record my screen", "turn on wifi", "enable wifi", "dark mode", "update ios", "update my iphone"],
        "default_action": "AUTO_HANDLE",
    },
    "account_appleid_issue": {
        "name": "account_appleid_issue",
        "description": "Apple ID lockouts, password resets, two-factor authentication, iCloud account sync, or compromised accounts.",
        "keywords": ["appleid", "apple id", "apple account", "recover", "recovery", "password", "icloud", "locked", "disabled", "security", "verification", "code", "2fa", "sign in", "login"],
        "default_action": "ESCALATE",  # Security sensitive
    },
    "payment_billing_issue": {
        "name": "payment_billing_issue",
        "description": "Unrecognized App Store charges, double billing, refund requests, payment method declined, or invoice questions.",
        "keywords": ["charge", "charged", "billing", "payment", "refund", "card", "bank", "money", "receipt", "purchase", "declined", "invoice"],
        "default_action": "ESCALATE",  # Fraud/financial sensitive
    },
    "card_atm_issue": {
        "name": "card_atm_issue",
        "description": "Debit or credit card declines, ATM cash withdrawal problems, or card access issues that require account-specific investigation.",
        "keywords": ["debit card", "credit card", "debit", "credit", "atm", "cash withdrawal", "cash machine", "cash", "card declined"],
        "default_action": "ESCALATE",  # Account-specific financial issue
    },
    "subscription_services": {
        "name": "subscription_services",
        "description": "Subscriptions for Apple Music, iCloud+, Apple TV+, Apple Arcade cancellation, renewal, or free trial inquiries.",
        "keywords": ["subscription", "music", "tv+", "arcade", "icloud+", "cancel", "trial", "plan", "membership", "renew"],
        "default_action": "AUTO_HANDLE",
    },
    "app_store_issue": {
        "name": "app_store_issue",
        "description": "App Store download errors, app crashing, app updates failing, or third-party app compatibility.",
        "keywords": ["app", "apps", "store", "download", "downloading", "install", "installing", "update app", "whatsapp", "instagram", "twitter"],
        "default_action": "AUTO_HANDLE",
    },
    "order_shipping_issue": {
        "name": "order_shipping_issue",
        "description": "Apple Store online orders, delivery delays, shipment tracking, trade-in status, or in-store pickup.",
        "keywords": ["order", "shipping", "delivery", "track", "tracking", "pickup", "store", "trade-in", "delivery date", "shipment", "arriving"],
        "default_action": "AUTO_HANDLE",
    },
    "other_general": {
        "name": "other_general",
        "description": "General feedback, feature requests, store hours, or unclassified support queries.",
        "keywords": ["help", "thanks", "hello", "hi", "question", "feedback", "feature", "suggestion"],
        "default_action": "AUTO_HANDLE",
    },
}

INTENT_NAMES: List[str] = list(INTENT_TAXONOMY.keys())
