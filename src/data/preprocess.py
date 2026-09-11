"""
Preprocessing and text normalization utilities for Twitter support tweets.
"""

import html
import re


def clean_tweet_text(text: str) -> str:
    """Clean tweet text by unescaping HTML, normalizing whitespace, and preserving key content."""
    if not isinstance(text, str) or not text:
        return ""

    # Unescape HTML entities (&amp; -> &, &gt; -> >, &lt; -> <)
    cleaned = html.unescape(text)

    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def extract_customer_initial_query(text: str) -> str:
    """Extract initial customer query text by removing initial handle mentions if present."""
    cleaned = clean_tweet_text(text)
    # Remove leading handle mentions like "@AppleSupport " or "@105838 @AppleSupport "
    cleaned = re.sub(r"^(?:@\w+\s*)+", "", cleaned).strip()
    return cleaned if cleaned else text
