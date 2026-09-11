"""
Reconstruct multi-turn support threads from in_response_to_tweet_id and response_tweet_id chains.
"""

from typing import Any, Dict, List
import pandas as pd
from src.data.preprocess import clean_tweet_text, extract_customer_initial_query


def build_conversation_threads(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Reconstruct support conversations from parent-child tweet reply relationships.

    Args:
        df: Filtered DataFrame of tweets.

    Returns:
        List of structured conversation objects.
    """
    # Create fast lookup dict by tweet_id
    tweet_dict = {}
    for _, row in df.iterrows():
        t_id = str(row["tweet_id"]).split(".")[0]
        tweet_dict[t_id] = {
            "tweet_id": t_id,
            "author_id": str(row["author_id"]),
            "inbound": bool(row["inbound"]),
            "created_at": str(row["created_at"]),
            "text": clean_tweet_text(str(row["text"])),
            "in_response_to_tweet_id": (
                str(row["in_response_to_tweet_id"]).split(".")[0]
                if pd.notna(row["in_response_to_tweet_id"])
                else None
            ),
            "response_tweet_id": (
                str(row["response_tweet_id"]) if pd.notna(row["response_tweet_id"]) else None
            ),
        }

    # Map parent to children
    parent_to_children = {}
    for t_id, t_data in tweet_dict.items():
        p_id = t_data["in_response_to_tweet_id"]
        if p_id and p_id in tweet_dict:
            parent_to_children.setdefault(p_id, []).append(t_id)

    # Find root customer queries (inbound=True, no parent in dataset or no parent)
    root_tweet_ids = []
    for t_id, t_data in tweet_dict.items():
        if t_data["inbound"]:
            p_id = t_data["in_response_to_tweet_id"]
            if not p_id or p_id not in tweet_dict:
                root_tweet_ids.append(t_id)

    conversations = []

    for root_id in root_tweet_ids:
        # Trace thread sequentially
        thread_turns = []
        curr_id = root_id
        visited = set()

        while curr_id and curr_id in tweet_dict and curr_id not in visited:
            visited.add(curr_id)
            t_info = tweet_dict[curr_id]
            thread_turns.append(
                {
                    "turn_index": len(thread_turns),
                    "tweet_id": t_info["tweet_id"],
                    "author_id": t_info["author_id"],
                    "inbound": t_info["inbound"],
                    "text": t_info["text"],
                    "created_at": t_info["created_at"],
                }
            )

            # Move to next child turn
            children = parent_to_children.get(curr_id, [])
            if children:
                # Prefer AppleSupport response if available, else first child
                apple_children = [
                    c for c in children if tweet_dict[c]["author_id"].lower() == "applesupport"
                ]
                curr_id = apple_children[0] if apple_children else children[0]
            else:
                curr_id = None

        if len(thread_turns) >= 1:
            initial_query = extract_customer_initial_query(thread_turns[0]["text"])

            # Find first brand reply if present
            first_reply = None
            for turn in thread_turns[1:]:
                if turn["author_id"].lower() == "applesupport":
                    first_reply = turn["text"]
                    break

            conversations.append(
                {
                    "conversation_id": f"conv_{root_id}",
                    "root_tweet_id": root_id,
                    "customer_id": thread_turns[0]["author_id"],
                    "num_turns": len(thread_turns),
                    "initial_query": initial_query,
                    "first_support_reply": first_reply,
                    "has_support_reply": first_reply is not None,
                    "turns": thread_turns,
                }
            )

    return conversations


def conversations_to_dataframe(conversations: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten reconstructed conversations into a tabular DataFrame format for analysis/retrieval."""
    rows = []
    for conv in conversations:
        rows.append(
            {
                "conversation_id": conv["conversation_id"],
                "root_tweet_id": conv["root_tweet_id"],
                "customer_id": conv["customer_id"],
                "num_turns": conv["num_turns"],
                "initial_query": conv["initial_query"],
                "first_support_reply": conv["first_support_reply"],
                "has_support_reply": conv["has_support_reply"],
                "full_dialogue": " | ".join(
                    [
                        f"{'Customer' if t['inbound'] else 'AppleSupport'}: {t['text']}"
                        for t in conv["turns"]
                    ]
                ),
            }
        )
    return pd.DataFrame(rows)
