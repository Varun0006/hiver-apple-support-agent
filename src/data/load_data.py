"""
DataLoader utility for twcs dataset and brand-specific filtering.
"""

from pathlib import Path
from typing import Set, Union
import pandas as pd


def load_raw_twcs(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load raw Twitter Customer Support CSV file."""
    return pd.read_csv(file_path, low_memory=False)


def extract_apple_support_tweets(df: pd.DataFrame) -> pd.DataFrame:
    """Extract tweets authored by or directed to AppleSupport, including linked thread context tweets.

    Args:
        df: Raw TWCS DataFrame.

    Returns:
        Filtered DataFrame containing AppleSupport conversations and contextual parent/child tweets.
    """
    df["tweet_id"] = df["tweet_id"].astype(str)
    df["author_id_str"] = df["author_id"].astype(str).str.lower()
    df["text_str"] = df["text"].astype(str).str.lower()

    # Direct AppleSupport tweets
    is_apple_author = df["author_id_str"] == "applesupport"
    is_apple_mention = df["text_str"].str.contains("@applesupport", regex=False)

    apple_tweet_ids: Set[str] = set(df[is_apple_author | is_apple_mention]["tweet_id"])

    # Expand to include referenced parents and response tweets to ensure complete thread reconstruction
    in_response_ids = set(
        df[df["tweet_id"].isin(apple_tweet_ids)]["in_response_to_tweet_id"]
        .dropna()
        .astype(str)
        .str.split(".")
        .str[0]  # strip any float conversion artifacts
    )

    # Response tweet ids can be comma separated
    response_ids = set()
    response_series = df[df["tweet_id"].isin(apple_tweet_ids)]["response_tweet_id"].dropna()
    for item in response_series:
        for r_id in str(item).split(","):
            r_clean = r_id.strip().split(".")[0]
            if r_clean:
                response_ids.add(r_clean)

    all_relevant_ids = apple_tweet_ids.union(in_response_ids).union(response_ids)

    filtered_df = df[df["tweet_id"].isin(all_relevant_ids)].copy()
    filtered_df.drop(columns=["author_id_str", "text_str"], inplace=True, errors="ignore")
    return filtered_df
