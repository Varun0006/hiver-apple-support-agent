"""
Historical AppleSupport conversation similarity retriever.
Uses TF-IDF + cosine similarity over the processed conversation index.
Index is built once and pickled for fast reuse.
"""

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

DEFAULT_INDEX_PATH = Path("data/processed/retrieval_index.pkl")
DEFAULT_CONVERSATIONS_CSV = Path("data/processed/apple_conversations.csv")


class HistoricalRetriever:
    """
    Retrieves top-k similar historical AppleSupport conversations as grounding evidence.

    Workflow:
        1. Load (or build) a TF-IDF index over all initial customer queries.
        2. For each new query, compute cosine similarity and return the top-k records.
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        conversations_csv: Optional[Path] = None,
        verbose: bool = True,
    ):
        self.index_path = Path(index_path) if index_path else DEFAULT_INDEX_PATH
        self.conversations_csv = (
            Path(conversations_csv) if conversations_csv else DEFAULT_CONVERSATIONS_CSV
        )
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._records: List[Dict[str, Any]] = []
        self._loaded = False
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    # Index construction / loading
    # ------------------------------------------------------------------ #

    def build_index(self, force: bool = False) -> None:
        """Build TF-IDF index from apple_conversations.csv and save to disk."""
        if self.index_path.exists() and not force:
            self._load_index()
            return

        if self.verbose:
            print(f"Building retrieval index from {self.conversations_csv}...")
        df = pd.read_csv(self.conversations_csv)

        # Keep only conversations that have a support reply
        df = df[df["has_support_reply"] == True].dropna(subset=["initial_query", "first_support_reply"])
        df = df.reset_index(drop=True)

        queries = df["initial_query"].astype(str).tolist()

        vectorizer = TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(queries)

        self._vectorizer = vectorizer
        self._matrix = matrix
        self._records = df.to_dict(orient="records")

        # Persist index
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": vectorizer,
                    "matrix": matrix,
                    "records": self._records,
                },
                f,
            )

        if self.verbose:
            print(f"Index built: {len(self._records):,} conversations indexed -> {self.index_path}")
        self._loaded = True

    def _load_index(self) -> None:
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
        self._vectorizer = data["vectorizer"]
        self._matrix = data["matrix"]
        self._records = data["records"]
        self._loaded = True
        if self.verbose:
            print(f"Retrieval index loaded: {len(self._records):,} conversations.")

    def ensure_loaded(self) -> None:
        if not self._loaded:
            if self.index_path.exists():
                self._load_index()
            else:
                self.build_index()

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top-k most similar historical conversations for a customer query.

        Args:
            query: Incoming customer message.
            top_k: Number of historical examples to return.

        Returns:
            List of dicts containing conversation metadata and similarity score.
        """
        self.ensure_loaded()

        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).flatten()

        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(sims[idx])
            if score < 0.01:          # skip near-zero matches
                continue
            record = dict(self._records[idx])
            record["similarity_score"] = round(score, 4)
            results.append(record)

        return results
