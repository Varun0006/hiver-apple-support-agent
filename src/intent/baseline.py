"""
Baseline Intent Classifiers:
1. Majority Class Baseline (predicts most frequent training class)
2. TF-IDF + Logistic Regression Baseline
"""

from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np


class MajorityClassBaseline:
    """Baseline #1: Always predicts the most frequent intent in training data."""

    def __init__(self):
        self.majority_class = "other_general"

    def fit(self, X: List[str], y: List[str]):
        if y:
            counts = {}
            for label in y:
                counts[label] = counts.get(label, 0) + 1
            self.majority_class = max(counts, key=counts.get)
        return self

    def predict(self, X: List[str]) -> List[str]:
        return [self.majority_class] * len(X)


class TfIdfLogisticRegressionBaseline:
    """Baseline #2: TF-IDF feature extraction + Logistic Regression classifier."""

    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self.model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        self.is_fitted = False

    def fit(self, X: List[str], y: List[str]):
        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)
        self.is_fitted = True
        return self

    def predict(self, X: List[str]) -> List[str]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict.")
        X_vec = self.vectorizer.transform(X)
        return list(self.model.predict(X_vec))
