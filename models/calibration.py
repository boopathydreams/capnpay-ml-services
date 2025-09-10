"""Probability calibration utilities (multiclass Platt)."""
from typing import Optional
import numpy as np
import pickle


class MulticlassPlatt:
    def __init__(self):
        self.classifiers = []  # one-vs-rest logistic scalers

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        # probs: (n, K) raw probabilities
        from sklearn.linear_model import LogisticRegression
        K = probs.shape[1]
        self.classifiers = []
        for k in range(K):
            yk = (y_true == k).astype(int)
            lr = LogisticRegression(max_iter=500)
            lr.fit(probs, yk)
            self.classifiers.append(lr)

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self.classifiers:
            return probs
        K = probs.shape[1]
        out = np.zeros_like(probs)
        for k in range(K):
            out[:, k] = self.classifiers[k].predict_proba(probs)[:, 1]
        # renormalize
        out_sum = out.sum(axis=1, keepdims=True) + 1e-9
        return out / out_sum

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> "MulticlassPlatt":
        with open(path, "rb") as f:
            return pickle.load(f)

