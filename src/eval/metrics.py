"""Metrics for the final evaluation (evaluation harness).

macro_f1 is macro-averaged F1 over all 65 classes; a class that is never predicted and never present
scores 0, the same definition as `macro_f1` in src/train/utils.py (sklearn, zero_division=0).
The fast version below builds the confusion matrix with bincount so a bootstrap of thousands of
resamples runs in seconds; tests/test_eval.py checks it against sklearn.
"""
from __future__ import annotations

import numpy as np

NUM_CLASSES = 65


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int = NUM_CLASSES) -> np.ndarray:
    """k x k matrix, rows = true class, columns = predicted class."""
    return np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int = NUM_CLASSES) -> float:
    cm = confusion(np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64), k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)               # = 2tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return float(f1.mean())


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, k: int = NUM_CLASSES) -> np.ndarray:
    cm = confusion(np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64), k)
    n = cm.sum(1)
    return np.divide(np.diag(cm), n, out=np.full(k, np.nan), where=n > 0)


def bootstrap_indices(n: int, reps: int, seed: int = 0) -> np.ndarray:
    """reps x n resampled image indices. Use the same matrix for every method being compared."""
    return np.random.default_rng(seed).integers(0, n, size=(reps, n))


def bootstrap_f1(y_true: np.ndarray, y_pred: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.array([macro_f1(y_true[i], y_pred[i]) for i in idx])


def percentile_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo, hi = np.percentile(x, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)
