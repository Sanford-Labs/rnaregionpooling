"""metrics.py — the average precision a logistic probe is scored by, as mRNABench reports it."""
import numpy as np
from sklearn.metrics import average_precision_score


def _array(v):
    v = v.detach().cpu().double().numpy() if hasattr(v, "detach") else np.asarray(v, dtype=float)
    return v[:, None] if v.ndim == 1 else v


def average_precision(y, score):
    """sklearn's average precision of one label; NaN when the rows hold no positive."""
    y, score = np.asarray(y, dtype=float), np.asarray(score, dtype=float)
    return float(average_precision_score(y, score)) if y.sum() > 0 else float("nan")


def score_logistic(y, proba):
    """`ap` pools every label's rows into one ranking (micro, mRNABench's binary and multilabel columns); `ap_macro`
    averages the labels' own areas over the labels with a positive row (its eCLIP columns). Rows that are not finite
    are left out."""
    y, proba = _array(y), _array(proba)
    keep = np.isfinite(proba).all(1) & np.isfinite(y).all(1)
    y, proba = y[keep], proba[keep]
    each = [average_precision(y[:, j], proba[:, j]) for j in range(y.shape[1])]
    held = [v for v in each if not np.isnan(v)]
    return {"ap": average_precision(y.reshape(-1), proba.reshape(-1)),
            "ap_macro": float(np.mean(held)) if held else float("nan")}
