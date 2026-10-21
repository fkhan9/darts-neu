"""
Classification metrics from raw logits, in plain NumPy (no sklearn needed).
Macro-F1 averages the per-class F1 with equal weight; a class with no
predictions and no support-hits gets F1 = 0.
"""

import numpy as np


def softmax(logits, axis=-1):
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def classification_report(logits, labels, class_names):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    C = len(class_names)
    probs = softmax(logits)
    preds = probs.argmax(axis=1)
    n = len(labels)

    cm = np.zeros((C, C), dtype=np.int64)          # rows = true class, columns = predicted
    np.add.at(cm, (labels, preds), 1)

    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros(C), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(C), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(C), where=denom > 0)

    p_true = np.clip(probs[np.arange(n), labels], 1e-12, None)
    return {
        'n': int(n),
        'accuracy': float((preds == labels).mean()),
        'errors': int((preds != labels).sum()),
        'macro_f1': float(f1.mean()),
        'nll': float(-np.log(p_true).mean()),
        'classes': list(class_names),
        'precision': precision.tolist(),
        'recall': recall.tolist(),       # equals per-class accuracy
        'f1': f1.tolist(),
        'support': support.astype(int).tolist(),
        'confusion_matrix': cm.tolist(),
    }
