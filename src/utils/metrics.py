import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)


def evaluate_classification(y_true, y_pred, y_prob=None, latency=0.0):
    """Calculates all mandatory metrics specified in Phase 3."""
    out = {}
    out['Accuracy']  = accuracy_score(y_true, y_pred)
    out['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    out['Recall']    = recall_score(y_true, y_pred, zero_division=0)
    out['F1-score']  = f1_score(y_true, y_pred, zero_division=0)
    out['Macro F1']  = f1_score(y_true, y_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        out['FPR'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        out['FPR'] = 0.0

    if y_prob is not None:
        try:
            out['ROC-AUC'] = roc_auc_score(y_true, y_prob)
            out['PR-AUC']  = average_precision_score(y_true, y_prob)
        except ValueError:
            out['ROC-AUC'] = 0.0
            out['PR-AUC']  = 0.0
    else:
        out['ROC-AUC'] = None
        out['PR-AUC']  = None

    out['Inference_latency_sec'] = latency
    return out
