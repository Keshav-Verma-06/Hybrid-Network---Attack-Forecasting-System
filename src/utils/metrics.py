"""
src/utils/metrics.py  (v3 — per-class + lead time)
----------------------------------------------------
Comprehensive metrics for all evaluation phases.

Functions:
  evaluate_classification()  — binary classification metrics (unchanged API)
  format_metrics_row()       — flatten for benchmark tables
  per_class_evaluation()     — per attack-type F1/precision/recall
  per_stage_evaluation()     — per MITRE stage breakdown
  compute_lead_time()        — detection lead-time before first compromise
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    classification_report,
)


# ── Binary classification ─────────────────────────────────────────────────────

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    latency: float = 0.0,
) -> dict:
    """
    Full binary classification metrics.

    Returns dict with:
      Accuracy, Precision, Recall, F1-score, Macro F1,
      FPR, ROC-AUC, PR-AUC, Inference_latency_sec,
      confusion_matrix_values {TP, FP, TN, FN},
      sample_counts {n_total, n_pos, n_neg}
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    out: dict = {}
    out["Accuracy"]  = float(accuracy_score(y_true, y_pred))
    out["Precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["Recall"]    = float(recall_score(y_true, y_pred, zero_division=0))
    out["F1-score"]  = float(f1_score(y_true, y_pred, zero_division=0))
    out["Macro F1"]  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        out["FPR"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        out["confusion_matrix_values"] = {
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        }
    else:
        out["FPR"] = 0.0
        out["confusion_matrix_values"] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        try:
            out["ROC-AUC"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            out["ROC-AUC"] = None
        try:
            out["PR-AUC"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            out["PR-AUC"] = None
    else:
        out["ROC-AUC"] = None
        out["PR-AUC"]  = None

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    out["sample_counts"] = {"n_total": len(y_true), "n_pos": n_pos, "n_neg": n_neg}
    out["Inference_latency_sec"] = float(latency)
    return out


def format_metrics_row(
    model_name: str,
    known_metrics: dict,
    unknown_metrics: dict,
    horizon: str = "K=1",
    include_forecast_horizons: bool = False,
    extra_horizons: dict | None = None,
) -> dict:
    """Flatten metrics dicts into a single benchmark table row."""
    row: dict = {
        "Model":               model_name,
        "Known F1":            round(known_metrics.get("F1-score", 0), 4),
        "Unknown F1":          round(unknown_metrics.get("F1-score", 0), 4),
        "Precision (Known)":   round(known_metrics.get("Precision", 0), 4),
        "Recall (Known)":      round(known_metrics.get("Recall", 0), 4),
        "FPR":                 round(known_metrics.get("FPR", 0), 4),
        "PR-AUC (Known)":      round(known_metrics.get("PR-AUC") or 0, 4),
        "Inference latency (ms)": round(known_metrics.get("Inference_latency_sec", 0) * 1000, 4),
        "N (Known)":   known_metrics.get("sample_counts", {}).get("n_total", "N/A"),
        "N (Unknown)": unknown_metrics.get("sample_counts", {}).get("n_total", "N/A"),
    }
    if include_forecast_horizons and extra_horizons:
        for k_label, m in extra_horizons.items():
            row[f"Forecast F1 {k_label}"] = round(m.get("F1-score", 0), 4)
    else:
        row["Forecast F1@1"] = "N/A (baseline)"
        row["Forecast F1@3"] = "N/A (baseline)"
        row["Forecast F1@5"] = "N/A (baseline)"
    return row


# ── Per-class / per-stage evaluation ─────────────────────────────────────────

def per_class_evaluation(
    y_true_labels: list[str],
    y_pred_binary: np.ndarray,
    attack_labels: list[str],
) -> pd.DataFrame:
    """
    Compute Precision, Recall, F1 for each individual attack type.

    Parameters
    ----------
    y_true_labels  : list of true attack label strings (per window), e.g. 'PortScan', 'BENIGN'
    y_pred_binary  : binary predictions (0/1) aligned with y_true_labels
    attack_labels  : unique attack types to evaluate (excluding 'BENIGN')

    Returns
    -------
    DataFrame with columns [Attack, N_windows, N_predicted, Precision, Recall, F1, Detected]
    """
    rows = []
    y_true_labels = np.asarray(y_true_labels)
    y_pred_binary = np.asarray(y_pred_binary, dtype=int)

    for attack in attack_labels:
        mask = y_true_labels == attack
        n_windows  = int(np.sum(mask))
        if n_windows == 0:
            continue
        # "Detected" = at least one predicted-positive in this attack's windows
        n_pred_pos = int(np.sum(y_pred_binary[mask]))
        detected   = n_pred_pos > 0

        # Precision and Recall over this attack's windows only
        y_t = (mask).astype(int)         # this attack = 1, else 0
        p   = float(precision_score(y_t, y_pred_binary, zero_division=0))
        r   = float(recall_score(y_t, y_pred_binary, zero_division=0))
        f   = float(f1_score(y_t, y_pred_binary, zero_division=0))

        rows.append({
            "Attack":      attack,
            "N_windows":   n_windows,
            "N_predicted": n_pred_pos,
            "Precision":   round(p, 4),
            "Recall":      round(r, 4),
            "F1":          round(f, 4),
            "Detected":    detected,
        })

    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)


def per_stage_evaluation(
    stage_labels: list[str],
    y_pred_binary: np.ndarray,
) -> pd.DataFrame:
    """
    Compute detection rate per MITRE ATT&CK stage.

    Parameters
    ----------
    stage_labels  : MITRE stage string per window (from 'attack_stage' column)
    y_pred_binary : binary predictions aligned with stage_labels

    Returns
    -------
    DataFrame with columns [Stage, N_windows, N_detected, Detection_Rate]
    """
    stage_labels  = np.asarray(stage_labels)
    y_pred_binary = np.asarray(y_pred_binary, dtype=int)
    rows = []

    for stage in sorted(set(stage_labels)):
        mask      = stage_labels == stage
        n_windows = int(np.sum(mask))
        n_det     = int(np.sum(y_pred_binary[mask]))
        rate      = n_det / max(n_windows, 1)
        rows.append({
            "Stage":           stage,
            "N_windows":       n_windows,
            "N_detected":      n_det,
            "Detection_Rate":  round(rate, 4),
        })

    return pd.DataFrame(rows).sort_values("Detection_Rate", ascending=False).reset_index(drop=True)


# ── Detection lead time ───────────────────────────────────────────────────────

def compute_lead_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window_duration_s: int = 30,
) -> dict:
    """
    Measure how many windows (and seconds) before the first true attack
    the model raises a True Positive alert.

    An "attack episode" starts when y_true transitions 0→1.
    "Lead time" = index of first TP alarm before that transition.

    Parameters
    ----------
    y_true             : binary ground truth (0=benign, 1=attack)
    y_pred             : binary predictions
    window_duration_s  : seconds per window (default 30)

    Returns
    -------
    dict with keys:
      n_episodes         — number of detected attack onsets
      lead_times_windows — list of per-episode lead times in windows
      lead_times_sec     — list of per-episode lead times in seconds
      mean_lead_sec      — mean lead time (seconds), or None if no episodes
      median_lead_sec    — median lead time
      min_lead_sec       — minimum lead time (earliest warning)
      max_lead_sec       — maximum lead time
      missed_episodes    — episodes with no pre-alarm (0 lead time)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # Find attack episode start indices (0→1 transition)
    transitions = np.where((y_true[:-1] == 0) & (y_true[1:] == 1))[0] + 1

    lead_times_windows = []
    missed = 0

    for onset_idx in transitions:
        # Look back up to 10 windows before the onset for any TP alarm
        # (alarm = predicted 1 while true is still 0)
        look_back = max(0, onset_idx - 10)
        pre_alarm_region = y_pred[look_back:onset_idx]
        alarm_indices = np.where(pre_alarm_region == 1)[0]

        if len(alarm_indices) > 0:
            # Lead time = distance from earliest pre-alarm to onset
            earliest = look_back + alarm_indices[0]
            lead     = onset_idx - earliest
            lead_times_windows.append(lead)
        else:
            lead_times_windows.append(0)
            missed += 1

    lead_times_sec = [lt * window_duration_s for lt in lead_times_windows]

    nonzero = [lt for lt in lead_times_sec if lt > 0]

    return {
        "n_episodes":          len(transitions),
        "lead_times_windows":  lead_times_windows,
        "lead_times_sec":      lead_times_sec,
        "mean_lead_sec":       float(np.mean(nonzero))   if nonzero else None,
        "median_lead_sec":     float(np.median(nonzero)) if nonzero else None,
        "min_lead_sec":        float(np.min(nonzero))    if nonzero else None,
        "max_lead_sec":        float(np.max(nonzero))    if nonzero else None,
        "missed_episodes":     missed,
    }
