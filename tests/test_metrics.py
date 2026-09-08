"""
tests/test_metrics.py
---------------------
Pytest tests for src/utils/metrics.py

Validates:
  - evaluate_classification() returns all required keys
  - FPR, Precision, Recall, F1 values are correct for known synthetic data
  - confusion_matrix_values are correct (TP, FP, TN, FN)
  - sample_counts are correct
  - PR-AUC and ROC-AUC are computed when y_prob is provided
  - format_metrics_row() returns a flat dict with expected columns
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
from src.utils.metrics import evaluate_classification, format_metrics_row

# ── Synthetic test cases ──────────────────────────────────────────────────────

# Perfect classifier: all correct
Y_TRUE_A  = np.array([0, 0, 0, 1, 1, 1])
Y_PRED_A  = np.array([0, 0, 0, 1, 1, 1])   # perfect
Y_PROB_A  = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])

# All zeros predicted (no attacks detected)
Y_TRUE_B  = np.array([0, 0, 1, 1])
Y_PRED_B  = np.array([0, 0, 0, 0])   # misses both attacks
Y_PROB_B  = np.array([0.1, 0.1, 0.3, 0.4])

# All ones predicted
Y_TRUE_C  = np.array([0, 0, 1, 1])
Y_PRED_C  = np.array([1, 1, 1, 1])   # FP=2, TP=2
Y_PROB_C  = np.array([0.9, 0.9, 0.9, 0.9])


# ── Key structure ─────────────────────────────────────────────────────────────

REQUIRED_KEYS = {
    "Accuracy", "Precision", "Recall", "F1-score", "Macro F1",
    "FPR", "ROC-AUC", "PR-AUC",
    "confusion_matrix_values", "sample_counts",
    "Inference_latency_sec",
}

def test_evaluate_returns_all_keys():
    m = evaluate_classification(Y_TRUE_A, Y_PRED_A, Y_PROB_A)
    missing = REQUIRED_KEYS - set(m.keys())
    assert not missing, f"Missing keys: {missing}"

def test_confusion_matrix_values_keys():
    m = evaluate_classification(Y_TRUE_A, Y_PRED_A)
    cm = m["confusion_matrix_values"]
    assert set(cm.keys()) == {"TP", "FP", "TN", "FN"}

def test_sample_counts_keys():
    m = evaluate_classification(Y_TRUE_A, Y_PRED_A)
    sc = m["sample_counts"]
    assert set(sc.keys()) == {"n_total", "n_pos", "n_neg"}


# ── Perfect classifier ────────────────────────────────────────────────────────

def test_perfect_f1():
    m = evaluate_classification(Y_TRUE_A, Y_PRED_A)
    assert m["F1-score"] == pytest.approx(1.0), "Perfect predictions should give F1=1.0"

def test_perfect_precision_recall():
    m = evaluate_classification(Y_TRUE_A, Y_PRED_A)
    assert m["Precision"] == pytest.approx(1.0)
    assert m["Recall"]    == pytest.approx(1.0)

def test_perfect_fpr_zero():
    m = evaluate_classification(Y_TRUE_A, Y_PRED_A)
    assert m["FPR"] == pytest.approx(0.0), "No false positives → FPR=0"

def test_perfect_confusion_matrix():
    m = evaluate_classification(Y_TRUE_A, Y_PRED_A)
    cm = m["confusion_matrix_values"]
    assert cm["TP"] == 3
    assert cm["TN"] == 3
    assert cm["FP"] == 0
    assert cm["FN"] == 0


# ── All-zero predictor ─────────────────────────────────────────────────────────

def test_all_zeros_recall_zero():
    m = evaluate_classification(Y_TRUE_B, Y_PRED_B)
    assert m["Recall"] == pytest.approx(0.0), "Recall should be 0 when all attacks missed"

def test_all_zeros_fpr_zero():
    m = evaluate_classification(Y_TRUE_B, Y_PRED_B)
    assert m["FPR"] == pytest.approx(0.0)


# ── All-ones predictor ────────────────────────────────────────────────────────

def test_all_ones_fpr_one():
    m = evaluate_classification(Y_TRUE_C, Y_PRED_C)
    assert m["FPR"] == pytest.approx(1.0), "All FP → FPR=1.0"

def test_all_ones_recall_one():
    m = evaluate_classification(Y_TRUE_C, Y_PRED_C)
    assert m["Recall"] == pytest.approx(1.0)

def test_all_ones_confusion_matrix():
    m = evaluate_classification(Y_TRUE_C, Y_PRED_C)
    cm = m["confusion_matrix_values"]
    assert cm["TP"] == 2
    assert cm["FP"] == 2
    assert cm["TN"] == 0
    assert cm["FN"] == 0


# ── AUC metrics ───────────────────────────────────────────────────────────────

def test_roc_auc_with_probs():
    m = evaluate_classification(Y_TRUE_A, Y_PRED_A, Y_PROB_A)
    assert m["ROC-AUC"] == pytest.approx(1.0), "Perfect probabilities → ROC-AUC=1.0"

def test_pr_auc_with_probs():
    m = evaluate_classification(Y_TRUE_A, Y_PRED_A, Y_PROB_A)
    assert m["PR-AUC"] == pytest.approx(1.0), "Perfect probabilities → PR-AUC=1.0"

def test_auc_none_without_probs():
    m = evaluate_classification(Y_TRUE_A, Y_PRED_A)
    assert m["ROC-AUC"] is None
    assert m["PR-AUC"]  is None


# ── Sample counts ─────────────────────────────────────────────────────────────

def test_sample_counts_correct():
    m = evaluate_classification(Y_TRUE_A, Y_PRED_A)
    sc = m["sample_counts"]
    assert sc["n_total"] == 6
    assert sc["n_pos"]   == 3
    assert sc["n_neg"]   == 3


# ── Latency passthrough ────────────────────────────────────────────────────────

def test_latency_passthrough():
    m = evaluate_classification(Y_TRUE_A, Y_PRED_A, latency=0.00123)
    assert m["Inference_latency_sec"] == pytest.approx(0.00123)


# ── format_metrics_row ────────────────────────────────────────────────────────

def test_format_metrics_row_has_model_key():
    known_m   = evaluate_classification(Y_TRUE_A, Y_PRED_A, Y_PROB_A)
    unknown_m = evaluate_classification(Y_TRUE_B, Y_PRED_B, Y_PROB_B)
    row = format_metrics_row("TestModel", known_m, unknown_m)
    assert row["Model"] == "TestModel"

def test_format_metrics_row_f1_values():
    known_m   = evaluate_classification(Y_TRUE_A, Y_PRED_A, Y_PROB_A)
    unknown_m = evaluate_classification(Y_TRUE_B, Y_PRED_B, Y_PROB_B)
    row = format_metrics_row("TestModel", known_m, unknown_m)
    assert row["Known F1"]   == pytest.approx(1.0, abs=1e-4)
    assert row["Unknown F1"] == pytest.approx(0.0, abs=1e-4)
