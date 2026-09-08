"""
tests/test_ingest_csv.py
-------------------------
Pytest tests for the CSV ingestion pipeline (src/data/ingest_csv.py).

Validates:
  - validate_csv() reports errors for insufficient rows
  - validate_csv() reports errors for missing required columns
  - validate_csv() passes on valid CIC-IDS style DataFrames
  - validate_csv() detects label column aliases
  - csv_to_windows() produces the required target columns
  - preprocess_for_inference() returns correct shape
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from src.data.ingest_csv import validate_csv, csv_to_windows, CSVValidationError


# ── Synthetic data helpers ─────────────────────────────────────────────────────

def make_flow_df(n_rows: int = 200, with_label: bool = True, label_col: str = "Label") -> pd.DataFrame:
    """Create a minimal but valid CIC-IDS-like DataFrame."""
    np.random.seed(42)
    ts_base = pd.Timestamp("2024-01-01 08:00:00")
    timestamps = [ts_base + pd.Timedelta(milliseconds=i * 100) for i in range(n_rows)]

    df = pd.DataFrame({
        "Timestamp":              timestamps,
        "Flow Duration":          np.random.exponential(1e6, n_rows),
        "Total Fwd Packets":      np.random.randint(1, 50, n_rows),
        "Total Backward Packets": np.random.randint(0, 30, n_rows),
        "SYN Flag Count":         np.random.randint(0, 3, n_rows),
        "ACK Flag Count":         np.random.randint(0, 10, n_rows),
        "Source IP":              ["192.168.1." + str(i % 10) for i in range(n_rows)],
        "Destination IP":         ["10.0.0." + str(i % 5) for i in range(n_rows)],
        "Destination Port":       np.random.choice([80, 443, 22, 21], n_rows),
        "Protocol":               np.random.choice([6, 17], n_rows),
    })

    if with_label:
        n_attack = min(20, max(0, n_rows - 1))
        labels = ["BENIGN"] * (n_rows - n_attack) + ["PortScan"] * n_attack
        np.random.shuffle(labels)
        df[label_col] = labels

    return df


# ── validate_csv tests ─────────────────────────────────────────────────────────

def test_validate_csv_too_few_rows():
    df = make_flow_df(n_rows=10)
    report = validate_csv(df)
    assert not report["ok"], "Should fail with < 60 rows"
    assert any("few rows" in e.lower() for e in report["errors"])


def test_validate_csv_missing_required_columns():
    df = make_flow_df(n_rows=200)
    df = df.drop(columns=["Flow Duration", "Total Fwd Packets"])
    report = validate_csv(df)
    assert not report["ok"], "Should fail when required columns are missing"
    assert len(report["missing_required"]) >= 2


def test_validate_csv_passes_valid_dataframe():
    df = make_flow_df(n_rows=200)
    report = validate_csv(df)
    assert report["ok"], f"Should pass on valid DataFrame. Errors: {report['errors']}"
    assert len(report["errors"]) == 0


def test_validate_csv_detects_label_column():
    df = make_flow_df(n_rows=200, label_col="Label")
    report = validate_csv(df)
    assert report["label_col"] == "Label"


def test_validate_csv_detects_label_aliases():
    for alias in ["label", "Category", "category"]:
        df = make_flow_df(n_rows=200, label_col=alias)
        report = validate_csv(df)
        assert report["label_col"] == alias, f"Failed to detect alias '{alias}'"


def test_validate_csv_no_label_column_warns():
    df = make_flow_df(n_rows=200, with_label=False)
    report = validate_csv(df)
    assert report["label_col"] is None
    assert any(report["warnings"]), "Should warn when no label column found"


def test_validate_csv_attack_rate():
    df = make_flow_df(n_rows=200)
    report = validate_csv(df)
    assert report["attack_rate"] is not None
    assert 0.0 <= report["attack_rate"] <= 1.0


# ── csv_to_windows tests ────────────────────────────────────────────────────────

REQUIRED_WINDOW_COLS = {"future_attack_1", "future_attack_3", "future_attack_5", "Label"}


def test_csv_to_windows_required_columns():
    df = make_flow_df(n_rows=500)
    windows = csv_to_windows(df, label_col="Label")
    missing = REQUIRED_WINDOW_COLS - set(windows.columns)
    assert not missing, f"Missing required window columns: {missing}"


def test_csv_to_windows_produces_rows():
    df = make_flow_df(n_rows=500)
    windows = csv_to_windows(df, label_col="Label")
    assert len(windows) > 0, "Should produce at least one window"


def test_csv_to_windows_future_labels_binary():
    df = make_flow_df(n_rows=500)
    windows = csv_to_windows(df, label_col="Label")
    for col in ["future_attack_1", "future_attack_3", "future_attack_5"]:
        unique_vals = set(windows[col].unique())
        assert unique_vals <= {0, 1}, f"Column {col} should be binary, got {unique_vals}"


def test_csv_to_windows_no_label_column():
    """Without a label column, windows should still be produced (all BENIGN)."""
    df = make_flow_df(n_rows=500, with_label=False)
    windows = csv_to_windows(df, label_col=None)
    assert len(windows) > 0
    assert (windows["Label"] == "BENIGN").all()


def test_csv_to_windows_synthesises_timestamps():
    """If no Timestamp column, should still work by synthesising timestamps."""
    df = make_flow_df(n_rows=500)
    df = df.drop(columns=["Timestamp"])
    windows = csv_to_windows(df, label_col="Label")
    assert len(windows) > 0
