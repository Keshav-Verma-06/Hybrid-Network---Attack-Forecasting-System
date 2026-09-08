"""
app/pages/1_Upload.py
----------------------
Step 1: Upload & Initialization

Supports:
  - Real CSV upload (flow-level network data) with column validation
  - PCAP upload with Scapy-based flow extraction
  - Offline fallback demo (known test set from training pipeline)
"""

import sys
from pathlib import Path

# Repo root: app/pages -> app -> root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import joblib
import numpy as np

from src.data.ingest_csv import validate_csv, csv_to_windows, preprocess_for_inference, CSVValidationError

st.title("📂 Step 1: Upload & Initialization")
st.markdown(
    "Upload a **CSV** (flow-level network traffic) or a **PCAP** file, "
    "or load the offline fallback demo from the training test set."
)

# ── Shared helpers ────────────────────────────────────────────────────────────

SCALER_PATH  = ROOT / "models" / "scalers" / "baseline_scaler.pkl"
DEMO_PARQUET = ROOT / "data" / "processed" / "windows" / "test_known_windows_30s.parquet"

EXCLUDE = {"Timestamp", "Label", "attack_stage",
           "future_attack_1", "future_attack_3", "future_attack_5"}


def _load_scaler():
    if not SCALER_PATH.exists():
        st.error(f"Scaler not found at `{SCALER_PATH}`. Run Phase 3 training first.")
        st.stop()
    return joblib.load(SCALER_PATH)


def _set_session(df: pd.DataFrame, X_scaled: np.ndarray, feature_names: list):
    exclude = EXCLUDE | {"current_attack_state"}
    feature_names = [f for f in feature_names if f not in exclude]
    st.session_state.raw_df        = df
    st.session_state.X_scaled      = X_scaled
    st.session_state.feature_names = feature_names
    st.session_state.demo_loaded   = True

    attack_indices = df[df.get("current_attack_state", pd.Series(dtype=int)) == 1].index \
        if "current_attack_state" in df.columns else pd.Index([])
    start_idx = max(30, int(attack_indices[0]) - 5) if len(attack_indices) > 0 else 30
    st.session_state.current_idx = start_idx


# ── Tab layout ────────────────────────────────────────────────────────────────

tab_csv, tab_pcap, tab_demo = st.tabs(["📄 CSV Upload", "📡 PCAP Upload", "🎯 Offline Demo"])

# ─── CSV Tab ─────────────────────────────────────────────────────────────────
with tab_csv:
    st.subheader("Upload a CSV of network flows")
    st.markdown(
        "Expected format: CIC-IDS compatible CSV with columns such as "
        "`Flow Duration`, `Total Fwd Packets`, `SYN Flag Count`, `Label`, etc."
    )

    uploaded_csv = st.file_uploader(
        "Choose a CSV file", type=["csv"], key="csv_uploader"
    )

    if uploaded_csv is not None:
        with st.spinner("Reading CSV…"):
            try:
                raw_df = pd.read_csv(uploaded_csv, encoding="latin1", low_memory=False)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                st.stop()

        # Validate
        report = validate_csv(raw_df)

        with st.expander("📋 Validation Report", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows",    report["n_rows"])
            col2.metric("Columns", report["n_cols"])
            col3.metric(
                "Attack Rate",
                f"{report['attack_rate']:.1%}" if report["attack_rate"] is not None else "Unknown"
            )

            if report["errors"]:
                for err in report["errors"]:
                    st.error(f"❌ {err}")
            if report["warnings"]:
                for warn in report["warnings"]:
                    st.warning(f"⚠️ {warn}")
            if report["ok"]:
                st.success(f"✅ Found {len(report['found_features'])} matching feature columns.")

        if not report["ok"]:
            st.stop()

        if st.button("🚀 Process CSV & Load into Dashboard", key="process_csv"):
            with st.spinner("Aggregating into 30-second windows…"):
                try:
                    windows = csv_to_windows(raw_df, label_col=report["label_col"])
                except Exception as e:
                    st.error(f"Preprocessing failed: {e}")
                    st.stop()

            if len(windows) < 31:
                st.error(
                    f"Only {len(windows)} windows produced — need at least 31 "
                    "(30 for the GRU sequence + 1 for next-state prediction). "
                    "Upload a longer capture."
                )
                st.stop()

            with st.spinner("Scaling features…"):
                try:
                    X_scaled, feature_names = preprocess_for_inference(windows, SCALER_PATH)
                except FileNotFoundError as e:
                    st.error(str(e))
                    st.stop()

            _set_session(windows, X_scaled, feature_names)

            st.success(
                f"✅ CSV loaded! {len(windows)} windows produced. "
                f"Proceed to **Network State** in the sidebar."
            )
            st.info(
                f"📊 Feature dimensions: **{X_scaled.shape[1]}** | "
                f"Window auto-advanced to index **{st.session_state.current_idx}**"
            )


# ─── PCAP Tab ─────────────────────────────────────────────────────────────────
with tab_pcap:
    st.subheader("Upload a PCAP / PCAPNG file")
    st.markdown(
        "Scapy will parse the packets into flows and then aggregate them "
        "into 30-second windows. **Note:** very large PCAPs (>100 MB) may "
        "take 30+ seconds to process."
    )

    uploaded_pcap = st.file_uploader(
        "Choose a PCAP file", type=["pcap", "pcapng", "cap"], key="pcap_uploader"
    )

    if uploaded_pcap is not None:
        if st.button("🚀 Parse PCAP & Load into Dashboard", key="process_pcap"):
            with st.spinner("Parsing packets with Scapy… (this may take a moment)"):
                try:
                    from src.data.ingest_pcap import pcap_bytes_to_windows
                    pcap_bytes = uploaded_pcap.read()
                    windows = pcap_bytes_to_windows(pcap_bytes)
                except ImportError:
                    st.error("Scapy is not installed. Run: `.venv\\Scripts\\pip install scapy`")
                    st.stop()
                except Exception as e:
                    st.error(f"PCAP parsing failed: {e}")
                    st.stop()

            if len(windows) < 31:
                st.error(
                    f"Only {len(windows)} windows from PCAP — capture is too short. "
                    "Need at least 31 windows (≈ 15.5 minutes of traffic)."
                )
                st.stop()

            with st.spinner("Scaling features…"):
                try:
                    X_scaled, feature_names = preprocess_for_inference(windows, SCALER_PATH)
                except FileNotFoundError as e:
                    st.error(str(e))
                    st.stop()

            _set_session(windows, X_scaled, feature_names)
            st.success(
                f"✅ PCAP processed! {len(windows)} windows. "
                "Proceed to **Network State** in the sidebar."
            )

            # Show extraction summary
            with st.expander("PCAP Extraction Summary"):
                st.metric("Windows produced", len(windows))
                if "flow_count" in windows.columns:
                    st.metric("Total flows (first window)", int(windows["flow_count"].iloc[0]))
                if "current_attack_state" in windows.columns:
                    attack_pct = windows["current_attack_state"].mean() * 100
                    st.metric("Windows with attack traffic", f"{attack_pct:.1f}%")


# ─── Offline Demo Tab ─────────────────────────────────────────────────────────
with tab_demo:
    st.subheader("Offline Fallback Demo (Known Test Set)")
    st.markdown(
        "Loads the pre-processed known test set from the training pipeline. "
        "Auto-advances to the window just before the first attack so you can "
        "watch the risk score rise in real time."
    )

    if not DEMO_PARQUET.exists():
        st.error(
            f"Demo data not found at `{DEMO_PARQUET}`. "
            "Run the training pipeline first."
        )
    else:
        if st.button("🎯 Load Offline Fallback Demo", key="load_demo"):
            with st.spinner("Loading temporal sequences and models…"):
                df = pd.read_parquet(DEMO_PARQUET)
                scaler = _load_scaler()
                features_all = [c for c in df.columns if c not in EXCLUDE]
                X_scaled = scaler.transform(df[features_all].fillna(0).values)
                _set_session(df, X_scaled, features_all)

            st.success(
                f"✅ Demo loaded! Auto-skipped to Window "
                f"**{st.session_state.current_idx}** (attack imminent). "
                "Proceed to **Network State**."
            )
