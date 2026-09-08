import streamlit as st
import sys
from pathlib import Path

# Repo root: app/ -> root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="NetSentinel World Model",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ NetSentinel: Offline Network-Attack Forecaster")
st.markdown("""
Welcome to the **NetSentinel World Model** prototype.

This system uses a **Temporal GRU + OCSVM Hybrid** architecture to:
- Forecast network attacks **before** they happen (horizons K=1, K=3, K=5)
- Detect **Zero-Day** threats via latent-space anomaly detection
- Map predicted attack stages to **MITRE ATT&CK** techniques

👈 **Use the sidebar to navigate through the 5 modules.**
""")

col1, col2, col3 = st.columns(3)
col1.info("📂 **Step 1** — Upload a CSV or PCAP file (or load the offline demo)")
col2.info("🔮 **Step 3** — View multi-horizon attack forecasts")
col3.info("📊 **Step 5** — Compare model performance with full metrics")

if "demo_loaded" not in st.session_state:
    st.session_state.demo_loaded = False
