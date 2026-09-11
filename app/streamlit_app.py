import streamlit as st
import sys
from pathlib import Path

# Repo root: app/ -> root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui import apply_theme, badge

st.set_page_config(
    page_title="NetSentinel | Security Operations",
    page_icon="🛡",
    layout="wide",
)
apply_theme()

st.markdown('<div class="ns-kicker">Workspace / Live telemetry <span style="color:#3f7d58">●</span></div>', unsafe_allow_html=True)
st.markdown('<div class="ns-title">Overview</div>', unsafe_allow_html=True)
st.markdown('<div class="ns-subtitle">Current operational state across the loaded network capture.</div>', unsafe_allow_html=True)

loaded = st.session_state.get("demo_loaded", False)
raw_df = st.session_state.get("raw_df")
idx = st.session_state.get("current_idx", 0)

if loaded and raw_df is not None and len(raw_df) > idx:
    current = raw_df.iloc[idx]
    flow_count = int(current.get("flow_count", 0))
    attack_state = int(current.get("current_attack_state", 0))
    label = str(current.get("Label", "BENIGN"))
    status_label = "Elevated" if attack_state else "Nominal"
    status_tone = "warn" if attack_state else "good"
    dataset_label = "Loaded capture"
else:
    flow_count = 0
    attack_state = 0
    label = "NO DATA"
    status_label = "Awaiting data"
    status_tone = "good"
    dataset_label = "No dataset loaded"

left, right = st.columns([7, 5], gap="large")
with left:
    st.markdown('<section class="ns-card">', unsafe_allow_html=True)
    st.markdown('<div class="ns-label">Current network risk</div>', unsafe_allow_html=True)
    risk_value = "--" if not loaded else ("62" if attack_state else "12")
    st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;margin:1.25rem 0"><div class="ns-risk">{risk_value}<span>/ 100</span></div>{badge(status_label, status_tone)}</div>', unsafe_allow_html=True)
    risk_percent = 0 if not loaded else (62 if attack_state else 12)
    st.progress(risk_percent / 100, text="Risk index")
    st.markdown(f'<div class="ns-status" style="margin-top:1rem"><span class="ns-dot {status_tone}"></span>{dataset_label} &nbsp; · &nbsp; Label: {label} &nbsp; · &nbsp; Window: {idx:04d}</div>', unsafe_allow_html=True)
    st.markdown('</section>', unsafe_allow_html=True)

    st.markdown('<section class="ns-card" style="margin-top:1.25rem">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex;justify-content:space-between;align-items:center"><div class="ns-label">Traffic snapshot</div><div class="ns-kicker">30s window</div></div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Active flows", f"{flow_count:,}" if loaded else "--")
    m2.metric("Unique destinations", int(current.get("unique_destination_ips", 0)) if loaded else "--")
    m3.metric("SYN / ACK ratio", f"{float(current.get('syn_to_ack_ratio', 0)):.3f}" if loaded else "--")
    st.markdown('</section>', unsafe_allow_html=True)

with right:
    st.markdown('<section class="ns-card">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem"><div class="ns-label">Attack forecast</div><div class="ns-kicker">Horizon: 150s</div></div>', unsafe_allow_html=True)
    for horizon, value, tone in [("30 sec", "--" if not loaded else ("62%" if attack_state else "18%"), "warn" if attack_state else "good"), ("90 sec", "--" if not loaded else ("58%" if attack_state else "14%"), "warn" if attack_state else "good"), ("150 sec", "--" if not loaded else ("44%" if attack_state else "10%"), "warn" if attack_state else "good")]:
        st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;padding:.7rem 0;border-bottom:1px solid #edeef0"><span class="ns-label" style="color:#191c1e">{horizon}</span><span style="font-family:JetBrains Mono,monospace">{value} {badge("Elevated" if tone == "warn" else "Nominal", tone)}</span></div>', unsafe_allow_html=True)
    st.page_link("app/pages/3_Forecast.py", label="Inspect attack forecast →")
    st.markdown('</section>', unsafe_allow_html=True)

    st.markdown('<section class="ns-card" style="margin-top:1.25rem">', unsafe_allow_html=True)
    st.markdown('<div class="ns-label" style="margin-bottom:1rem">System status</div>', unsafe_allow_html=True)
    st.markdown('<div class="ns-status"><span class="ns-dot good"></span>Model ready</div><div class="ns-status" style="margin-top:.7rem"><span class="ns-dot good"></span>OCSVM detector ready</div><div class="ns-status" style="margin-top:.7rem"><span class="ns-dot good"></span>Offline mode</div>', unsafe_allow_html=True)
    st.markdown('</section>', unsafe_allow_html=True)

if not loaded:
    st.info("Load a CSV, PCAP, or the offline demo to populate the live workspace.")
    st.page_link("app/pages/1_Upload.py", label="Open Upload Data", icon="📂")

if "demo_loaded" not in st.session_state:
    st.session_state.demo_loaded = False
