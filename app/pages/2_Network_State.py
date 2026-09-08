"""
app/pages/2_Network_State.py
-----------------------------
Step 2: Current Network State visualisation.
Uses repo-relative ROOT for any future file access.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st
import plotly.graph_objects as go

st.title("🌐 Step 2: Current Network State")

if not st.session_state.get("demo_loaded", False):
    st.warning("Please load data in the **Upload** page first.")
    st.stop()

idx    = st.session_state.current_idx
raw_df = st.session_state.raw_df

# ── Time navigation ────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("⏪ Previous Window") and idx > 30:
        st.session_state.current_idx -= 1
        st.rerun()
with col2:
    if st.button("▶ Next Window") and idx < len(raw_df) - 1:
        st.session_state.current_idx += 1
        st.rerun()
with col3:
    if st.button("Fast Forward 10 ⏩") and idx < len(raw_df) - 10:
        st.session_state.current_idx += 10
        st.rerun()

st.caption(f"Window **{idx}** / {len(raw_df) - 1}")

current_window = raw_df.iloc[st.session_state.current_idx]

# ── Metrics strip ──────────────────────────────────────────────────────────────
st.subheader("Current Traffic Snapshot")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Unique Dest IPs",   int(current_window.get("unique_destination_ips", 0)))
c2.metric("Active Flows",      int(current_window.get("flow_count", 0)))
c3.metric("SYN/ACK Ratio",     round(current_window.get("syn_to_ack_ratio", 0), 3))
c4.metric("True Label",        current_window.get("Label", "BENIGN"))
c5.metric(
    "Attack State",
    "🔴 ATTACK" if current_window.get("current_attack_state", 0) == 1 else "🟢 BENIGN",
)

# ── Gauge + feature bar ────────────────────────────────────────────────────────
col_gauge, col_bar = st.columns([1, 1])

with col_gauge:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(current_window.get("flow_count", 0)),
        title={"text": "Current Flow Volume"},
        gauge={
            "axis": {"range": [0, max(500, float(current_window.get("flow_count", 1)) * 1.5)]},
            "bar": {"color": "#e74c3c" if current_window.get("current_attack_state", 0) else "#2ecc71"},
            "steps": [
                {"range": [0, 100],  "color": "#d5f5e3"},
                {"range": [100, 300],"color": "#f9e79f"},
                {"range": [300, 500],"color": "#fadbd8"},
            ],
        },
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_bar:
    # Show top numeric features as a horizontal bar
    feature_names = st.session_state.get("feature_names", [])
    X_scaled      = st.session_state.get("X_scaled", None)
    if X_scaled is not None and idx < len(X_scaled):
        row_vals = X_scaled[idx]
        top_n = min(8, len(feature_names))
        top_idx = sorted(range(len(row_vals)), key=lambda i: abs(row_vals[i]), reverse=True)[:top_n]
        top_names  = [feature_names[i] for i in top_idx]
        top_values = [float(row_vals[i]) for i in top_idx]

        fig_bar = go.Figure(go.Bar(
            x=top_values,
            y=top_names,
            orientation="h",
            marker_color=["#e74c3c" if v > 0 else "#3498db" for v in top_values],
        ))
        fig_bar.update_layout(
            title="Top Scaled Feature Values",
            xaxis_title="Scaled Value",
            height=300,
            margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Feature values not available.")
