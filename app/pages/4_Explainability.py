"""
app/pages/4_Explainability.py
------------------------------
Step 4: Explainability using Captum Integrated Gradients.
Shows feature-level and temporal (time-step) attributions.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import joblib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.gru_world_model import GRUWorldModel
from src.inference.explain import explain_prediction, explain_temporal

st.set_page_config(page_title="Step 4: Explainability", page_icon="🔍", layout="wide")
st.title("🔍 Explainability (Integrated Gradients)")

if "current_x_tensor" not in st.session_state or "feature_names" not in st.session_state:
    st.warning("Please run the forecast in **Step 3** first to generate explainability context.")
    st.stop()

x_tensor = st.session_state["current_x_tensor"]
feature_names = st.session_state["feature_names"]

@st.cache_resource
def load_gru():
    scaler_path = ROOT / "models" / "scalers" / "baseline_scaler.pkl"
    gru_path    = ROOT / "models" / "gru" / "gru_world_model.pth"
    scaler = joblib.load(scaler_path)
    input_dim = scaler.n_features_in_
    
    model = GRUWorldModel(input_dim=input_dim, hidden_dim=128, num_layers=2)
    model.load_state_dict(torch.load(gru_path, map_location="cpu"))
    model.eval()
    return model

model = load_gru()

with st.spinner("Calculating Integrated Gradients (Feature & Temporal)..."):
    # Feature-level attribution (summed over time)
    top_features = explain_prediction(model, x_tensor, feature_names, top_n=10)
    
    # Temporal attribution (per timestep, per feature)
    attr_matrix, attr_by_step = explain_temporal(model, x_tensor, feature_names)

# ── Feature Importance Chart ──────────────────────────────────────────────────
st.markdown("### Top Influential Features")
st.caption("Shows which network features contributed most heavily to the model's risk score (summed across all 30 time windows).")

df_importance = pd.DataFrame(top_features)
# Sort ascending for horizontal bar chart
df_importance = df_importance.sort_values(by="Contribution", key=abs, ascending=True)

fig = px.bar(
    df_importance, 
    x='Contribution', 
    y='Feature', 
    orientation='h',
    color='Contribution',
    color_continuous_scale=px.colors.diverging.RdBu_r,
    color_continuous_midpoint=0
)
fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)


# ── Temporal Attribution Heatmap ──────────────────────────────────────────────
st.markdown("---")
st.markdown("### Temporal Attribution (When did the attack start?)")
st.caption("Shows exactly *which time windows* in the 30-window history triggered the alert. Darker red indicates higher contribution to the risk score.")

# Only plot the top 10 features to keep the heatmap readable
top_feat_names = [item["Feature"] for item in top_features]
top_feat_indices = [feature_names.index(f) for f in top_feat_names]

heatmap_data = attr_matrix[:, top_feat_indices].T  # shape: (10, 30)

fig_heat = go.Figure(data=go.Heatmap(
    z=heatmap_data,
    x=[f"T-{30-i}" for i in range(30)],
    y=top_feat_names,
    colorscale="RdBu_r",
    zmid=0
))
fig_heat.update_layout(
    xaxis_title="Time Window (T-30 to T-1)",
    height=400,
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig_heat, use_container_width=True)

# ── Total Attribution per Step ────────────────────────────────────────────────
st.markdown("### Total Risk Contribution per Time Window")
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    x=[f"T-{30-i}" for i in range(30)], 
    y=attr_by_step, 
    mode='lines+markers',
    line=dict(color='purple', width=3)
))
fig_line.update_layout(
    xaxis_title="Time Window",
    yaxis_title="Total Absolute Attribution",
    height=300,
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig_line, use_container_width=True)

st.info("💡 **How to interpret:** Integrated Gradients assigns a contribution score to every feature at every timestep. A large spike in the temporal plot indicates the exact moment the model detected anomalous behavior.")
