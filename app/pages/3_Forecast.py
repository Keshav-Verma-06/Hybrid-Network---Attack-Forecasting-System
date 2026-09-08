"""
app/pages/3_Forecast.py
-----------------------
Step 3: Attack Forecast & Hybrid Risk

Changes:
  - Displays full autoregressive rollout timeline using src.inference.rollout
  - Shows breakdown of hybrid risk score (P_forecast, a_ocsvm, r_error)
  - Uses full YAML-based MITRE evaluation
"""

import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import joblib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.gru_world_model import GRUWorldModel
from src.models.fusion_model import RiskFusionEngine
from src.inference.mitre_mapper import MitreAttackMapper
from src.inference.rollout import recursive_rollout
from src.enterprise.asset_manager import AssetManager
from src.enterprise.siem_notifier import SIEMNotifier
from src.enterprise.audit_logger import AuditLogger

# ── Setup & Load ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Step 3: Forecast", page_icon="🔮", layout="wide")
st.title("🔮 Attack Forecast & Fused Risk")

if "df" not in st.session_state or "current_idx" not in st.session_state:
    st.warning("Please upload data in **Step 1** first.")
    st.stop()

@st.cache_resource
def load_models():
    scaler_path = ROOT / "models" / "scalers" / "baseline_scaler.pkl"
    gru_path    = ROOT / "models" / "gru" / "gru_world_model.pth"
    fusion_pfx  = str(ROOT / "models" / "gru" / "fusion")

    scaler = joblib.load(scaler_path)
    
    # Needs to match features from scaler exactly. Our scaler was fit on input_dim
    # We can infer input_dim from scaler.mean_.shape[0] if it's StandardScaler
    # But since it's just imputing medians usually, wait... The baseline scaler
    # is a MinMaxScaler fit on the features.
    input_dim = scaler.n_features_in_
    
    model = GRUWorldModel(input_dim=input_dim, hidden_dim=128, num_layers=2)
    model.load_state_dict(torch.load(gru_path, map_location="cpu"))
    model.eval()

    engine = RiskFusionEngine()
    engine.load(fusion_pfx)

    mapper = MitreAttackMapper()
    
    asset_mgr = AssetManager()
    siem = SIEMNotifier(webhook_url="http://mock-siem:8080/hec", syslog_host="mock")
    logger = AuditLogger()
    
    threshold_file = ROOT / "outputs" / "metrics" / "optimal_threshold.json"
    threshold = 0.10
    if threshold_file.exists():
        with open(threshold_file) as f:
            threshold = json.load(f).get("threshold", 0.10)
            
    return scaler, model, engine, mapper, asset_mgr, siem, logger, threshold

scaler, model, engine, mapper, asset_mgr, siem, logger, optimal_threshold = load_models()

df = st.session_state["df"]
idx = st.session_state["current_idx"]
seq_len = 30

if idx < seq_len - 1:
    st.warning(f"Need at least {seq_len} windows of history to forecast. Please advance the timeline.")
    st.stop()

# ── Prepare Context Window ────────────────────────────────────────────────────
exclude_cols = {"Timestamp", "Label", "attack_stage"}
exclude_cols.update({f"future_attack_{k}" for k in [1, 3, 5]})
features = [c for c in df.columns if c not in exclude_cols]

context_df = df.iloc[idx - seq_len + 1 : idx + 1]
current_features = context_df.iloc[-1][features].to_dict()

X_context = scaler.transform(context_df[features].fillna(0).values)
x_tensor  = torch.tensor(X_context, dtype=torch.float32).unsqueeze(0) # (1, 30, dim)

# ── Run Inference & Rollout ───────────────────────────────────────────────────
with st.spinner("Simulating future states..."):
    # Current timestep (t) prediction
    with torch.no_grad():
        next_pred, fut_logits, stage_logits, z_t = model(x_tensor)
        probs = torch.sigmoid(fut_logits[0]).cpu().numpy()
        p_k1, p_k3, p_k5 = probs
        
        predicted_stage_idx = torch.argmax(stage_logits[0]).item()
        
        # State reconstruction error
        last_real = x_tensor[0, -1, :].numpy()
        state_err = float(np.mean((next_pred[0].numpy() - last_real) ** 2))
        
        # Hybrid Risk
        risk_arr, a_arr, r_arr = engine.predict_risk(np.array([p_k1]), z_t.numpy(), np.array([state_err]))
        current_risk = float(risk_arr[0])
        current_a_ocsvm = float(a_arr[0])
        current_r_error = float(r_arr[0])

    # MITRE Mapping
    # Convert predicted_stage_idx back to name.  Assuming STAGE_MAP inverse:
    stage_names = ["Benign", "Reconnaissance", "Initial Access", "Lateral Movement", "Command and Control", "Exfiltration"]
    gru_stage_name = stage_names[predicted_stage_idx] if predicted_stage_idx < len(stage_names) else "Unknown"
    
    mitre_info = mapper.map_to_mitre(gru_stage_name, current_risk, current_features)

    # Autoregressive Rollout (10 steps)
    timeline = recursive_rollout(model, engine, x_tensor, n_steps=10)

# ── Enterprise Asset Context ──────────────────────────────────────────────────
# Default to a mock destination IP since it's not explicitly in the df
target_ip = "10.0.0.50" 
asset_info = asset_mgr.get_asset_info(target_ip)
adjusted_risk = asset_mgr.adjust_risk(current_risk, target_ip)

is_alert = adjusted_risk > optimal_threshold

# Log alert and send to SIEM if it just crossed the threshold
if is_alert:
    alert_payload = {
        "risk": adjusted_risk,
        "target_ip": target_ip,
        "asset_zone": asset_info["zone"],
        "predicted_stage": mitre_info["predicted_stage"],
        "mapped_technique": mitre_info["mapped_technique"],
        "technique_id": mitre_info["technique_id"],
        "evidence": mitre_info["evidence"]
    }
    logger.log_alert(alert_payload)
    siem.emit(alert_payload)

# ── UI Presentation ───────────────────────────────────────────────────────────
st.markdown("### Hybrid Risk Assessment")
st.caption(f"Target Asset: **{target_ip}** | Zone: **{asset_info['zone']}** | Criticality: **{asset_info['criticality']}/5** ({asset_info['role']})")

col1, col2, col3, col4 = st.columns(4)

alert_color = "red" if is_alert else "green"
alert_text  = "🚨 CRITICAL ALERT" if is_alert else "✅ NORMAL"

col1.metric("Risk Status", alert_text)
col2.metric("Adjusted Risk Score", f"{adjusted_risk:.3f}", delta=f"Base Risk: {current_risk:.3f}", delta_color="off")
col3.metric("OCSVM Anomaly (Z_t)", f"{current_a_ocsvm:.3f}")
col4.metric("State Error (R_err)", f"{current_r_error:.3f}")

# ── MITRE ATT&CK Mapping ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### MITRE ATT&CK Mapping")

if mitre_info["mapped_technique"] != "Normal Background Traffic":
    st.error(f"**Detected Tactic/Stage:** {mitre_info['predicted_stage']}  \n"
             f"**Mapped Technique:** {mitre_info['mapped_technique']} ({mitre_info['technique_id']})")
    
    if mitre_info.get("data_driven"):
        st.caption("Mapped via YAML condition rules overriding GRU base prediction.")
    
    st.markdown("**Evidence Triggers:**")
    for ev in mitre_info["evidence"]:
        st.markdown(f"- {ev}")
else:
    st.success("**Traffic appears benign.** No active MITRE ATT&CK techniques detected.")

# ── Autoregressive Timeline Chart ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### Future Risk Rollout (Next 5 Minutes)")

time_offsets = [0] + [t["timestamp_offset_s"] for t in timeline]
risk_values  = [current_risk] + [t["risk"] for t in timeline]
p1_values    = [p_k1] + [t["p_k1"] for t in timeline]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=time_offsets, y=risk_values, 
    mode='lines+markers', name='Fused Risk (Hybrid)',
    line=dict(color='red', width=3),
    marker=dict(size=8)
))
fig.add_trace(go.Scatter(
    x=time_offsets, y=p1_values, 
    mode='lines', name='P_forecast (GRU)',
    line=dict(color='orange', width=2, dash='dash')
))

# Threshold line
fig.add_hline(y=optimal_threshold, line_dash="dot", line_color="red", annotation_text="Alert Threshold")

fig.update_layout(
    xaxis_title="Seconds into the Future",
    yaxis_title="Risk Score (0-1)",
    yaxis_range=[0, 1.05],
    height=400,
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# ── Score Breakdown Chart ────────────────────────────────────────────────────
st.markdown("### Current Score Breakdown")
breakdown_fig = go.Figure(data=[
    go.Bar(name='GRU Forecast (P_forecast)', x=['Component Weighting'], y=[p_k1 * 0.50], marker_color='orange'),
    go.Bar(name='OCSVM Anomaly (A_ocsvm)', x=['Component Weighting'], y=[current_a_ocsvm * 0.30], marker_color='purple'),
    go.Bar(name='Recon Error (R_err)', x=['Component Weighting'], y=[current_r_error * 0.20], marker_color='blue')
])
breakdown_fig.update_layout(
    barmode='stack',
    height=300,
    yaxis_title="Contribution to Risk",
    yaxis_range=[0, 1.0],
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(breakdown_fig, use_container_width=True)

st.session_state["current_x_tensor"] = x_tensor
st.session_state["feature_names"] = features
