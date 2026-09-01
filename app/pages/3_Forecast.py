import streamlit as st
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.gru_world_model import GRUWorldModel
from src.models.fusion_model import RiskFusionEngine
from src.inference.mitre_mapper import MitreAttackMapper

st.title("ðŸ”® Step 3: Attack Forecast & Hybrid Risk")

if not st.session_state.get('demo_loaded', False):
    st.warning("Please load data in the Upload page first.")
    st.stop()

idx = st.session_state.current_idx
X_scaled = st.session_state.X_scaled
feature_names = st.session_state.feature_names
raw_df = st.session_state.raw_df

# Load Models
@st.cache_resource
def load_models():
    device = torch.device('cpu')
    model = GRUWorldModel(input_dim=len(feature_names), hidden_dim=128, num_layers=2)
    model.load_state_dict(torch.load("models/gru/gru_world_model.pth", map_location=device))
    model.eval()
    
    engine = RiskFusionEngine()
    engine.load("models/gru/fusion")
    
    mapper = MitreAttackMapper()
    return model, engine, mapper

model, engine, mapper = load_models()

# Extract 30-window sequence
seq = X_scaled[idx-30 : idx]
seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0) # Batch size 1

# Inference
with torch.no_grad():
    next_pred, fut_logits, stage_logits, z_t = model(seq_tensor)
    
p_forecast = torch.sigmoid(fut_logits[:, 0]).item()
stage_idx = torch.argmax(stage_logits, dim=1).item()
stages = ['Benign', 'Reconnaissance', 'Initial Access', 'Lateral Movement', 'Command and Control', 'Exfiltration']
pred_stage = stages[stage_idx] if stage_idx < len(stages) else 'Unknown'

state_error = torch.mean((next_pred - torch.tensor(X_scaled[idx], dtype=torch.float32).unsqueeze(0))**2).item()

# Hybrid Fusion
risk, a_ocsvm, r_err = engine.predict_risk([p_forecast], z_t.numpy(), [state_error])
final_risk = risk[0]

# UI Display (Exact requirements from Page 19)
st.markdown("### Essential Dashboard Display")
st.markdown(f"**Current infiltration probability:** {p_forecast:.2f}")
st.markdown(f"**Forecasted infiltration probability after 30 seconds (Risk):** {final_risk:.2f}")
st.markdown(f"**Novelty / Anomaly Score (OCSVM):** {a_ocsvm[0]:.2f}")

risk_class = "HIGH (Critical Warning)" if final_risk > 0.10 else "LOW (Benign)"
st.markdown(f"**Final risk classification:** {risk_class}")

# MITRE Mapping
current_features = raw_df.iloc[idx].to_dict()
mitre_info = mapper.map_to_mitre(pred_stage, final_risk, current_features)

st.success(f"**Predicted MITRE stage:** {mitre_info['predicted_stage']} -> {mitre_info['mapped_technique']} ({mitre_info['technique_id']})")

if mitre_info['evidence']:
    st.info("**Top Evidence:**\n- " + "\n- ".join(mitre_info['evidence']))

# Save tensors to session state for Explainability page
st.session_state.seq_tensor = seq_tensor
st.session_state.model = model
