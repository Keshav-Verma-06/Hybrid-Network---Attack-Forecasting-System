"""
app/pages/6_Enterprise_Admin.py
-------------------------------
Enterprise Administration Dashboard.
Displays Audit Logs, Active Alerts, and Statistical Drift Monitoring.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.enterprise.audit_logger import AuditLogger

st.set_page_config(page_title="Enterprise Admin", page_icon="🏢", layout="wide")
st.title("🏢 Enterprise Administration")

# Initialize Audit Logger
logger = AuditLogger()

tab1, tab2, tab3 = st.tabs(["Alert History", "Audit Logs", "Model Drift Monitor"])

with tab1:
    st.markdown("### Recent High-Confidence Alerts")
    alerts = logger.get_recent_alerts(limit=100)
    if alerts:
        df_alerts = pd.DataFrame(alerts)
        st.dataframe(
            df_alerts.style.applymap(
                lambda x: 'background-color: #ffcccc; color: black;' if isinstance(x, float) and x > 0.8 else '', 
                subset=['risk_score']
            ),
            use_container_width=True
        )
    else:
        st.info("No active alerts in the history log.")

with tab2:
    st.markdown("### System Audit Logs")
    logs = logger.get_recent_logs(limit=100)
    if logs:
        st.dataframe(pd.DataFrame(logs), use_container_width=True)
    else:
        st.info("No audit logs available.")
        
    if st.button("Enforce 90-Day Retention Policy"):
        logs_del, alerts_del = logger.enforce_retention_policy(90)
        logger.log_event("Retention Policy Enforced", "Admin", f"Deleted {logs_del} logs, {alerts_del} alerts.")
        st.success(f"Deleted {logs_del} old logs and {alerts_del} old alerts.")

with tab3:
    st.markdown("### Statistical Concept Drift Monitoring")
    st.caption("Monitors the latent embedding space (z_t) to detect when the model requires recalibration.")
    
    drift_file = ROOT / "outputs" / "metrics" / "drift_monitoring.json"
    
    if drift_file.exists():
        with open(drift_file) as f:
            drift_data = json.load(f)
            
        col1, col2, col3 = st.columns(3)
        col1.metric("Drifted Latent Dimensions", f"{drift_data['drifted_dimensions_count']} / {drift_data['latent_dimensions']}")
        col2.metric("Drift Ratio", f"{drift_data['drift_ratio']:.1%}", delta=f"Threshold: {drift_data['recalibration_threshold_ratio']:.1%}", delta_color="inverse")
        col3.metric("Mean KS Statistic", f"{drift_data['mean_ks_statistic']:.4f}")
        
        if drift_data['requires_recalibration']:
            st.error("🚨 **Concept Drift Detected!** The current network traffic distribution has significantly deviated from the baseline. Model recalibration is highly recommended.")
        else:
            st.success("✅ **Model Stable.** The current traffic distribution remains within acceptable baseline parameters.")
            
    else:
        st.warning("Drift monitoring data not available. Please run `scripts/eval_drift.py`.")
        
    st.markdown("---")
    st.markdown("### Adversarial Robustness")
    adv_file = ROOT / "outputs" / "metrics" / "adversarial_robustness.json"
    
    if adv_file.exists():
        with open(adv_file) as f:
            adv_data = json.load(f)
            
        ac1, ac2, ac3 = st.columns(3)
        ac1.metric("Baseline F1", f"{adv_data['baseline_f1']:.4f}")
        ac2.metric("Noise (Sensor Degradation) F1", f"{adv_data['noise_f1']:.4f}", f"-{adv_data['noise_f1_drop_pct']:.1f}%")
        ac3.metric("Targeted Evasion F1", f"{adv_data['evasion_f1']:.4f}", f"-{adv_data['evasion_f1_drop_pct']:.1f}%")
    else:
        st.warning("Adversarial robustness data not available. Please run `scripts/eval_adversarial.py`.")
