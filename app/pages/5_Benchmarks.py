"""
app/pages/5_Benchmarks.py
--------------------------
Step 5: Benchmarks & Performance Comparison

Displays full suite of evaluation metrics dynamically loaded from JSON/CSV:
  - Multi-horizon F1 chart
  - Overall Benchmark Table
  - Confusion Matrices
  - Per-Attack Family F1
  - Per-Stage Detection Rate
  - Detection Lead Time
  - Holdout Family (Zero-Day) Evaluation
"""

import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Step 5: Benchmarks", page_icon="📊", layout="wide")
st.title("📊 Benchmarks & Performance Comparison")

METRICS_DIR = ROOT / "outputs" / "metrics"

@st.cache_data
def load_all_metrics():
    # Baselines
    b_df = pd.DataFrame()
    if (METRICS_DIR / "baseline_metrics_table.csv").exists():
        b_df = pd.read_csv(METRICS_DIR / "baseline_metrics_table.csv")
    
    # GRU & Hybrid
    gru_m, hyb_m = {}, {}
    if (METRICS_DIR / "gru_metrics.json").exists():
        gru_m = json.load(open(METRICS_DIR / "gru_metrics.json"))
    if (METRICS_DIR / "hybrid_metrics.json").exists():
        hyb_m = json.load(open(METRICS_DIR / "hybrid_metrics.json"))
        
    # Per-attack & Per-stage
    attack_df = pd.DataFrame()
    if (METRICS_DIR / "per_attack_evaluation.csv").exists():
        attack_df = pd.read_csv(METRICS_DIR / "per_attack_evaluation.csv")
        
    stage_df = pd.DataFrame()
    if (METRICS_DIR / "per_stage_evaluation.csv").exists():
        stage_df = pd.read_csv(METRICS_DIR / "per_stage_evaluation.csv")
        
    # Lead time
    lead_time = {}
    if (METRICS_DIR / "lead_time_results.json").exists():
        lead_time = json.load(open(METRICS_DIR / "lead_time_results.json"))
        
    # Holdout family
    holdout = {}
    if (METRICS_DIR / "holdout_family_evaluation.json").exists():
        holdout = json.load(open(METRICS_DIR / "holdout_family_evaluation.json"))

    return b_df, gru_m, hyb_m, attack_df, stage_df, lead_time, holdout

b_df, gru_m, hyb_m, attack_df, stage_df, lead_time, holdout = load_all_metrics()

if hyb_m == {}:
    st.warning("Hybrid metrics not found. Please run the training pipeline.")
    st.stop()


# ── TAB 1: Multi-Horizon & Overall ───────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Overall Benchmarks", "Per-Attack & Stage", "Detection Lead Time", "Zero-Day (Holdout)"])

with tab1:
    st.markdown("### 1. Forecast Horizon Degradation (F1-Score)")
    st.caption("How well does the model predict attacks 30s, 90s, and 150s into the future?")
    
    horizons = ["K=1", "K=3", "K=5"]
    fig = go.Figure()
    
    # GRU
    gru_known = [gru_m.get("known_test", {}).get(k, {}).get("F1-score", 0) for k in horizons]
    gru_unkn  = [gru_m.get("unknown_test", {}).get(k, {}).get("F1-score", 0) for k in horizons]
    
    # Hybrid
    hyb_known = [hyb_m.get("hybrid", {}).get("known_test", {}).get(k, {}).get("F1-score", 0) for k in horizons]
    hyb_unkn  = [hyb_m.get("hybrid", {}).get("unknown_test", {}).get(k, {}).get("F1-score", 0) for k in horizons]

    fig.add_trace(go.Scatter(x=horizons, y=hyb_known, mode='lines+markers', name='Hybrid (Known)', line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=horizons, y=hyb_unkn,  mode='lines+markers', name='Hybrid (Unknown)', line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter(x=horizons, y=gru_known, mode='lines+markers', name='GRU Only (Known)', line=dict(color='lightblue', dash='dash')))
    fig.add_trace(go.Scatter(x=horizons, y=gru_unkn,  mode='lines+markers', name='GRU Only (Unknown)', line=dict(color='pink', dash='dash')))

    fig.update_layout(yaxis_title="F1 Score", yaxis_range=[0, 1.05], height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 2. Full Performance Table (K=1)")
    
    # Build combined table
    from src.utils.metrics import format_metrics_row
    rows = []
    if not b_df.empty:
        rows = b_df.to_dict('records')
        
    rows.append(format_metrics_row(
        "GRU World Model", 
        gru_m.get("known_test", {}).get("K=1", {}), 
        gru_m.get("unknown_test", {}).get("K=1", {})
    ))
    rows.append(format_metrics_row(
        "GRU + OCSVM Hybrid", 
        hyb_m.get("hybrid", {}).get("known_test", {}).get("K=1", {}), 
        hyb_m.get("hybrid", {}).get("unknown_test", {}).get("K=1", {})
    ))
    
    combo_df = pd.DataFrame(rows)
    # Reorder columns slightly for better reading
    cols = ["Model", "Known F1", "Unknown F1", "Precision (Known)", "Recall (Known)", "FPR", "PR-AUC (Known)"]
    st.dataframe(combo_df[cols].style.highlight_max(subset=["Known F1", "Unknown F1", "PR-AUC (Known)"], axis=0, color='lightgreen')
                                    .highlight_min(subset=["FPR"], axis=0, color='lightgreen'))

    st.markdown("### 3. Confusion Matrices (Hybrid, K=1)")
    col1, col2 = st.columns(2)
    
    def plot_cm(cm_dict, title):
        if not cm_dict: return None
        z = [[cm_dict["TN"], cm_dict["FP"]], [cm_dict["FN"], cm_dict["TP"]]]
        fig = px.imshow(z, text_auto=True, labels=dict(x="Predicted", y="True", color="Count"),
                        x=['Benign', 'Attack'], y=['Benign', 'Attack'], color_continuous_scale='Blues')
        fig.update_layout(title=title, height=350)
        return fig

    cm_k = hyb_m.get("hybrid", {}).get("known_test", {}).get("K=1", {}).get("confusion_matrix_values")
    cm_u = hyb_m.get("hybrid", {}).get("unknown_test", {}).get("K=1", {}).get("confusion_matrix_values")
    
    if cm_k: col1.plotly_chart(plot_cm(cm_k, "Known Test Set"), use_container_width=True)
    if cm_u: col2.plotly_chart(plot_cm(cm_u, "Unknown Test Set"), use_container_width=True)


# ── TAB 2: Per-Attack & Stage ────────────────────────────────────────────────
with tab2:
    st.markdown("### Per-Attack Family Performance (Hybrid)")
    if not attack_df.empty:
        st.dataframe(attack_df.style.background_gradient(subset=['F1'], cmap='Blues'))
        
        fig = px.bar(attack_df, x="Attack", y="F1", color="Split", barmode="group",
                     title="F1 Score by Attack Family")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Per-attack evaluation not found. Run `scripts/eval_per_class.py`.")

    st.markdown("---")
    st.markdown("### Per-MITRE Stage Detection Rate")
    if not stage_df.empty:
        st.dataframe(stage_df.style.background_gradient(subset=['Detection_Rate'], cmap='Greens'))
    else:
        st.info("Per-stage evaluation not found.")


# ── TAB 3: Detection Lead Time ───────────────────────────────────────────────
with tab3:
    st.markdown("### Early Warning: Detection Lead Time")
    st.caption("How many seconds before the first true compromise does the model raise an alert?")
    
    if lead_time:
        col1, col2 = st.columns(2)
        
        def display_lt(data, title, col):
            col.markdown(f"**{title}**")
            col.metric("Attack Episodes Detected", f"{data['n_episodes']}")
            if data['mean_lead_sec'] is not None:
                col.metric("Mean Lead Time", f"{data['mean_lead_sec']:.1f}s")
                col.metric("Max Lead Time (Earliest Warning)", f"{data['max_lead_sec']:.1f}s")
                
                # Histogram
                fig = px.histogram(data["lead_times_sec"], nbins=10, 
                                   labels={"value": "Lead Time (seconds)"},
                                   title="Distribution of Lead Times")
                fig.update_layout(showlegend=False, height=300)
                col.plotly_chart(fig, use_container_width=True)
            else:
                col.warning("No pre-attack alarms found (lead time = 0s)")

        display_lt(lead_time.get("Known", {}), "Known Test Set", col1)
        display_lt(lead_time.get("Unknown", {}), "Unknown Test Set", col2)
    else:
        st.info("Lead time analysis not found. Run `scripts/eval_lead_time.py`.")


# ── TAB 4: Zero-Day (Holdout) ────────────────────────────────────────────────
with tab4:
    st.markdown("### Zero-Day Generalisation (Strict Holdout)")
    st.caption("Performance on attack families completely removed from the training set.")
    
    if holdout:
        st.info(f"**Held-out families:** {', '.join(holdout.get('held_out_attacks', []))}")
        
        ov = holdout.get("overall", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall F1", f"{ov.get('F1-score', 0):.4f}")
        col2.metric("Precision", f"{ov.get('Precision', 0):.4f}")
        col3.metric("Recall", f"{ov.get('Recall', 0):.4f}")
        col4.metric("PR-AUC", f"{ov.get('PR-AUC', 0):.4f}")
        
        per_fam = pd.DataFrame(holdout.get("per_family", []))
        if not per_fam.empty:
            st.markdown("#### Breakdown by Family")
            st.dataframe(per_fam)
    else:
        st.info("Holdout evaluation not found. Run `scripts/eval_holdout.py`.")
