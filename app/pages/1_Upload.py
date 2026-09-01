import streamlit as st
import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

st.title("📂 Step 1: Upload & Initialization")
st.markdown("Upload a PCAP/CSV file, or load the offline fallback demo as required by the SIH Definition of Done.")

if st.button("Load Offline Fallback Demo (Known Test Set)"):
    with st.spinner("Loading temporal sequences and models..."):
        df = pd.read_parquet("data/processed/windows/test_known_windows_30s.parquet")
        
        exclude = {'Timestamp', 'Label', 'attack_stage', 'future_attack_1', 'future_attack_3', 'future_attack_5'}
        features = [c for c in df.columns if c not in exclude]
        
        scaler = joblib.load("models/scalers/baseline_scaler.pkl")
        X_scaled = scaler.transform(df[features].fillna(0).values)
        
        st.session_state.raw_df = df
        st.session_state.X_scaled = X_scaled
        st.session_state.feature_names = features
        
        # FIX: Find the first window where an attack occurs, and start 5 windows before it
        # so the judges can watch the forecast risk rise!
        attack_indices = df[df['current_attack_state'] == 1].index
        start_idx = max(30, attack_indices[0] - 5) if len(attack_indices) > 0 else 30
        
        st.session_state.current_idx = start_idx
        st.session_state.demo_loaded = True
        
    st.success(f"✅ Demo Loaded! Auto-skipped to Window {start_idx} (Attack imminent). Proceed to **Network State**.")
