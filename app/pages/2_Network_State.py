import streamlit as st
import plotly.graph_objects as go

st.title("ðŸŒ Step 2: Current Network State")

if not st.session_state.get('demo_loaded', False):
    st.warning("Please load data in the Upload page first.")
    st.stop()

idx = st.session_state.current_idx
raw_df = st.session_state.raw_df

# UI Controls to step through time
col1, col2 = st.columns(2)
with col1:
    if st.button("âª Previous Window") and idx > 30:
        st.session_state.current_idx -= 1
        st.rerun()
with col2:
    if st.button("Fast Forward 10 Windows â©") and idx < len(raw_df) - 10:
        st.session_state.current_idx += 10
        st.rerun()

current_window = raw_df.iloc[st.session_state.current_idx]

st.subheader("Current Traffic Snapshot")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Host Count (Dest IPs)", int(current_window.get('unique_destination_ips', 0)))
c2.metric("Active Flows", int(current_window.get('flow_count', 0)))
c3.metric("SYN-to-ACK Ratio", round(current_window.get('syn_to_ack_ratio', 0), 2))
c4.metric("Current True Label", current_window.get('Label', 'BENIGN'))

# Gauge Chart for basic visualization
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = current_window.get('flow_count', 0),
    title = {'text': "Current Flow Volume"},
    gauge = {'axis': {'range': [None, 500]}, 'bar': {'color': "darkblue"}}
))
st.plotly_chart(fig, use_container_width=True)
