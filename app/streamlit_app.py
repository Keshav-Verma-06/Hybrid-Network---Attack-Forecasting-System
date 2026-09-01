import streamlit as st
import sys
import os

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="NetSentinel World Model", page_icon="ðŸ›¡ï¸", layout="wide")

st.title("ðŸ›¡ï¸ NetSentinel: Offline Network-Attack Forecaster")
st.markdown("""
Welcome to the **NetSentinel World Model** prototype. 
This system uses a Temporal GRU + OCSVM Hybrid architecture to forecast network attacks **before** they happen and detect **Zero-Day** threats.

ðŸ‘ˆ **Please use the sidebar to navigate through the 5 modules.**
""")

if 'demo_loaded' not in st.session_state:
    st.session_state.demo_loaded = False
