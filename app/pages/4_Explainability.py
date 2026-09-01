import streamlit as st
import plotly.express as px
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.inference.explain import explain_prediction

st.title("ðŸ§  Step 4: Explainability (Captum)")

if not st.session_state.get('demo_loaded', False) or 'seq_tensor' not in st.session_state:
    st.warning("Please run the Forecast page first to generate a prediction.")
    st.stop()

st.markdown("Using **Integrated Gradients** to identify which features influenced the GRU's forecast risk.")

with st.spinner("Calculating gradients..."):
    importance = explain_prediction(
        st.session_state.model, 
        st.session_state.seq_tensor, 
        st.session_state.feature_names, 
        device='cpu'
    )
    
df_imp = pd.DataFrame(importance)

fig = px.bar(
    df_imp, x="Contribution", y="Feature", orientation='h',
    title="Top 5 Driving Features for Alert",
    color="Contribution", color_continuous_scale="Reds"
)
fig.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig, use_container_width=True)

st.table(df_imp)
