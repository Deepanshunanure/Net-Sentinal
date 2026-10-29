
import streamlit as st

st.set_page_config(page_title="NetSentinel Test", layout="wide")

st.title("🛡️ NetSentinel")
st.success("Streamlit is working!")

col1, col2, col3 = st.columns(3)
col1.metric("RBM", "Ready")
col2.metric("MRF", "Ready")
col3.metric("Bayesian Network", "Ready")

st.write("Dashboard rendering test successful.")
