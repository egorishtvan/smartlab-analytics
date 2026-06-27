import streamlit as st
import pandas as pd
import os
import sys

# Append project root to path for core module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.feature_extractor import extract_sensor_features
from core.llm_analyzer import analyze_metrics

st.set_page_config(page_title="SmartLab Analytics", layout="wide")
st.title("🔬 SmartLab Analytics Dashboard")

st.sidebar.header("Experiment Settings")
project = st.sidebar.selectbox("Select Project", ["Experiment_Thermodynamics_01"])
sensor = st.sidebar.selectbox("Select Sensor", ["Sensor_Temperature"])

csv_path = f"data/Projects/{project}/Sensors/{sensor}/2026-06-26.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.subheader("📊 Raw Sensor Data Stream")
    st.line_chart(df.set_index("Timestamp"))
    
    if st.button("Run Edge AI Analysis"):
        with st.spinner("Processing: Extracting features and invoking Local LLM..."):
            # Step 1: Feature Extraction
            features = extract_sensor_features(csv_path)
            st.write("### 📐 Extracted Statistical Features (tsfresh profile)", features)
            
            # Step 2: Local AI Inference
            ai_insights = analyze_metrics({}, features)
            st.write("### 🤖 Edge AI Analysis Report", ai_insights)
else:
    st.error(f"Data file {csv_path} not found.")
