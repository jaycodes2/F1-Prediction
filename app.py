#!/usr/bin/env python3
"""
F1 Race Prediction System - Simple Entry Point for Streamlit Cloud
"""
import streamlit as st
import sys
import os

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))
sys.path.insert(0, os.path.join(current_dir, 'web'))

# Configure page
st.set_page_config(
    page_title="F1 Race Prediction System",
    page_icon="🏎️",
    layout="wide"
)

try:
    # Import and run the main app
    from web.app import F1PredictionApp
    
    app = F1PredictionApp()
    app.run()
    
except Exception as e:
    st.error(f"Error: {e}")
    st.info("Please check the application setup.")