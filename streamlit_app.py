#!/usr/bin/env python3
"""
F1 Race Prediction System - Streamlit Cloud Entry Point
"""
import sys
import os
import streamlit as st

# Add project paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))
sys.path.insert(0, os.path.join(current_dir, 'web'))

# Configure Streamlit page
st.set_page_config(
    page_title="F1 Race Prediction System",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    # Import and run the main app
    from web.app import F1PredictionApp
    
    # Initialize and run the app
    app = F1PredictionApp()
    app.run()
    
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.info("Please ensure all dependencies are installed correctly.")
except Exception as e:
    st.error(f"Application Error: {e}")
    st.info("Please check the application logs for more details.")