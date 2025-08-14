#!/usr/bin/env python3
"""
Engagement Concordance Score - Simplified Streamlit App
====================================================

A simplified version to test deployment and identify structural issues.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
import sys

# Page configuration
st.set_page_config(
    page_title="ECS Simplified",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">🔍 Engagement Concordance Score</div>', unsafe_allow_html=True)
st.markdown("### Simplified Test Version")

# Simple test section
st.header("🚀 Quick Test")
st.write("This is a simplified version to test deployment.")

# Test button
if st.button("🧪 Run Test"):
    with st.spinner("Running test..."):
        time.sleep(2)
        st.success("✅ Test completed successfully!")
        st.info("🎉 App is working! Deployment successful.")

# Status section
st.header("📊 Status")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Status", "✅ Running")
    
with col2:
    st.metric("Version", "1.0.0")
    
with col3:
    st.metric("Deployment", "Simplified")

# Debug info
st.header("🔧 Debug Information")
st.code(f"""
Python version: {sys.version}
Working directory: {os.getcwd()}
Files in directory: {len(os.listdir('.'))}
""")

st.success("🎯 This simplified app should deploy much faster!")
