#!/usr/bin/env python3
"""
Engagement Concordance Score - Azure Connected Simplified App
==========================================================

A simplified version that connects to Azure SQL Database for testing.
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
    page_title="ECS Azure Test",
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
st.markdown("### Azure Connected Test Version")

# Database connection function
@st.cache_resource
def get_azure_database_engine():
    """Create and return a SQLAlchemy database engine for Azure SQL Database."""
    try:
        # Database connection parameters from secrets
        server = st.secrets.get("azure_db.server", "ecs-sql-server-engagementmiser.database.windows.net")
        database = st.secrets.get("azure_db.database", "ecs_tweets_db")
        username = st.secrets.get("azure_db.username", "ecsadmin")
        password = st.secrets.get("azure_db.password", "EngagementMiser!")
        
        # Create Azure SQL Database connection string for SQLAlchemy
        conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no&Connection+Timeout=30"
        
        # Create SQLAlchemy engine
        from sqlalchemy import create_engine
        engine = create_engine(conn_str, pool_pre_ping=True, pool_recycle=300)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        
        st.success("✅ Azure SQL Database connection successful!")
        return engine
        
    except Exception as e:
        st.error(f"❌ Azure SQL Database connection failed: {e}")
        st.info("Please check your Azure database credentials in the secrets configuration.")
        return None

# Test database connection
st.header("🔗 Database Connection Test")
if st.button("🧪 Test Azure Connection"):
    with st.spinner("Testing Azure connection..."):
        engine = get_azure_database_engine()
        if engine:
            st.success("🎉 Azure connection working!")
        else:
            st.error("❌ Azure connection failed!")

# Simple data fetch test
st.header("📊 Data Fetch Test")
if st.button("📥 Fetch Sample Data"):
    try:
        engine = get_azure_database_engine()
        if engine:
            # Try to fetch a small sample
            query = "SELECT TOP 5 tweet_id, text FROM [EngagementMiser].[dbo].[Tweets_Sample_4M]"
            df = pd.read_sql(query, engine)
            
            if not df.empty:
                st.success(f"✅ Successfully fetched {len(df)} tweets from Azure!")
                st.dataframe(df)
            else:
                st.warning("⚠️ Query returned no data")
        else:
            st.error("❌ No database connection available")
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")

# Status section
st.header("📊 Status")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Status", "✅ Running")
    
with col2:
    st.metric("Version", "1.0.0")
    
with col3:
    st.metric("Database", "Azure SQL")

# Debug info
st.header("🔧 Debug Information")
st.code(f"""
Python version: {sys.version}
Working directory: {os.getcwd()}
Files in directory: {len(os.listdir('.'))}
""")

st.success("🎯 This app should connect to Azure and deploy quickly!")
