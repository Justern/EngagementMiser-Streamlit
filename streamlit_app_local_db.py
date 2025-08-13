#!/usr/bin/env python3
"""
Engagement Concordance Score - Streamlit Web App (Local Database)
==============================================================

A professional web application for testing and demonstrating the ECS system.
Connects directly to your local MSSQL database through ngrok tunnel.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import json
import os
import sys
import subprocess
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for ECS system
sys.path.append('../../')

# Constants
RISK_LEVELS = ["Low Risk", "Medium Risk", "High Risk"]

# Page configuration
st.set_page_config(
    page_title="Engagement Concordance Score - Local DB",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-score {
        background-color: #1DA1F2;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high { color: #d62728; font-weight: bold; }
    .risk-medium { color: #ff7f0e; font-weight: bold; }
    .risk-low { color: #2ca02c; font-weight: bold; }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Database connection configuration for LOCAL database through ngrok
@st.cache_resource
def get_database_engine():
    """Create and return a SQLAlchemy database engine for LOCAL MSSQL through ngrok."""
    try:
        # Database connection parameters from secrets (now pointing to local DB)
        server = st.secrets.get("local_db.server", "localhost")
        database = st.secrets.get("local_db.database", "EngagementMiser")
        username = st.secrets.get("local_db.username", "")
        password = st.secrets.get("local_db.password", "")
        
        # Check if we're using Windows Authentication or SQL Authentication
        if username and password:
            # SQL Authentication
            conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes"
        else:
            # Windows Authentication (recommended for local)
            conn_str = f"mssql+pyodbc://@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&Trusted_Connection=yes&TrustServerCertificate=yes"
        
        # Create SQLAlchemy engine
        from sqlalchemy import create_engine
        engine = create_engine(conn_str, pool_pre_ping=True, pool_recycle=300)
        
        # Test the connection
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text("SELECT 1 as test"))
            st.success("✅ Connected to local MSSQL database through ngrok!")
        
        return engine
        
    except Exception as e:
        st.error(f"❌ Local database connection failed: {e}")
        st.info("""
        **Troubleshooting:**
        1. Make sure ngrok is running: `ngrok tcp 1433`
        2. Check your secrets.toml file has correct local_db settings
        3. Verify your local SQL Server is running
        4. Check Windows Firewall allows connections on port 1433
        """)
        return None

def fetch_tweets_from_database(limit=1020):
    """Fetch recent tweets from the LOCAL database for selection."""
    try:
        engine = get_database_engine()
        if not engine:
            return pd.DataFrame()
        
        # Query your local Tweets_Sample_4M table
        query = f"""
        SELECT TOP ({limit}) 
            tweet_id, 
            text as tweet_text,  -- Note: local table has 'text' column
            author_id, 
            created_at, 
            retweet_count, 
            like_count, 
            followers_count,
            total_engagements,
            engagement_rate,
            hashtag1,
            hashtag2,
            hashtag3
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M]
        ORDER BY created_at DESC
        """
        
        df = pd.read_sql(query, engine)
        st.success(f"✅ Fetched {len(df)} tweets from local database")
        return df
        
    except Exception as e:
        st.error(f"❌ Error fetching tweets: {e}")
        return pd.DataFrame()

def fetch_tweet_by_id(tweet_id):
    """Fetch a specific tweet by ID from the LOCAL database."""
    try:
        engine = get_database_engine()
        if not engine:
            return None
        
        # Query your local Tweets_Sample_4M table
        query = f"""
        SELECT 
            tweet_id, 
            text as tweet_text,  -- Note: local table has 'text' column
            author_id, 
            created_at, 
            retweet_count, 
            like_count, 
            followers_count,
            total_engagements,
            engagement_rate,
            hashtag1,
            hashtag2,
            hashtag3
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M]
        WHERE tweet_id = {tweet_id}
        """
        
        df = pd.read_sql(query, engine)
        if not df.empty:
            return df.iloc[0]
        return None
        
    except Exception as e:
        st.error(f"❌ Error fetching tweet {tweet_id}: {e}")
        return None

def show_tweet_selection():
    """Show dropdown for selecting tweets from the database."""
    st.subheader("📊 Select Tweet from Database")
    
    # Fetch tweets from database
    tweets_df = fetch_tweets_from_database(limit=1020)
    
    if tweets_df.empty:
        st.warning("No tweets found in database. Please check your connection.")
        return None
    
    # Create display text for dropdown
    tweets_df['display_text'] = tweets_df['tweet_text'].astype(str).str[:80] + "..."
    tweets_df['option_text'] = tweets_df.apply(
        lambda row: f"ID: {row['tweet_id']} | {row['display_text']}", axis=1
    )
    
    # Show tweet selection dropdown
    selected_tweet_option = st.selectbox(
        "Choose a tweet to analyze:",
        options=tweets_df['option_text'].tolist(),
        index=0
    )
    
    if selected_tweet_option:
        # Extract tweet ID from selection
        selected_tweet_id = int(selected_tweet_option.split(" | ")[0].replace("ID: ", ""))
        
        # Get full tweet data
        selected_tweet = fetch_tweet_by_id(selected_tweet_id)
        
        if selected_tweet is not None:
            st.success(f"✅ Selected Tweet ID: {selected_tweet_id}")
            
            # Display tweet details
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Tweet Text:**")
                st.write(selected_tweet['tweet_text'])
                
                st.write("**Author ID:**", selected_tweet['author_id'])
                st.write("**Created:**", selected_tweet['created_at'])
            
            with col2:
                st.write("**Engagement Metrics:**")
                st.write("Retweets:", selected_tweet['retweet_count'])
                st.write("Likes:", selected_tweet['like_count'])
                st.write("Followers:", selected_tweet['followers_count'])
                
                if selected_tweet['total_engagements'] is not None:
                    st.write("Total Engagements:", selected_tweet['total_engagements'])
                
                if selected_tweet['engagement_rate'] is not None:
                    st.write("Engagement Rate:", f"{selected_tweet['engagement_rate']:.4f}")
            
            return selected_tweet_id
    
    return None

# Import the ECS system
try:
    from deployment_config import DeploymentModels
    ecs_system = DeploymentModels()
    st.sidebar.success("✅ ECS System Loaded")
except Exception as e:
    st.sidebar.error(f"❌ ECS System Error: {e}")
    ecs_system = None

def analyze_tweet_with_ecs(tweet_id):
    """Analyze a tweet using the ECS system."""
    if not ecs_system:
        st.error("❌ ECS system not available")
        return None
    
    try:
        st.info(f"🔍 Starting ECS analysis for Tweet ID: {tweet_id}")
        
        # Get tweet text from database
        tweet_data = fetch_tweet_by_id(tweet_id)
        if not tweet_data:
            st.error("❌ Could not fetch tweet data")
            return None
        
        tweet_text = tweet_data['tweet_text']
        st.info(f"📝 Analyzing text: '{tweet_text[:100]}...'")
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run all 10 models
        model_scores = {}
        models_to_run = [
            ('hyperbole_falsehood_score', 'Hyperbole & Falsehood'),
            ('clickbait_score', 'Clickbait Detection'),
            ('emotional_manipulation_score', 'Emotional Manipulation'),
            ('authority_appeal_score', 'Authority Appeal'),
            ('urgency_score', 'Urgency & Scarcity'),
            ('social_proof_score', 'Social Proof'),
            ('reciprocity_score', 'Reciprocity'),
            ('commitment_score', 'Commitment & Consistency'),
            ('liking_score', 'Liking & Similarity'),
            ('scarcity_score', 'Scarcity & Loss Aversion')
        ]
        
        for i, (model_method, model_display_name) in enumerate(models_to_run):
            try:
                # Update progress
                progress = (i + 1) / len(models_to_run)
                progress_bar.progress(progress)
                status_text.text(f"🔄 Running {model_display_name}...")
                
                # Call the model method
                method = getattr(ecs_system, model_method)
                score = method(tweet_text)
                
                model_scores[model_display_name] = score
                
                # Show success for this model with color coding
                if score < 0.4:
                    st.success(f"✅ {model_display_name}: {score:.3f} (Low Risk)")
                elif score < 0.7:
                    st.warning(f"⚠️ {model_display_name}: {score:.3f} (Medium Risk)")
                else:
                    st.error(f"🚨 {model_display_name}: {score:.3f} (High Risk)")
                
                # Small delay to make progress visible
                time.sleep(0.1)
                
            except Exception as e:
                st.error(f"❌ Error in {model_display_name}: {e}")
                model_scores[model_display_name] = 0.0
        
        # Final progress update
        progress_bar.progress(1.0)
        status_text.text("🎯 All models completed! Calculating composite score...")
        
        # Clear progress indicators after a moment
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        st.info(f"📊 All models completed. Scores: {model_scores}")
        
        # Calculate composite score (weighted average)
        weights = {
            'Hyperbole & Falsehood': 0.15,
            'Clickbait Detection': 0.15,
            'Emotional Manipulation': 0.12,
            'Authority Appeal': 0.10,
            'Urgency & Scarcity': 0.10,
            'Social Proof': 0.08,
            'Reciprocity': 0.08,
            'Commitment & Consistency': 0.07,
            'Liking & Similarity': 0.08,
            'Scarcity & Loss Aversion': 0.07
        }
        
        composite_score = sum(
            model_scores[model_name] * weights[model_name]
            for model_name in weights.keys()
            if model_name in model_scores
        )
        
        # Determine risk level
        if composite_score < 0.4:
            risk_level = "Low Risk"
            risk_color = "green"
        elif composite_score < 0.7:
            risk_level = "Medium Risk"
            risk_color = "orange"
        else:
            risk_level = "High Risk"
            risk_color = "red"
        
        # Display results
        st.markdown("---")
        st.markdown(f"## 🎯 **Composite ECS Score: {composite_score:.3f}**")
        st.markdown(f"### 🚨 **Risk Level: {risk_level}**")
        
        # Create detailed results DataFrame
        results_df = pd.DataFrame([
            {
                'Model': model_name,
                'Score': score,
                'Weight': weights[model_name],
                'Weighted Score': score * weights[model_name]
            }
            for model_name, score in model_scores.items()
        ])
        
        st.dataframe(results_df, use_container_width=True)
        
        return {
            'composite_score': composite_score,
            'risk_level': risk_level,
            'model_scores': model_scores
        }
        
    except Exception as e:
        st.error(f"❌ Error in ECS analysis: {e}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        return None

def show_live_analysis():
    """Show the live tweet analysis interface."""
    st.header("🔍 Live Tweet Analysis")
    st.write("Analyze tweets from your local database using the ECS system.")
    
    # Tweet selection
    selected_tweet_id = show_tweet_selection()
    
    if selected_tweet_id:
        st.markdown("---")
        
        # Analysis button
        if st.button("🚀 Analyze Tweet with ECS", type="primary"):
            with st.spinner("Running ECS analysis..."):
                results = analyze_tweet_with_ecs(selected_tweet_id)
                
                if results:
                    st.success("🎉 Analysis completed!")
                    
                    # Show additional insights
                    st.markdown("### 📊 Analysis Insights")
                    
                    # Find highest scoring models
                    top_models = sorted(
                        results['model_scores'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                    
                    st.write("**Top Risk Indicators:**")
                    for model_name, score in top_models:
                        if score > 0.5:
                            st.warning(f"⚠️ {model_name}: {score:.3f}")
                        else:
                            st.info(f"ℹ️ {model_name}: {score:.3f}")

def show_model_performance():
    """Show model performance metrics."""
    st.header("📊 Model Performance")
    st.write("ECS system performance and metrics will be displayed here.")
    
    # Placeholder for model performance metrics
    st.info("Model performance metrics coming soon...")

def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">🔍 Engagement Concordance Score</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🔍 Live Tweet Analysis", "📊 Model Performance"]
    )
    
    # Sidebar status
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    
    # Check database connection
    engine = get_database_engine()
    if engine:
        st.sidebar.success("✅ Database Connected")
        st.sidebar.info("🌐 Connected to local MSSQL through ngrok")
    else:
        st.sidebar.error("❌ Database Disconnected")
    
    # Check ECS system
    if ecs_system:
        st.sidebar.success("✅ ECS System Ready")
    else:
        st.sidebar.error("❌ ECS System Error")
    
    # Page routing
    if page == "🔍 Live Tweet Analysis":
        show_live_analysis()
    elif page == "📊 Model Performance":
        show_model_performance()

if __name__ == "__main__":
    main()
