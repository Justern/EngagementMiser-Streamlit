#!/usr/bin/env python3
"""
Engagement Concordance Score - Streamlit Web App (Azure Optimized)
================================================================

A professional web application for testing and demonstrating the ECS system.
Connects to Azure SQL Database with optimized data loading:
- Popular Entities: Top 4,000 rows only
- Tweets: Random selection of 2,000 tweets
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
    page_title="Engagement Concordance Score - Azure Optimized",
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

# Database connection configuration for AZURE SQL Database
@st.cache_resource
def get_database_engine():
    """Create and return a SQLAlchemy database engine for Azure SQL Database."""
    try:
        # Database connection parameters from secrets (Azure)
        server = st.secrets.get("azure_db.server", "ecs-sql-server-engagementmiser.database.windows.net")
        database = st.secrets.get("azure_db.database", "ecs_tweets_db")
        username = st.secrets.get("azure_db.username", "ecsadmin")
        password = st.secrets.get("azure_db.password", "")
        
        # Azure SQL Database connection string
        conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
        
        # Create SQLAlchemy engine
        from sqlalchemy import create_engine
        engine = create_engine(conn_str, pool_pre_ping=True, pool_recycle=300)
        
        # Test the connection
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text("SELECT 1 as test"))
            st.success("✅ Connected to Azure SQL Database!")
            return engine
            
    except Exception as e:
        st.error(f"❌ Azure database connection failed: {e}")
        st.info("""
        **Troubleshooting:**
        1. Check your `.streamlit/secrets.toml` file has correct Azure credentials
        2. Verify Azure SQL Database is running and accessible
        3. Check firewall rules allow your IP address
        4. Ensure username and password are correct
        """)
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_popular_entities(_engine, limit=4000):
    """Fetch top 4,000 popular entities for optimization."""
    try:
        query = f"""
        SELECT TOP {limit} 
            entity_name,
            entity_type,
            frequency,
            engagement_score,
            risk_level
        FROM [EngagementMiser].[dbo].[Popular_Entities_Corpus]
        ORDER BY frequency DESC, engagement_score DESC
        """
        
        df = pd.read_sql(query, _engine)
        st.info(f"📊 Loaded {len(df):,} popular entities (top {limit:,} by frequency)")
        return df
        
    except Exception as e:
        st.error(f"❌ Error fetching popular entities: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_tweets_sample(_engine, sample_size=2000):
    """Fetch random sample of 2,000 tweets for optimization."""
    try:
        query = f"""
        SELECT TOP {sample_size} 
            tweet_id,
            tweet_text,
            created_at,
            user_id,
            retweet_count,
            like_count,
            reply_count,
            quote_count,
            follower_count,
            verified,
            account_age_days
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M]
        ORDER BY NEWID()  -- Random selection
        """
        
        df = pd.read_sql(query, _engine)
        st.info(f"🐦 Loaded {len(df):,} random tweets (sample of {sample_size:,})")
        return df
        
    except Exception as e:
        st.error(f"❌ Error fetching tweets sample: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_tweet_by_id(_engine, tweet_id):
    """Fetch a specific tweet by ID."""
    try:
        query = """
        SELECT 
            tweet_id,
            tweet_text,
            created_at,
            user_id,
            retweet_count,
            like_count,
            reply_count,
            quote_count,
            follower_count,
            verified,
            account_age_days
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M]
        WHERE tweet_id = :tweet_id
        """
        
        df = pd.read_sql(query, _engine, params={'tweet_id': tweet_id})
        return df
        
    except Exception as e:
        st.error(f"❌ Error fetching tweet: {e}")
        return pd.DataFrame()

def analyze_tweet_with_ecs(tweet_text, tweet_id=None):
    """Analyze a tweet using the ECS system."""
    try:
        # Import ECS system
        from deployment_config import DeploymentModels
        
        # Initialize models
        models = DeploymentModels()
        
        # Run all 10 specialized models
        st.subheader("🔍 Running ECS Analysis...")
        
        # Progress bar for analysis
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        models_to_run = [
            ('hyperbole_falsehood_score', 'Hyperbole & Falsehood'),
            ('clickbait_score', 'Clickbait Detection'),
            ('emotive_manipulation_score', 'Emotive Manipulation'),
            ('authority_signal_score', 'Authority Signal'),
            ('rapid_engagement_spike_score', 'Rapid Engagement Spike'),
            ('coordinated_network_score', 'Coordinated Network'),
            ('engagement_mismatch_score', 'Engagement Mismatch'),
            ('content_recycling_score', 'Content Recycling'),
            ('generic_comment_score', 'Generic Comment'),
            ('reply_bait_score', 'Reply Bait')
        ]
        
        # Weights for each model (sum should be 1.0)
        model_weights = {
            'Hyperbole & Falsehood': 0.15,
            'Clickbait Detection': 0.15,
            'Emotive Manipulation': 0.12,
            'Authority Signal': 0.10,
            'Rapid Engagement Spike': 0.10,
            'Coordinated Network': 0.08,
            'Engagement Mismatch': 0.08,
            'Content Recycling': 0.07,
            'Generic Comment': 0.08,
            'Reply Bait': 0.07
        }
        
        st.subheader("Engagement Concordance Score (ECS) Breakdown")
        
        scores = {}
        total_weighted_score = 0.0
        
        for i, (method_name, display_name) in enumerate(models_to_run):
            try:
                # Update progress
                progress = (i + 1) / len(models_to_run)
                progress_bar.progress(progress)
                status_text.text(f"Running {display_name}...")
                
                # Get the method from models instance
                method = getattr(models, method_name)
                
                # Call the method with tweet_id if available, otherwise with tweet_text
                if tweet_id is not None:
                    score = method(tweet_id)
                else:
                    score = method(tweet_text)
                
                scores[display_name] = score
                weighted_score = score * model_weights[display_name]
                total_weighted_score += weighted_score
                
                # Display individual model score
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{display_name}**")
                with col2:
                    st.write(f"{score:.3f}")
                with col3:
                    st.write(f"Weight: {model_weights[display_name]:.2f}")
                
                # Add a small delay for visual feedback
                time.sleep(0.1)
                
            except Exception as e:
                st.error(f"❌ Error in {display_name}: {e}")
                scores[display_name] = 0.0
        
        # Complete progress bar
        progress_bar.progress(1.0)
        status_text.text("Analysis complete!")
        
        # Calculate final ECS score
        st.subheader("🎯 Final ECS Score")
        
        # Determine risk level
        if total_weighted_score >= 0.7:
            risk_level = "High Risk"
            risk_class = "risk-high"
        elif total_weighted_score >= 0.4:
            risk_level = "Medium Risk"
            risk_class = "risk-medium"
        else:
            risk_level = "Low Risk"
            risk_class = "risk-low"
        
        # Display final score
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="text-align: center; color: #1f77b4;">{total_weighted_score:.3f}</h2>
                <p style="text-align: center; font-size: 1.2rem;">ECS Score</p>
                <p style="text-align: center; font-size: 1.1rem;" class="{risk_class}">{risk_level}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display breakdown chart
        st.subheader("📊 Model Score Breakdown")
        
        # Create bar chart
        fig = px.bar(
            x=list(scores.keys()),
            y=list(scores.values()),
            title="Individual Model Scores",
            labels={'x': 'Model', 'y': 'Score'},
            color=list(scores.values()),
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        return total_weighted_score, scores
        
    except Exception as e:
        st.error(f"❌ Error in ECS analysis: {e}")
        return 0.0, {}

def show_tweet_selection(engine):
    """Show tweet selection interface."""
    st.subheader("🐦 Select a Tweet for Analysis")
    
    # Fetch sample tweets
    tweets_df = fetch_tweets_sample(engine)
    
    if tweets_df.empty:
        st.warning("No tweets available for selection.")
        return
    
    # Create a selection interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Show tweet previews
        selected_tweet_id = st.selectbox(
            "Choose a tweet to analyze:",
            options=tweets_df['tweet_id'].tolist(),
            format_func=lambda x: f"Tweet {x} - {tweets_df[tweets_df['tweet_id'] == x]['tweet_text'].iloc[0][:100]}..."
        )
    
    with col2:
        if selected_tweet_id:
            selected_tweet = fetch_tweet_by_id(engine, selected_tweet_id)
            if selected_tweet is not None and not selected_tweet.empty:
                st.write("**Selected Tweet:**")
                st.write(selected_tweet['tweet_text'].iloc[0])
                
                if st.button("🔍 Analyze This Tweet"):
                    analyze_tweet_with_ecs(
                        selected_tweet['tweet_text'].iloc[0], 
                        selected_tweet_id
                    )
            else:
                st.error("❌ Could not fetch tweet data")

def show_popular_entities_analysis(engine):
    """Show popular entities analysis."""
    st.subheader("📊 Popular Entities Analysis")
    
    # Fetch popular entities
    entities_df = fetch_popular_entities(engine)
    
    if entities_df.empty:
        st.warning("No popular entities data available.")
        return
    
    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Entities", f"{len(entities_df):,}")
    with col2:
        st.metric("High Risk", f"{len(entities_df[entities_df['risk_level'] == 'High']):,}")
    with col3:
        st.metric("Medium Risk", f"{len(entities_df[entities_df['risk_level'] == 'Medium']):,}")
    with col4:
        st.metric("Low Risk", f"{len(entities_df[entities_df['risk_level'] == 'Low']):,}")
    
    # Show top entities by frequency
    st.subheader("🏆 Top Entities by Frequency")
    top_entities = entities_df.head(20)
    
    fig = px.bar(
        top_entities,
        x='entity_name',
        y='frequency',
        title="Top 20 Entities by Frequency",
        labels={'entity_name': 'Entity', 'frequency': 'Frequency'},
        color='risk_level',
        color_discrete_map={'High': '#d62728', 'Medium': '#ff7f0e', 'Low': '#2ca02c'}
    )
    fig.update_layout(xaxis_tickangle=-45, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show entities by risk level
    st.subheader("⚠️ Risk Level Distribution")
    risk_counts = entities_df['risk_level'].value_counts()
    
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Entities by Risk Level"
    )
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">🔍 Engagement Concordance Score</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🐦 Tweet Analysis", "📊 Popular Entities", "🔧 Model Testing"]
    )
    
    # Get database connection
    engine = get_database_engine()
    
    if engine is None:
        st.error("❌ Cannot proceed without database connection.")
        st.info("Please check your Azure database configuration and try again.")
        return
    
    # Page routing
    if page == "🏠 Home":
        st.subheader("Welcome to the ECS System")
        st.write("""
        This application demonstrates the Engagement Concordance Score (ECS) system, 
        which analyzes social media content for various forms of engagement manipulation.
        
        **Features:**
        - 🐦 **Tweet Analysis**: Analyze individual tweets with the ECS system
        - 📊 **Popular Entities**: View analysis of trending entities and their risk levels
        - 🔧 **Model Testing**: Test individual ECS models
        
        **Optimizations Applied:**
        - Popular Entities: Limited to top 4,000 rows for performance
        - Tweets: Random sample of 2,000 tweets for faster loading
        - Other Tables: Complete data uploaded for full analysis capabilities
        """)
        
        # Show system status
        st.subheader("🔄 System Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("✅ Database Connected")
        with col2:
            st.info("📊 Data Optimized")
        with col3:
            st.success("🔧 Models Ready")
    
    elif page == "🐦 Tweet Analysis":
        show_tweet_selection(engine)
    
    elif page == "📊 Popular Entities":
        show_popular_entities_analysis(engine)
    
    elif page == "🔧 Model Testing":
        st.subheader("Model Testing Interface")
        st.write("Use the Tweet Analysis page to test the ECS models with real data.")
    
    # Footer
    st.markdown("---")
    st.markdown("**ECS System v1.0** | Azure SQL Database | Optimized Data Loading")

if __name__ == "__main__":
    main()
