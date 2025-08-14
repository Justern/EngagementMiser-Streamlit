#!/usr/bin/env python3
"""
Engagement Concordance Score - Simple Azure App
==============================================

A simplified version that connects to Azure SQL Database and shows data
without the complex ECS model dependencies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from datetime import datetime
import subprocess
import os
import tempfile

# Page configuration
st.set_page_config(
    page_title="ECS Azure Simple",
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

# ECS Model Scoring Functions
def run_ecs_model(tweet_text, model_name):
    """Run individual ECS model and return score."""
    try:
        # Create a temporary file with the tweet text
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(tweet_text)
            temp_file = f.name
        
        # Run the model script
        result = subprocess.run([
            'python', 'simple_score.py', 
            '--text', temp_file,
            '--model', model_name
        ], capture_output=True, text=True, timeout=30)
        
        # Clean up temp file
        os.unlink(temp_file)
        
        if result.returncode == 0:
            # Parse the score from output
            try:
                score = float(result.stdout.strip().split()[-1])
                return score
            except:
                return 0.0
        else:
            st.warning(f"Model {model_name} failed: {result.stderr}")
            return 0.0
            
    except Exception as e:
        st.error(f"Error running {model_name}: {e}")
        return 0.0

def calculate_ecs_scores(tweet_text):
    """Calculate all 10 ECS model scores."""
    
    # Define the 10 specialized models
    models = {
        'Authority_Signal_Manipulation': 0.12,
        'Clickbait_Headline_Classifier': 0.11,
        'Content_Recycling_Detector': 0.10,
        'Coordinated_Account_Network_Model': 0.10,
        'Emotive_Manipulation_Detector': 0.11,
        'Engagement_Mismatch_Detector': 0.10,
        'Generic_Comment_Detector': 0.09,
        'Hyperbole_Falsehood_detector': 0.10,
        'Rapid_Engagement_Spike_Detector': 0.09,
        'Reply_Bait_Detector': 0.08
    }
    
    st.info("🔍 Running ECS models... This may take a moment.")
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    scores = {}
    total_weight = 0
    weighted_sum = 0
    
    for i, (model_name, weight) in enumerate(models.items()):
        status_text.text(f"Running {model_name}...")
        progress_bar.progress((i + 1) / len(models))
        
        score = run_ecs_model(tweet_text, model_name)
        scores[model_name] = score
        
        total_weight += weight
        weighted_sum += score * weight
        
        time.sleep(0.1)  # Small delay for progress visibility
    
    # Calculate final weighted score
    final_score = weighted_sum / total_weight if total_weight > 0 else 0
    
    progress_bar.empty()
    status_text.empty()
    
    return scores, final_score, models

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
        
        # Azure SQL Database connection string - use generic driver for Linux compatibility
        conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes"
        
        # Create SQLAlchemy engine
        from sqlalchemy import create_engine
        engine = create_engine(conn_str, pool_pre_ping=True, pool_recycle=300)
        
        # Test connection
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text("SELECT 1 as test"))
            st.success("✅ Connected to Azure SQL Database!")
            return engine
            
    except Exception as e:
        st.error(f"❌ Azure database connection failed: {e}")
        st.info("""
        **Troubleshooting:**
        1. Check your Streamlit Cloud secrets have correct Azure credentials
        2. Verify Azure SQL Database is running and accessible
        3. Check firewall rules allow your IP address
        4. Ensure username and password are correct
        """)
        return None

@st.cache_data(ttl=3600)
def fetch_popular_entities(_engine, limit=4000):
    """Fetch top popular entities."""
    try:
        query = f"""
        SELECT TOP {limit} 
            entity_name,
            entity_type,
            frequency,
            engagement_score,
            risk_level
        FROM [dbo].[Popular_Entities_Corpus]
        ORDER BY frequency DESC, engagement_score DESC
        """
        
        df = pd.read_sql(query, _engine)
        st.info(f"📊 Loaded {len(df):,} popular entities")
        return df
        
    except Exception as e:
        st.error(f"❌ Error fetching popular entities: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def fetch_tweets_sample(_engine, sample_size=2000):
    """Fetch random sample of tweets."""
    try:
        query = f"""
        SELECT TOP {sample_size} 
            tweet_id,
            text as tweet_text,
            created_at,
            author_id as user_id,
            retweet_count,
            like_count,
            reply_count,
            quote_count,
            followers_count as follower_count,
            followers_count,  -- Use this instead of verified
            DATEDIFF(day, created_at, GETDATE()) as account_age_days
        FROM [dbo].[Tweets_Sample_4M]
        ORDER BY NEWID()
        """
        
        df = pd.read_sql(query, _engine)
        st.info(f"🐦 Loaded {len(df):,} random tweets")
        return df
        
    except Exception as e:
        st.error(f"❌ Error fetching tweets: {e}")
        return pd.DataFrame()

def show_tweet_selection(engine):
    """Show tweet selection interface."""
    st.subheader("🐦 Select a Tweet for Analysis")
    
    tweets_df = fetch_tweets_sample(engine)
    
    if tweets_df.empty:
        st.warning("No tweets available.")
        return
    
    # Create selection interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_tweet_id = st.selectbox(
            "Choose a tweet to analyze:",
            options=tweets_df['tweet_id'].tolist(),
            format_func=lambda x: f"Tweet {x} - {tweets_df[tweets_df['tweet_id'] == x]['tweet_text'].iloc[0][:100]}..."
        )
    
    with col2:
        if selected_tweet_id:
            selected_tweet = tweets_df[tweets_df['tweet_id'] == selected_tweet_id].iloc[0]
            st.write("**Selected Tweet:**")
            st.write(selected_tweet['tweet_text'])
            
                         if st.button("🔍 Analyze This Tweet"):
                 # Basic tweet metrics
                 st.subheader("📊 Basic Tweet Metrics")
                 
                 col1, col2, col3, col4 = st.columns(4)
                 with col1:
                     st.metric("Likes", selected_tweet['like_count'])
                 with col2:
                     st.metric("Retweets", selected_tweet['retweet_count'])
                 with col3:
                     st.metric("Replies", selected_tweet['reply_count'])
                 with col4:
                     st.metric("Followers", selected_tweet['follower_count'])
                 
                 # Show additional info
                 st.write(f"**Tweet Text:** {selected_tweet['tweet_text']}")
                 st.write(f"**Created:** {selected_tweet['created_at']}")
                 st.write(f"**Account Age:** {selected_tweet['account_age_days']} days")
                 
                 # Simple engagement score
                 engagement = (selected_tweet['like_count'] + selected_tweet['retweet_count'] + selected_tweet['reply_count']) / max(selected_tweet['follower_count'], 1)
                 st.metric("Engagement Rate", f"{engagement:.3f}")
                 
                 # Run ECS Models
                 st.subheader("🔍 ECS Model Analysis")
                 
                 # Run all ECS models
                 scores, final_score, weights = calculate_ecs_scores(selected_tweet['tweet_text'])
                 
                 # Display individual model scores
                 st.subheader("📊 Individual Model Scores")
                 
                 # Create columns for scores
                 col1, col2 = st.columns(2)
                 
                 with col1:
                     for i, (model_name, score) in enumerate(scores.items()):
                         if i < 5:  # First 5 models
                             weight = weights[model_name]
                             st.metric(
                                 f"{model_name} (Weight: {weight:.2f})", 
                                 f"{score:.3f}",
                                 help=f"Score: {score:.3f}, Weight: {weight:.2f}"
                             )
                 
                 with col2:
                     for i, (model_name, score) in enumerate(scores.items()):
                         if i >= 5:  # Last 5 models
                             weight = weights[model_name]
                             st.metric(
                                 f"{model_name} (Weight: {weight:.2f})", 
                                 f"{score:.3f}",
                                 help=f"Score: {score:.3f}, Weight: {weight:.2f}"
                             )
                 
                 # Display final weighted score
                 st.subheader("🏆 Final ECS Score")
                 
                 # Color code based on score
                 if final_score >= 0.7:
                     score_color = "🟢"
                     risk_level = "LOW RISK"
                 elif final_score >= 0.4:
                     score_color = "🟡"
                     risk_level = "MEDIUM RISK"
                 else:
                     score_color = "🔴"
                     risk_level = "HIGH RISK"
                 
                 col1, col2, col3 = st.columns(3)
                 with col1:
                     st.metric("Final ECS Score", f"{final_score:.3f}")
                 with col2:
                     st.metric("Risk Level", risk_level)
                 with col3:
                     st.metric("Confidence", f"{score_color}")
                 
                 # Show calculation breakdown
                 st.subheader("🧮 Score Calculation Breakdown")
                 
                 # Create a DataFrame for better visualization
                 score_df = pd.DataFrame([
                     {'Model': name, 'Score': score, 'Weight': weights[name], 'Weighted Score': score * weights[name]}
                     for name, score in scores.items()
                 ])
                 
                 st.dataframe(score_df, use_container_width=True)
                 
                 # Calculate totals for display
                 total_weighted = sum(score * weights[name] for name, score in scores.items())
                 total_weight = sum(weights.values())
                 
                 # Show formula
                 st.info(f"""
                 **Final Score Formula:**
                 
                 Final ECS Score = Σ(Model Score × Weight) / Σ(Weights)
                 
                 **Your Result:** {final_score:.3f} = {total_weighted:.3f} / {total_weight:.2f}
                 """)

def show_popular_entities_analysis(engine):
    """Show popular entities analysis."""
    st.subheader("📊 Popular Entities Analysis")
    
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
    
    # Show top entities
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

def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">🔍 Engagement Concordance Score</h1>', unsafe_allow_html=True)
             st.markdown("### Azure Connected - Full ECS Model Analysis")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🐦 Tweet Analysis", "📊 Popular Entities"]
    )
    
    # Get database connection
    engine = get_azure_database_engine()
    
    if engine is None:
        st.error("❌ Cannot proceed without database connection.")
        return
    
    # Page routing
    if page == "🏠 Home":
        st.subheader("Welcome to the ECS System")
                 st.write("""
         This is the **full ECS system** that connects to Azure SQL Database and runs all 10 specialized models.
         
         **Features:**
         - 🐦 **Tweet Analysis**: View and analyze individual tweets with full ECS scoring
         - 📊 **Popular Entities**: View trending entities and risk levels
         - 🔍 **10 Specialized Models**: Authority, Clickbait, Content Recycling, Coordinated Networks, Emotive Manipulation, Engagement Mismatch, Generic Comments, Hyperbole/Falsehood, Rapid Engagement, Reply Bait
         
         **Note:** This version includes complete ECS model analysis with weighted scoring.
         """)
        
        # Show system status
        st.subheader("🔄 System Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("✅ Database Connected")
        with col2:
            st.info("📊 Data Available")
                 with col3:
             st.info("🔧 Full ECS Mode")
    
    elif page == "🐦 Tweet Analysis":
        show_tweet_selection(engine)
    
    elif page == "📊 Popular Entities":
        show_popular_entities_analysis(engine)
    
    # Footer
    st.markdown("---")
         st.markdown("**ECS System v1.0** | Azure SQL Database | Full ECS Model Analysis")

if __name__ == "__main__":
    main()
