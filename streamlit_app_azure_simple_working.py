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
        FROM [EngagementMiser].[dbo].[Popular_Entities_Corpus]
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
                # Simple analysis without complex models
                st.subheader("📊 Tweet Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Likes", selected_tweet['like_count'])
                with col2:
                    st.metric("Retweets", selected_tweet['retweet_count'])
                with col3:
                    st.metric("Replies", selected_tweet['reply_count'])
                with col4:
                    st.metric("Followers", selected_tweet['follower_count'])
                
                # Simple engagement score
                engagement = (selected_tweet['like_count'] + selected_tweet['retweet_count'] + selected_tweet['reply_count']) / max(selected_tweet['follower_count'], 1)
                st.metric("Engagement Rate", f"{engagement:.3f}")

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
    st.markdown("### Azure Connected - Simplified Version")
    
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
        This is a **simplified version** that connects to Azure SQL Database and shows your data.
        
        **Features:**
        - 🐦 **Tweet Analysis**: View and analyze individual tweets
        - 📊 **Popular Entities**: View trending entities and risk levels
        
        **Note:** This version shows data without the complex ECS model analysis.
        """)
        
        # Show system status
        st.subheader("🔄 System Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("✅ Database Connected")
        with col2:
            st.info("📊 Data Available")
        with col3:
            st.info("🔧 Simplified Mode")
    
    elif page == "🐦 Tweet Analysis":
        show_tweet_selection(engine)
    
    elif page == "📊 Popular Entities":
        show_popular_entities_analysis(engine)
    
    # Footer
    st.markdown("---")
    st.markdown("**ECS System v1.0** | Azure SQL Database | Simplified Mode")

if __name__ == "__main__":
    main()
