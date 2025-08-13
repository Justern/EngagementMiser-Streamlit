#!/usr/bin/env python3
"""
Engagement Concordance Score - Streamlit Web App
=============================================

A professional web application for testing and demonstrating the ECS system.
Integrates with the actual 10-model Engagement Concordance Score system.
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
    page_title="Engagement Concordance Score",
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

# Database connection configuration
@st.cache_resource
def get_database_engine():
    """Create and return a SQLAlchemy database engine for Azure SQL Database."""
    try:
        # Database connection parameters from secrets
        server = st.secrets.get("azure_db.server", "ecs-sql-server-engagementmiser.database.windows.net")
        database = st.secrets.get("azure_db.database", "ecs_tweets_db")
        username = st.secrets.get("azure_db.username", "ecsadmin")
        password = st.secrets.get("azure_db.password", "EngagementMiser!")
        
        # Create Azure SQL Database connection string for SQLAlchemy
        # Use the correct Azure SQL Database format
        conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no&Connection+Timeout=30"
        
        # Create SQLAlchemy engine
        from sqlalchemy import create_engine
        engine = create_engine(conn_str, pool_pre_ping=True, pool_recycle=300)
        return engine
        
    except Exception as e:
        st.error(f"❌ Azure SQL Database connection failed: {e}")
        st.info("Please check your Azure database credentials in the secrets configuration.")
        return None

def fetch_tweets_from_database(limit=50):
    """Fetch recent tweets from the database for selection."""
    try:
        engine = get_database_engine()
        if engine is None:
            return []
        
        # SQL query to fetch tweets - use string formatting for TOP clause
        query = f"""
        SELECT TOP ({limit}) 
            tweet_id,
            tweet_text,
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
        FROM tweets 
        ORDER BY created_at DESC
        """
        
        # Execute query directly without parameter binding for TOP clause
        from sqlalchemy import text
        with engine.connect() as connection:
            result = connection.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        return df.to_dict('records')
        
    except Exception as e:
        st.error(f"❌ Error fetching tweets: {e}")
        return []

def fetch_tweet_by_id(tweet_id):
    """Fetch a specific tweet by ID from the database."""
    try:
        engine = get_database_engine()
        if engine is None:
            return None
        
        # SQL query to fetch specific tweet - use f-string formatting for Azure SQL compatibility
        query = f"""
        SELECT 
            tweet_id,
            tweet_text,
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
        FROM tweets 
        WHERE tweet_id = {tweet_id}
        """
        
        # Execute query directly without parameter binding for Azure SQL compatibility
        from sqlalchemy import text
        with engine.connect() as connection:
            result = connection.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        if not df.empty:
            return df.iloc[0].to_dict()
        return None
        
    except Exception as e:
        st.error(f"❌ Error fetching tweet: {e}")
        return None

# Initialize the Engagement Concordance Score system
@st.cache_resource
def load_ecs_system():
    """Load the deployment-ready ECS system with Azure database integration."""
    try:
        with st.spinner("Loading Engagement Concordance Score system..."):
            st.info("🔄 Attempting to import deployment models...")
            
            # Import the deployment models
            from deployment_config import deployment_models
            
            st.info(f"📦 Imported deployment_models: {type(deployment_models)}")
            
            # Check if models are available
            if deployment_models:
                st.info("🔍 Checking model methods...")
                
                # Check if all 10 model methods exist
                required_methods = [
                    'hyperbole_falsehood_score',
                    'clickbait_score', 
                    'engagement_mismatch_score',
                    'content_recycling_score',
                    'coordinated_network_score',
                    'emotive_manipulation_score',
                    'rapid_engagement_spike_score',
                    'generic_comment_score',
                    'authority_signal_score',
                    'reply_bait_score'
                ]
                
                missing_methods = []
                for method in required_methods:
                    if not hasattr(deployment_models, method):
                        missing_methods.append(method)
                
                if missing_methods:
                    st.warning(f"⚠️ Missing model methods: {missing_methods}")
                    return None
                
                st.success("✅ ECS system loaded successfully with Azure database integration!")
                st.info("All 10 specialized detection models are ready for analysis.")
                return deployment_models
            else:
                st.warning("⚠️ Deployment models not loaded successfully.")
                return None
                
    except Exception as e:
        st.error(f"❌ Error loading ECS system: {e}")
        st.info("Please ensure deployment_config.py is available.")
        import traceback
        st.error(f"Full error: {traceback.format_exc()}")
        return None

def analyze_text_with_ecs(tweet_text):
    """Analyze text using the Engagement Concordance Score system."""
    try:
        # For now, we'll use a simplified approach since the ECS system expects tweet IDs
        # In a full deployment, you'd need to adapt the models to work with text input
        
        st.info("🔍 Note: Full ECS analysis requires tweet ID analysis. This is a simplified text analysis.")
        
        # For demonstration, return a basic analysis structure
        return {
            'composite_score': 0.5,  # Placeholder
            'risk_assessment': {
                'risk_level': 'MEDIUM',
                'risk_description': 'Analysis requires full ECS system integration'
            },
            'model_results': {},
            'summary': 'Text analysis mode - full ECS integration pending'
        }
        
    except Exception as e:
        st.error(f"❌ Error analyzing text: {e}")
        return None

def analyze_tweet_with_ecs(tweet_id):
    """Analyze a tweet using the ECS system with all 10 models."""
    try:
        st.info(f"🔍 Starting analysis for tweet ID: {tweet_id}")
        
        ecs_system = st.session_state.get('ecs_system')
        if not ecs_system:
            st.error("❌ ECS system not found in session state")
            return None
        
        st.info(f"📦 ECS system type: {type(ecs_system)}")
        st.info(f"📦 ECS system methods: {[method for method in dir(ecs_system) if 'score' in method]}")
        
        # Run all 10 models
        model_scores = {}
        
        st.info("🚀 Running individual models...")
        
        try:
            model_scores['hyperbole_falsehood'] = ecs_system.hyperbole_falsehood_score(tweet_id)
            st.info(f"✅ Hyperbole model: {model_scores['hyperbole_falsehood']}")
        except Exception as e:
            st.error(f"❌ Hyperbole model error: {e}")
            model_scores['hyperbole_falsehood'] = 0.0
        
        try:
            model_scores['clickbait'] = ecs_system.clickbait_score(tweet_id)
            st.info(f"✅ Clickbait model: {model_scores['clickbait']}")
        except Exception as e:
            st.error(f"❌ Clickbait model error: {e}")
            model_scores['clickbait'] = 0.0
        
        try:
            model_scores['engagement_mismatch'] = ecs_system.engagement_mismatch_score(tweet_id)
            st.info(f"✅ Engagement mismatch model: {model_scores['engagement_mismatch']}")
        except Exception as e:
            st.error(f"❌ Engagement mismatch model error: {e}")
            model_scores['engagement_mismatch'] = 0.0
        
        try:
            model_scores['content_recycling'] = ecs_system.content_recycling_score(tweet_id)
            st.info(f"✅ Content recycling model: {model_scores['content_recycling']}")
        except Exception as e:
            st.error(f"❌ Content recycling model error: {e}")
            model_scores['content_recycling'] = 0.0
        
        try:
            model_scores['coordinated_network'] = ecs_system.coordinated_network_score(tweet_id)
            st.info(f"✅ Coordinated network model: {model_scores['coordinated_network']}")
        except Exception as e:
            st.error(f"❌ Coordinated network model error: {e}")
            model_scores['coordinated_network'] = 0.0
        
        try:
            model_scores['emotive_manipulation'] = ecs_system.emotive_manipulation_score(tweet_id)
            st.info(f"✅ Emotive manipulation model: {model_scores['emotive_manipulation']}")
        except Exception as e:
            st.error(f"❌ Emotive manipulation model error: {e}")
            model_scores['emotive_manipulation'] = 0.0
        
        try:
            model_scores['rapid_engagement_spike'] = ecs_system.rapid_engagement_spike_score(tweet_id)
            st.info(f"✅ Rapid engagement spike model: {model_scores['rapid_engagement_spike']}")
        except Exception as e:
            st.error(f"❌ Rapid engagement spike model error: {e}")
            model_scores['rapid_engagement_spike'] = 0.0
        
        try:
            model_scores['generic_comment'] = ecs_system.generic_comment_score(tweet_id)
            st.info(f"✅ Generic comment model: {model_scores['generic_comment']}")
        except Exception as e:
            st.error(f"❌ Generic comment model error: {e}")
            model_scores['generic_comment'] = 0.0
        
        try:
            model_scores['authority_signal'] = ecs_system.authority_signal_score(tweet_id)
            st.info(f"✅ Authority signal model: {model_scores['authority_signal']}")
        except Exception as e:
            st.error(f"❌ Authority signal model error: {e}")
            model_scores['authority_signal'] = 0.0
        
        try:
            model_scores['reply_bait'] = ecs_system.reply_bait_score(tweet_id)
            st.info(f"✅ Reply bait model: {model_scores['reply_bait']}")
        except Exception as e:
            st.error(f"❌ Reply bait model error: {e}")
            model_scores['reply_bait'] = 0.0
        
        st.info(f"📊 All models completed. Scores: {model_scores}")
        
        # Calculate weighted composite score
        weights = {
            'hyperbole_falsehood': 0.6,
            'clickbait': 0.8,
            'engagement_mismatch': 1.0,
            'content_recycling': 0.9,
            'coordinated_network': 1.0,
            'emotive_manipulation': 0.6,
            'rapid_engagement_spike': 0.5,
            'generic_comment': 0.6,
            'authority_signal': 0.7,
            'reply_bait': 0.8
        }
        
        total_weight = sum(weights.values())
        weighted_sum = sum(model_scores[model] * weights[model] for model in model_scores)
        composite_score = weighted_sum / total_weight
        
        st.info(f"🎯 Composite score calculated: {composite_score}")
        
        # Determine risk level
        if composite_score < 0.4:
            risk_level = 'LOW'
            risk_description = 'Content appears genuine with minimal manipulation indicators'
        elif composite_score < 0.7:
            risk_level = 'MEDIUM'
            risk_description = 'Some concerning patterns detected, moderate manipulation risk'
        else:
            risk_level = 'HIGH'
            risk_description = 'Multiple manipulation patterns detected, high risk content'
        
        return {
            'composite_score': composite_score,
            'risk_assessment': {
                'risk_level': risk_level,
                'risk_description': risk_description
            },
            'model_results': model_scores,
            'weights': weights,
            'summary': f'Analyzed with {len(model_scores)} specialized models'
        }
        
    except Exception as e:
        st.error(f"❌ Error analyzing tweet: {e}")
        import traceback
        st.error(f"Full error: {traceback.format_exc()}")
        return None

def show_tweet_selection():
    """Show a dropdown to select from available tweets in the database."""
    st.subheader("📊 Select Tweet from Database")
    
    # Fetch ALL tweets from database (not just 50)
    tweets = fetch_tweets_from_database(limit=1020)
    
    if not tweets:
        st.warning("No tweets found in database. Please ensure your database connection is working.")
        return None
    
    # Create a selection interface
    st.write(f"Found {len(tweets)} tweets in database. Select one to analyze:")
    
    # Create a formatted display for each tweet
    tweet_options = []
    for tweet in tweets:
        # Truncate tweet text for display
        display_text = tweet['tweet_text'][:80] + "..." if len(tweet['tweet_text']) > 80 else tweet['tweet_text']
        
        # Format the option text with more info
        option_text = f"ID: {tweet['tweet_id']} | {display_text}"
        tweet_options.append((option_text, tweet['tweet_id']))
    
    # Create dropdown with search functionality
    selected_option = st.selectbox(
        "Choose a tweet to analyze:",
        options=[opt[0] for opt in tweet_options],
        index=0,
        help="Select a tweet from your database to automatically populate the tweet ID field below"
    )
    
    # Get the selected tweet ID
    selected_tweet_id = None
    for option_text, tweet_id in tweet_options:
        if option_text == selected_option:
            selected_tweet_id = tweet_id
            break
    
    if selected_tweet_id:
        # Show tweet details
        tweet_data = fetch_tweet_by_id(selected_tweet_id)
        if tweet_data:
            st.success(f"✅ Selected Tweet ID: {selected_tweet_id}")
            
            # Display tweet information in a cleaner format
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Tweet Text:**")
                st.write(tweet_data['tweet_text'])
                
                st.write("**Author ID:**")
                st.write(tweet_data['author_id'])
                
                st.write("**Created At:**")
                st.write(tweet_data['created_at'])
            
            with col2:
                st.write("**Engagement Metrics:**")
                st.write(f"Retweets: {tweet_data['retweet_count']}")
                st.write(f"Likes: {tweet_data['like_count']}")
                st.write(f"Followers: {tweet_data['followers_count']}")
                st.write(f"Total Engagements: {tweet_data['total_engagements']}")
                
                # Handle None values for engagement_rate
                engagement_rate = tweet_data['engagement_rate']
                if engagement_rate is not None:
                    st.write(f"Engagement Rate: {engagement_rate:.4f}")
                else:
                    st.write("Engagement Rate: N/A")
            
            # Show hashtags if available
            hashtags = [tag for tag in [tweet_data['hashtag1'], tweet_data['hashtag2'], tweet_data['hashtag3']] if tag]
            if hashtags:
                st.write("**Hashtags:**")
                st.write(", ".join(hashtags))
            
            return selected_tweet_id
    
    return None

def show_live_analysis():
    """Show live tweet analysis section with database integration."""
    st.subheader("🔍 Live Tweet Analysis")
    
    # Check database connection
    engine = get_database_engine()
    if engine is None:
        st.error("❌ Cannot connect to database. Please check your Azure SQL Database connection.")
        return
    
    # Show tweet selection from database
    selected_tweet_id = show_tweet_selection()
    
    if selected_tweet_id:
        st.write("---")
        st.write("**Now you can use this Tweet ID in the scoring section below!**")
        
        # Auto-populate the tweet ID field in the main scoring section
        st.session_state['selected_tweet_id'] = selected_tweet_id
        st.success(f"🎯 Tweet ID {selected_tweet_id} is ready for scoring!")
        
        # Add a button to copy the tweet ID
        if st.button("📋 Copy Tweet ID to Clipboard"):
            st.write("Tweet ID copied! You can now paste it in the scoring section.")
    
    st.write("---")
    st.write("**Use the Tweet ID above or manually enter one below:**")
    
    # Manual tweet ID input
    tweet_id_input = st.text_input(
        "Enter Tweet ID:",
        value=st.session_state.get('selected_tweet_id', ''),
        help="Enter a specific tweet ID to analyze, or use the selection above"
    )
    
    if tweet_id_input:
        try:
            tweet_id = int(tweet_id_input)
            
            # Fetch tweet from database
            tweet_data = fetch_tweet_by_id(tweet_id)
            
            if tweet_data:
                st.success(f"✅ Found tweet {tweet_id} in database!")
                
                # Display tweet information
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Tweet Text:**")
                    st.write(tweet_data['tweet_text'])
                    
                    st.write("**Author ID:**")
                    st.write(tweet_data['author_id'])
                    
                    st.write("**Created At:**")
                    st.write(tweet_data['created_at'])
                
                with col2:
                    st.write("**Engagement Metrics:**")
                    st.write(f"Retweets: {tweet_data['retweet_count']}")
                    st.write(f"Likes: {tweet_data['like_count']}")
                    st.write(f"Followers: {tweet_data['followers_count']}")
                    st.write(f"Total Engagements: {tweet_data['total_engagements']}")
                    
                    # Handle None values for engagement_rate
                    engagement_rate = tweet_data['engagement_rate']
                    if engagement_rate is not None:
                        st.write(f"Engagement Rate: {engagement_rate:.4f}")
                    else:
                        st.write("Engagement Rate: N/A")
                
                # Show hashtags if available
                hashtags = [tag for tag in [tweet_data['hashtag1'], tweet_data['hashtag2'], tweet_data['hashtag3']] if tag]
                if hashtags:
                    st.write("**Hashtags:**")
                    st.write(", ".join(hashtags))
                
                # Now analyze the tweet
                st.write("---")
                st.write("**AI Analysis Results:**")
                
                # Get the tweet text for analysis
                tweet_text = tweet_data['tweet_text']
                
                # Perform analysis using ECS system
                if st.session_state.get('ecs_system'):
                    # Analyze with ECS system using tweet ID
                    analysis_results = analyze_tweet_with_ecs(int(tweet_id))
                    
                    if analysis_results:
                        # Extract results
                        composite_score = analysis_results.get('composite_score', 0.0)
                        risk_assessment = analysis_results.get('risk_assessment', {})
                        risk_level = risk_assessment.get('risk_level', 'UNKNOWN')
                        risk_description = risk_assessment.get('risk_description', 'Analysis completed')
                        model_scores = analysis_results.get('model_results', {})
                        
                        # Display results
                        st.write("**🤖 ECS Analysis Results:**")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Risk Level", risk_level)
                        
                        with col2:
                            st.metric("Composite Score", f"{composite_score:.3f}")
                        
                        with col3:
                            # Color code based on risk level
                            if risk_level == 'LOW':
                                st.success("✅ Low Risk")
                            elif risk_level == 'MEDIUM':
                                st.warning("⚠️ Medium Risk")
                            else:
                                st.error("🚨 High Risk")
                        
                        # Show risk description
                        st.write("**Risk Assessment:**")
                        st.info(risk_description)
                        
                        # Show individual model scores
                        st.write("**Individual Model Scores:**")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            for i, (model, score) in enumerate(list(model_scores.items())[:5]):
                                st.metric(f"{model.replace('_', ' ').title()}", f"{score:.3f}")
                        
                        with col2:
                            for i, (model, score) in enumerate(list(model_scores.items())[5:]):
                                st.metric(f"{model.replace('_', ' ').title()}", f"{score:.3f}")
                        
                        # Show the analyzed text
                        st.write("**Analyzed Text:**")
                        st.write(tweet_text)
                        
                    else:
                        st.error("❌ Analysis failed. Please try again.")
                else:
                    st.error("❌ ECS system not loaded. Please check your model configuration.")
            
            else:
                st.warning(f"⚠️ Tweet ID {tweet_id} not found in database.")
                st.info("This tweet ID doesn't exist in your current database. You can still analyze it manually by entering the tweet text below.")
                
                # Fallback to manual text input
                manual_text = st.text_area(
                    "Enter Tweet Text for Analysis:",
                    height=100,
                    help="If the tweet isn't in your database, you can manually enter the text here"
                )
                
                if manual_text and st.button("Analyze Text"):
                    if st.session_state.get('ecs_system'):
                        # Perform analysis on manual text
                        analysis_results = analyze_text_with_ecs(manual_text)
                        
                        if analysis_results:
                            # Extract results
                            composite_score = analysis_results.get('composite_score', 0.0)
                            risk_assessment = analysis_results.get('risk_assessment', {})
                            risk_level = risk_assessment.get('risk_level', 'UNKNOWN')
                            risk_description = risk_assessment.get('risk_description', 'Analysis completed')
                            
                            st.success(f"Analysis Complete! Risk Level: {risk_level} (Score: {composite_score:.3f})")
                            
                            # Display results
                            st.write("**🤖 ECS Analysis Results:**")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Risk Level", risk_level)
                                st.metric("Composite Score", f"{composite_score:.3f}")
                            
                            with col2:
                                # Color code based on risk level
                                if risk_level == 'LOW':
                                    st.success("✅ Low Risk")
                                elif risk_level == 'MEDIUM':
                                    st.warning("⚠️ Medium Risk")
                                else:
                                    st.error("🚨 High Risk")
                            
                            # Show risk description
                            st.write("**Risk Assessment:**")
                            st.info(risk_description)
                        else:
                            st.error("❌ Analysis failed.")
                    else:
                        st.error("❌ ECS system not loaded. Please check your model configuration.")
        
        except ValueError:
            st.error("❌ Please enter a valid numeric Tweet ID.")

def show_main_scoring():
    """Show the main scoring interface with tweet ID input."""
    st.subheader("🎯 Tweet Scoring Interface")
    
    # Check if we have a selected tweet ID from the database
    selected_tweet_id = st.session_state.get('selected_tweet_id', '')
    
    if selected_tweet_id:
        st.success(f"🎯 Using Tweet ID from database: {selected_tweet_id}")
        st.info("This tweet ID was selected from your database above. You can modify it below if needed.")
    
    # Tweet ID input
    tweet_id = st.text_input(
        "Tweet ID:",
        value=str(selected_tweet_id) if selected_tweet_id else "",
        placeholder="Enter a valid tweet ID (e.g., 123456789012345)",
        help="Enter the numeric ID of the tweet you want to analyze"
    )
    
    # Analysis button
    analyze_button = st.button("🚀 Analyze Tweet", type="primary")
    
    if analyze_button and tweet_id:
        if tweet_id.isdigit() and len(tweet_id) >= 10:
            # Check if ECS system is available
            if not st.session_state.get('ecs_system'):
                st.error("❌ ECS system not loaded. Please check your model configuration.")
                return
            
            # Fetch tweet from database
            tweet_data = fetch_tweet_by_id(int(tweet_id))
            
            if tweet_data is None:
                st.error(f"❌ Tweet with ID {tweet_id} not found in database.")
                st.info("💡 Try using the tweet selection feature above to find available tweets.")
                return
            
            # Use tweet text from database
            tweet_text = tweet_data.get('tweet_text', '')
            
            if not tweet_text.strip():
                st.warning("⚠️ Tweet text is empty in database.")
                return
            
            # Analyze with ECS system
            with st.spinner("🔍 Analyzing tweet with ECS system..."):
                # Analyze with the ECS system
                analysis_results = analyze_tweet_with_ecs(int(tweet_id))
                
                if analysis_results:
                    # Extract results
                    composite_score = analysis_results.get('composite_score', 0.0)
                    risk_assessment = analysis_results.get('risk_assessment', {})
                    risk_level = risk_assessment.get('risk_level', 'UNKNOWN')
                    risk_description = risk_assessment.get('risk_description', 'Analysis completed')
                    model_scores = analysis_results.get('model_results', {})
                    
                    # Display results
                    st.success(f"✅ Analysis complete! Tweet ID: {tweet_id}")
                    
                    # Use actual tweet data from database
                    tweet_data_display = {
                        "text": tweet_text,
                        "author_id": tweet_data.get('author_id', 'Unknown'),
                        "created_at": tweet_data.get('created_at', 'Unknown'),
                        "retweet_count": tweet_data.get('retweet_count', 0),
                        "like_count": tweet_data.get('like_count', 0),
                        "followers_count": tweet_data.get('followers_count', 0),
                        "total_engagements": tweet_data.get('total_engagements', 0),
                        "engagement_rate": tweet_data.get('engagement_rate', 0.0)
                    }
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**Tweet Content:**")
                        st.info(tweet_data_display["text"])
                        
                        # Tweet metadata
                        st.markdown("**Tweet Details:**")
                        col1a, col1b = st.columns(2)
                        with col1a:
                            st.markdown(f"**Tweet ID:** `{tweet_id}`")
                            st.markdown(f"**Author ID:** `{tweet_data_display['author_id']}`")
                            st.markdown(f"**Created:** {tweet_data_display['created_at']}")
                        with col1b:
                            st.markdown(f"**Retweets:** {tweet_data_display['retweet_count']:,}")
                            st.markdown(f"**Likes:** {tweet_data_display['like_count']:,}")
                            st.markdown(f"**Followers:** {tweet_data_display['followers_count']:,}")
                            st.markdown(f"**Total Engagements:** {tweet_data_display['total_engagements']:,}")
                            
                            # Handle None values for engagement_rate
                            engagement_rate = tweet_data_display['engagement_rate']
                            if engagement_rate is not None:
                                st.markdown(f"**Engagement Rate:** {engagement_rate:.4f}")
                            else:
                                st.markdown("**Engagement Rate:** N/A")
                        
                        st.markdown("**Analysis Results:**")
                        
                        # Show ECS results
                        st.write("**🤖 ECS Analysis:**")
                        col1a, col1b = st.columns(2)
                        with col1a:
                            st.metric("Risk Level", risk_level)
                            st.metric("Composite Score", f"{composite_score:.3f}")
                        with col1b:
                            if risk_level == 'LOW':
                                st.success("✅ Low Risk")
                            elif risk_level == 'MEDIUM':
                                st.warning("⚠️ Medium Risk")
                            else:
                                st.error("🚨 High Risk")
                    
                    with col2:
                        st.markdown("**Analysis Summary:**")
                        st.info(f"Risk Level: {risk_level}")
                        st.info(f"Composite Score: {composite_score:.3f}")
                        st.info(f"Analyzed with ECS system")
                    
                    # Show individual model scores
                    st.subheader("📊 Individual Model Scores")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        for i, (model, score) in enumerate(list(model_scores.items())[:5]):
                            st.metric(f"{model.replace('_', ' ').title()}", f"{score:.3f}")
                    
                    with col2:
                        for i, (model, score) in enumerate(list(model_scores.items())[5:]):
                            st.metric(f"{model.replace('_', ' ').title()}", f"{score:.3f}")
                    
                    # Show risk description
                    st.subheader("📊 Risk Assessment")
                    st.info(risk_description)
                    
                else:
                    st.error("❌ Analysis failed.")
        
        else:
            st.error("❌ Please enter a valid tweet ID (numeric, at least 10 digits)")
            st.info("💡 Use the tweet selection feature above to find valid tweet IDs from your database.")

def show_system_overview():
    """Show system overview and methodology."""
    st.header("📈 System Overview")
    
    st.markdown("""
    ## 🎯 What is the Engagement Concordance Score?
    
    The Engagement Concordance Score (ECS) is an AI-powered system that analyzes social media content 
    for various types of engagement manipulation and low-quality content patterns using **10 specialized detection models**.
    
    **🤖 AI Models:** 10 specialized models, each focusing on specific manipulation patterns
    
    ## 🔍 Detection Models
    
    Our system employs **10 specialized detection models**, each focusing on specific manipulation patterns:
    
    1. **Hyperbole & Falsehood Detector** (Weight: 0.6) - Detects exaggerated claims and false statements
    2. **Clickbait Headline Classifier** (Weight: 0.8) - Identifies sensationalist headlines
    3. **Engagement Mismatch Detector** (Weight: 1.0) - Detects engagement patterns inconsistent with content quality
    4. **Content Recycling Detector** (Weight: 0.9) - Identifies duplicate or recycled content
    5. **Coordinated Account Network Model** (Weight: 1.0) - Detects coordinated behavior and bot networks
    6. **Emotive Manipulation Detector** (Weight: 0.6) - Identifies emotional manipulation techniques
    7. **Rapid Engagement Spike Detector** (Weight: 0.5) - Detects unusual engagement velocity patterns
    8. **Generic Comment Detector** (Weight: 0.6) - Identifies low-quality, generic engagement
    9. **Authority-Signal Manipulation** (Weight: 0.7) - Detects false authority claims
    10. **Reply-Bait Detector** (Weight: 0.8) - Identifies manipulative reply patterns
    
    ## 🧮 Scoring Methodology
    
    **Composite Score Calculation:**
    - Each model produces a score from 0.0 to 1.0
    - Scores are weighted by model importance
    - Final composite score represents overall risk level
    
    **Risk Assessment:**
    - **LOW RISK** (0.0 - 0.4): Content appears genuine and high-quality
    - **MEDIUM RISK** (0.4 - 0.7): Some concerning patterns detected
    - **HIGH RISK** (0.7 - 1.0): Multiple manipulation patterns detected
    
    ## 🔬 Research Applications
    
    This system is designed for:
    - Social media research and analysis
    - Content moderation studies
    - Engagement manipulation detection
    - Digital literacy research
    - Platform integrity analysis
    """)
    
    # Technical details
    st.subheader("🛠️ Technical Implementation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Architecture:**
        - 10 specialized detection models
        - Weighted scoring system
        - Azure SQL Database integration
        - Real-time analysis capabilities
        
        **Model Integration:**
        - Each model has its own analysis script
        - Models can be run independently or together
        - Configurable weights for different use cases
        """)
    
    with col2:
        st.markdown("""
        **Features:**
        - Real-time tweet analysis
        - Comprehensive risk assessment
        - Individual model scores
        - Weighted composite scoring
        
        **Deployment:**
        - Streamlit web interface
        - Cloud-ready architecture
        - Scalable model system
        - Real-time processing
        """)

def show_model_performance():
    """Show model performance metrics and analysis."""
    st.header("📊 Model Performance")
    
    st.info("Model performance metrics and analysis will be displayed here.")
    st.write("This section will show detailed performance analytics for all 10 specialized detection models.")

def show_settings():
    """Show application settings and configuration."""
    st.header("⚙️ Settings")
    
    st.info("Application settings and configuration options will be displayed here.")
    st.write("This section will allow users to configure various application parameters.")

# Main app structure
def main():
    # Initialize session state
    if 'selected_tweet_id' not in st.session_state:
        st.session_state['selected_tweet_id'] = ''
    
    # Load ECS system
    if 'ecs_system' not in st.session_state:
        st.session_state['ecs_system'] = load_ecs_system()
    
    # Header
    st.title("🎯 Engagement Concordance Score")
    st.markdown("AI-powered social media content risk assessment using 10 specialized detection models")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 System Overview", "🔍 Live Tweet Analysis", "📊 Model Performance", "⚙️ Settings"]
    )
    
    # Page routing
    if page == "🏠 System Overview":
        show_system_overview()
    elif page == "🔍 Live Tweet Analysis":
        show_live_analysis()
        st.write("---")
        show_main_scoring()
    elif page == "📊 Model Performance":
        show_model_performance()
    elif page == "⚙️ Settings":
        show_settings()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Database Status:**")
    engine = get_database_engine()
    if engine:
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) as count FROM tweets"))
                count = result.fetchone()[0]
                st.sidebar.success(f"✅ Connected ({count:,} tweets)")
        except:
            st.sidebar.error("❌ Connection failed")
    else:
        st.sidebar.error("❌ Not configured")
    
    # ECS System Status
    st.sidebar.markdown("**ECS System Status:**")
    if st.session_state.get('ecs_system'):
        st.sidebar.success("✅ ECS System (10 models)")
        st.sidebar.info("All specialized detection models loaded")
        st.sidebar.info("Azure database integration active")
    else:
        st.sidebar.error("❌ ECS System not loaded")

if __name__ == "__main__":
    main() 
