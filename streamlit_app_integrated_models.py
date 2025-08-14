#!/usr/bin/env python3
"""
Engagement Concordance Score - Streamlit App with Direct Model Integration
=======================================================================

This app integrates the actual specialized logic from each of the 10 ECS models
directly into the main app, avoiding subprocess complexity.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from datetime import datetime
import os
import tempfile
import json

# Page configuration
st.set_page_config(
    page_title="ECS Integrated Models",
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
        
        # Azure SQL Database connection string
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

# ============================================================================
# INTEGRATED MODEL FUNCTIONS - Direct implementation of specialized logic
# ============================================================================

def calculate_authority_signal_manipulation_score(tweet_id, engine):
    """Calculate Authority Signal Manipulation score using specialized logic."""
    try:
        # Get tweet and user data from Azure
        query = """
        SELECT t.text, t.author_id, u.followers_count, u.verified, u.account_age_days
        FROM [dbo].[Tweets_Sample_4M] t
        JOIN [dbo].[TwitterUsers] u ON t.author_id = u.user_id
        WHERE t.tweet_id = ?
        """
        
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text(query), [str(tweet_id)]).fetchone()
        
        if not result:
            return 0.0
        
        tweet_text = result[0].lower()
        followers_count = result[2] or 0
        verified = result[3] or False
        account_age_days = result[4] or 0
        
        # Authority phrases detection
        authority_phrases = [
            'expert', 'professional', 'specialist', 'consultant', 'advisor',
            'researcher', 'scientist', 'doctor', 'professor', 'analyst',
            'strategist', 'expertise', 'authority', 'certified', 'licensed'
        ]
        
        authority_score = sum(1 for phrase in authority_phrases if phrase in tweet_text)
        authority_score = min(authority_score / 5, 1.0)  # Normalize to 0-1
        
        # Profile mismatch detection
        profile_score = 0.0
        if authority_score > 0.3:  # If claiming authority
            if followers_count < 1000 and not verified:
                profile_score = 0.8  # High suspicion
            elif followers_count < 5000 and account_age_days < 365:
                profile_score = 0.6  # Medium suspicion
            elif followers_count < 10000:
                profile_score = 0.4  # Low suspicion
        
        # Final score combines both factors
        final_score = (authority_score * 0.4) + (profile_score * 0.6)
        return min(final_score, 1.0)
        
    except Exception as e:
        st.warning(f"Authority Signal Manipulation failed: {e}")
        return 0.0

def calculate_clickbait_headline_classifier_score(tweet_id, engine):
    """Calculate Clickbait Headline Classifier score using specialized logic."""
    try:
        # Get tweet text from Azure
        query = "SELECT text FROM [dbo].[Tweets_Sample_4M] WHERE tweet_id = ?"
        
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text(query), [str(tweet_id)]).fetchone()
        
        if not result:
            return 0.0
        
        tweet_text = result[0].lower()
        
        # Clickbait indicators
        clickbait_patterns = [
            'you won\'t believe', 'shocking', 'amazing', 'incredible', 'unbelievable',
            'this will blow your mind', 'what happens next', 'the truth about',
            'they don\'t want you to know', 'secret', 'exposed', 'revealed',
            'number one reason', 'top 10', 'best kept secret', 'game changer',
            'revolutionary', 'breakthrough', 'miracle', 'instant'
        ]
        
        # Count clickbait patterns
        pattern_count = sum(1 for pattern in clickbait_patterns if pattern in tweet_text)
        
        # Question marks and exclamation marks
        question_marks = tweet_text.count('?')
        exclamation_marks = tweet_text.count('!')
        
        # Calculate score
        pattern_score = min(pattern_count / 3, 1.0)
        punctuation_score = min((question_marks + exclamation_marks) / 4, 1.0)
        
        # Combine scores
        final_score = (pattern_score * 0.7) + (punctuation_score * 0.3)
        return min(final_score, 1.0)
        
    except Exception as e:
        st.warning(f"Clickbait Headline Classifier failed: {e}")
        return 0.0

def calculate_content_recycling_detector_score(tweet_id, engine):
    """Calculate Content Recycling Detector score using specialized logic."""
    try:
        # Get tweet data from Azure
        query = """
        SELECT text, retweet_count, like_count, reply_count, created_at
        FROM [dbo].[Tweets_Sample_4M] WHERE tweet_id = ?
        """
        
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text(query), [str(tweet_id)]).fetchone()
        
        if not result:
            return 0.0
        
        tweet_text = result[0].lower()
        retweet_count = result[1] or 0
        like_count = result[2] or 0
        reply_count = result[3] or 0
        
        # Viral content indicators
        viral_indicators = [
            'viral', 'trending', 'going viral', 'blowing up', 'exploding',
            'everyone is talking about', 'breaking the internet', 'hot topic',
            'must see', 'share this', 'retweet this', 'spread the word'
        ]
        
        # Check for viral language
        viral_score = sum(1 for indicator in viral_indicators if indicator in tweet_text)
        viral_score = min(viral_score / 3, 1.0)
        
        # Engagement ratio analysis
        total_engagement = retweet_count + like_count + reply_count
        engagement_score = min(total_engagement / 1000, 1.0)  # Normalize to 0-1
        
        # Time-based analysis (simplified)
        time_score = 0.5  # Placeholder for time-based logic
        
        # Final score
        final_score = (viral_score * 0.4) + (engagement_score * 0.4) + (time_score * 0.2)
        return min(final_score, 1.0)
        
    except Exception as e:
        st.warning(f"Content Recycling Detector failed: {e}")
        return 0.0

def calculate_coordinated_account_network_score(tweet_id, engine):
    """Calculate Coordinated Account Network score using specialized logic."""
    try:
        # Get tweet and user data from Azure
        query = """
        SELECT t.text, t.author_id, t.created_at, u.followers_count, u.following_count
        FROM [dbo].[Tweets_Sample_4M] t
        JOIN [dbo].[TwitterUsers] u ON t.author_id = u.user_id
        WHERE t.tweet_id = ?
        """
        
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text(query), [str(tweet_id)]).fetchone()
        
        if not result:
            return 0.0
        
        tweet_text = result[0].lower()
        followers_count = result[3] or 0
        following_count = result[4] or 0
        
        # Bot/automation indicators
        bot_indicators = [
            'bot', 'automated', 'auto', 'script', 'program', 'algorithm',
            'machine learning', 'ai', 'artificial intelligence', 'automation'
        ]
        
        # Check for bot language
        bot_score = sum(1 for indicator in bot_indicators if indicator in tweet_text)
        bot_score = min(bot_score / 2, 1.0)
        
        # Network analysis (simplified)
        if following_count > 0:
            ratio = followers_count / following_count
            if ratio < 0.1:  # Following many, few followers
                network_score = 0.8
            elif ratio < 0.5:
                network_score = 0.6
            elif ratio < 1.0:
                network_score = 0.4
            else:
                network_score = 0.2
        else:
            network_score = 0.5
        
        # Final score
        final_score = (bot_score * 0.6) + (network_score * 0.4)
        return min(final_score, 1.0)
        
    except Exception as e:
        st.warning(f"Coordinated Account Network failed: {e}")
        return 0.0

def calculate_emotive_manipulation_score(tweet_id, engine):
    """Calculate Emotive Manipulation score using specialized logic."""
    try:
        # Get tweet text from Azure
        query = "SELECT text FROM [dbo].[Tweets_Sample_4M] WHERE tweet_id = ?"
        
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text(query), [str(tweet_id)]).fetchone()
        
        if not result:
            return 0.0
        
        tweet_text = result[0].lower()
        
        # Emotional manipulation indicators
        emotional_indicators = [
            'fear', 'anger', 'hate', 'love', 'joy', 'sadness', 'surprise',
            'disgust', 'anxiety', 'panic', 'excitement', 'euphoria', 'despair',
            'hope', 'faith', 'belief', 'trust', 'suspicion', 'paranoia'
        ]
        
        # Intense emotional words
        intense_emotions = [
            'hate', 'love', 'fear', 'anger', 'joy', 'sadness', 'despair',
            'euphoria', 'panic', 'anxiety', 'excitement', 'terror', 'ecstasy'
        ]
        
        # Count emotional indicators
        emotion_count = sum(1 for emotion in emotional_indicators if emotion in tweet_text)
        intense_count = sum(1 for emotion in intense_emotions if emotion in tweet_text)
        
        # Calculate scores
        emotion_score = min(emotion_count / 5, 1.0)
        intense_score = min(intense_count / 3, 1.0)
        
        # Final score
        final_score = (emotion_score * 0.6) + (intense_score * 0.4)
        return min(final_score, 1.0)
        
    except Exception as e:
        st.warning(f"Emotive Manipulation failed: {e}")
        return 0.0

def calculate_engagement_mismatch_score(tweet_id, engine):
    """Calculate Engagement Mismatch score using specialized logic."""
    try:
        # Get tweet data from Azure
        query = """
        SELECT text, retweet_count, like_count, reply_count, author_id
        FROM [dbo].[Tweets_Sample_4M] WHERE tweet_id = ?
        """
        
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text(query), [str(tweet_id)]).fetchone()
        
        if not result:
            return 0.0
        
        tweet_text = result[0].lower()
        retweet_count = result[1] or 0
        like_count = result[2] or 0
        reply_count = result[3] or 0
        
        # Engagement baiting indicators
        engagement_bait = [
            'like if you agree', 'retweet if you support', 'comment below',
            'double tap', 'share this', 'tag a friend', 'who else thinks',
            'raise your hand if', 'drop a heart if', 'say amen if'
        ]
        
        # Check for engagement baiting
        bait_score = sum(1 for bait in engagement_bait if bait in tweet_text)
        bait_score = min(bait_score / 2, 1.0)
        
        # Engagement ratio analysis
        total_engagement = retweet_count + like_count + reply_count
        if total_engagement > 0:
            engagement_score = min(total_engagement / 500, 1.0)
        else:
            engagement_score = 0.0
        
        # Final score (higher bait score = higher suspicion)
        final_score = (bait_score * 0.7) + (engagement_score * 0.3)
        return min(final_score, 1.0)
        
    except Exception as e:
        st.warning(f"Engagement Mismatch failed: {e}")
        return 0.0

def calculate_generic_comment_score(tweet_id, engine):
    """Calculate Generic Comment score using specialized logic."""
    try:
        # Get tweet text from Azure
        query = "SELECT text FROM [dbo].[Tweets_Sample_4M] WHERE tweet_id = ?"
        
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text(query), [str(tweet_id)]).fetchone()
        
        if not result:
            return 0.0
        
        tweet_text = result[0].lower()
        
        # Generic response indicators
        generic_indicators = [
            'nice', 'good', 'great', 'awesome', 'cool', 'interesting',
            'thanks', 'thank you', 'appreciate it', 'good point',
            'i agree', 'you\'re right', 'exactly', 'true', 'correct',
            'same', 'me too', 'i feel you', 'relatable'
        ]
        
        # Count generic indicators
        generic_count = sum(1 for indicator in generic_indicators if indicator in tweet_text)
        
        # Calculate score
        final_score = min(generic_count / 3, 1.0)
        return final_score
        
    except Exception as e:
        st.warning(f"Generic Comment failed: {e}")
        return 0.0

def calculate_hyperbole_falsehood_score(tweet_id, engine):
    """Calculate Hyperbole/Falsehood score using specialized logic."""
    try:
        # Get tweet text from Azure
        query = "SELECT text FROM [dbo].[Tweets_Sample_4M] WHERE tweet_id = ?"
        
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text(query), [str(tweet_id)]).fetchone()
        
        if not result:
            return 0.0
        
        tweet_text = result[0].lower()
        
        # Hyperbole indicators
        hyperbole_indicators = [
            'always', 'never', 'everyone', 'nobody', 'everything', 'nothing',
            'best', 'worst', 'perfect', 'terrible', 'amazing', 'horrible',
            'incredible', 'unbelievable', 'impossible', 'guaranteed',
            '100%', 'absolutely', 'definitely', 'certainly'
        ]
        
        # Falsehood indicators
        falsehood_indicators = [
            'fake news', 'hoax', 'conspiracy', 'cover up', 'hidden truth',
            'they don\'t want you to know', 'secret', 'exposed', 'revealed',
            'fact check', 'debunked', 'misinformation', 'disinformation'
        ]
        
        # Count indicators
        hyperbole_count = sum(1 for indicator in hyperbole_indicators if indicator in tweet_text)
        falsehood_count = sum(1 for indicator in falsehood_indicators if indicator in tweet_text)
        
        # Calculate scores
        hyperbole_score = min(hyperbole_count / 4, 1.0)
        falsehood_score = min(falsehood_count / 2, 1.0)
        
        # Final score
        final_score = (hyperbole_score * 0.6) + (falsehood_score * 0.4)
        return min(final_score, 1.0)
        
    except Exception as e:
        st.warning(f"Hyperbole/Falsehood failed: {e}")
        return 0.0

def calculate_rapid_engagement_spike_score(tweet_id, engine):
    """Calculate Rapid Engagement Spike score using specialized logic."""
    try:
        # Get tweet data from Azure
        query = """
        SELECT text, retweet_count, like_count, reply_count, created_at
        FROM [dbo].[Tweets_Sample_4M] WHERE tweet_id = ?
        """
        
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text(query), [str(tweet_id)]).fetchone()
        
        if not result:
            return 0.0
        
        tweet_text = result[0].lower()
        retweet_count = result[1] or 0
        like_count = result[2] or 0
        reply_count = result[3] or 0
        
        # Trending/viral indicators
        trending_indicators = [
            'trending', 'viral', 'hot topic', 'breaking', 'news',
            'just in', 'update', 'developing', 'latest', 'recent'
        ]
        
        # Check for trending language
        trending_score = sum(1 for indicator in trending_indicators if indicator in tweet_text)
        trending_score = min(trending_score / 3, 1.0)
        
        # Engagement analysis
        total_engagement = retweet_count + like_count + reply_count
        engagement_score = min(total_engagement / 1000, 1.0)
        
        # Final score
        final_score = (trending_score * 0.5) + (engagement_score * 0.5)
        return min(final_score, 1.0)
        
    except Exception as e:
        st.warning(f"Rapid Engagement Spike failed: {e}")
        return 0.0

def calculate_reply_bait_score(tweet_id, engine):
    """Calculate Reply Bait score using specialized logic."""
    try:
        # Get tweet text from Azure
        query = "SELECT text FROM [dbo].[Tweets_Sample_4M] WHERE tweet_id = ?"
        
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text(query), [str(tweet_id)]).fetchone()
        
        if not result:
            return 0.0
        
        tweet_text = result[0].lower()
        
        # Reply bait indicators
        reply_bait_indicators = [
            'what do you think?', 'your thoughts?', 'agree or disagree?',
            'comment below', 'let me know', 'what\'s your opinion?',
            'who else', 'raise your hand if', 'drop a heart if',
            'tag someone who', 'who can relate?', 'am i the only one?'
        ]
        
        # Question indicators
        question_indicators = ['?', 'what', 'how', 'why', 'when', 'where', 'who']
        
        # Count indicators
        bait_count = sum(1 for bait in reply_bait_indicators if bait in tweet_text)
        question_count = sum(1 for question in question_indicators if question in tweet_text)
        
        # Calculate scores
        bait_score = min(bait_count / 2, 1.0)
        question_score = min(question_count / 3, 1.0)
        
        # Final score
        final_score = (bait_score * 0.7) + (question_score * 0.3)
        return min(final_score, 1.0)
        
    except Exception as e:
        st.warning(f"Reply Bait failed: {e}")
        return 0.0

def calculate_ecs_scores_with_integrated_models(tweet_id, engine):
    """Calculate all 10 ECS model scores using integrated specialized logic."""
    
    # Define the 10 specialized models with their weights
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
    
    st.info("🔍 Running INTEGRATED ECS models with specialized logic... This may take a moment.")
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    scores = {}
    total_weight = 0
    weighted_sum = 0
    
    # Run each model function directly
    model_functions = [
        ('Authority_Signal_Manipulation', calculate_authority_signal_manipulation_score),
        ('Clickbait_Headline_Classifier', calculate_clickbait_headline_classifier_score),
        ('Content_Recycling_Detector', calculate_content_recycling_detector_score),
        ('Coordinated_Account_Network_Model', calculate_coordinated_account_network_score),
        ('Emotive_Manipulation_Detector', calculate_emotive_manipulation_score),
        ('Engagement_Mismatch_Detector', calculate_engagement_mismatch_score),
        ('Generic_Comment_Detector', calculate_generic_comment_score),
        ('Hyperbole_Falsehood_detector', calculate_hyperbole_falsehood_score),
        ('Rapid_Engagement_Spike_Detector', calculate_rapid_engagement_spike_score),
        ('Reply_Bait_Detector', calculate_reply_bait_score)
    ]
    
    for i, (model_name, model_func) in enumerate(model_functions):
        status_text.text(f"Running {model_name} with integrated specialized logic...")
        progress_bar.progress((i + 1) / len(model_functions))
        
        # Run the model function directly
        score = model_func(tweet_id, engine)
        scores[model_name] = score
        
        weight = models[model_name]
        total_weight += weight
        weighted_sum += score * weight
        
        time.sleep(0.1)  # Small delay for progress visibility
    
    # Calculate final weighted score
    final_score = weighted_sum / total_weight if total_weight > 0 else 0
    
    progress_bar.empty()
    status_text.empty()
    
    return scores, final_score, models

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
            
            if st.button("🔍 Analyze This Tweet with INTEGRATED Models"):
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
                
                # Run INTEGRATED ECS Models
                st.subheader("🔍 INTEGRATED ECS Model Analysis (Direct Specialized Logic)")
                
                # Run all ECS models with integrated specialized logic
                scores, final_score, weights = calculate_ecs_scores_with_integrated_models(selected_tweet['tweet_id'], engine)
                
                # Display individual model scores
                st.subheader("📊 Individual Model Scores (Integrated Specialized Logic)")
                
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
                st.subheader("🏆 Final ECS Score (Integrated Models)")
                
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
                st.subheader("🧮 Score Calculation Breakdown (Integrated Models)")
                
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
                
                **Note:** These scores are calculated using INTEGRATED specialized logic directly in the app!
                """)

def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">🔍 Engagement Concordance Score</h1>', unsafe_allow_html=True)
    st.markdown("### Integrated Model System - Direct Specialized Logic")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🐦 Tweet Analysis"]
    )
    
    # Get database connection
    engine = get_azure_database_engine()
    
    if engine is None:
        st.error("❌ Cannot proceed without database connection.")
        return
    
    # Page routing
    if page == "🏠 Home":
        st.subheader("Welcome to the INTEGRATED ECS System")
        st.write("""
        This is the **integrated ECS system** that runs all specialized model logic directly in the main app.
        
        **What's Different:**
        - 🐦 **Integrated Model Logic**: Each of the 10 models runs its specialized logic directly
        - 📊 **Specialized Scoring**: Authority, Clickbait, Content Recycling, etc. - each with unique algorithms
        - 🔍 **Database Connected**: Uses Azure SQL Database with your migrated data
        - 🚀 **No Subprocess Issues**: All logic runs in the same Python environment
        
        **Models Integrated:**
        1. **Authority Signal Manipulation** - Expert/professional phrase detection + profile analysis
        2. **Clickbait Headline Classifier** - Sensational language detection
        3. **Content Recycling Detector** - Viral/sharing pattern detection
        4. **Coordinated Account Network** - Bot/automation pattern analysis
        5. **Emotive Manipulation** - Emotional language detection
        6. **Engagement Mismatch** - Engagement-baiting detection
        7. **Generic Comment** - Basic response detection
        8. **Hyperbole/Falsehood** - Extreme language detection
        9. **Rapid Engagement Spike** - Trending/viral detection
        10. **Reply Bait** - Question/opinion baiting detection
        
        **Note:** This version integrates all specialized logic directly, avoiding subprocess complexity!
        """)
        
        # Show system status
        st.subheader("🔄 System Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("✅ Database Connected")
        with col2:
            st.info("📊 Models Integrated")
        with col3:
            st.info("🔧 Direct Logic Active")
    
    elif page == "🐦 Tweet Analysis":
        show_tweet_selection(engine)
    
    # Footer
    st.markdown("---")
    st.markdown("**ECS System v3.0** | Integrated Models | Azure SQL Database | Direct Specialized Logic")

if __name__ == "__main__":
    main()
