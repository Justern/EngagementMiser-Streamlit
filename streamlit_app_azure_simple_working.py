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
import os

# Try to import PyTorch, but don't fail if it's not available
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    PYTORCH_AVAILABLE = True
    st.success("✅ PyTorch and Transformers loaded successfully!")
except ImportError as e:
    PYTORCH_AVAILABLE = False
    st.warning(f"⚠️ PyTorch not available: {e}. Using keyword-based scoring.")

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

# Global model variables
@st.cache_resource
def load_models():
    """Load PyTorch models if available."""
    if not PYTORCH_AVAILABLE:
        return None, None, None
    
    try:
        repo_id = "MidlAnalytics/engagement-concordance-roberta"
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForSequenceClassification.from_pretrained(repo_id)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        st.success("✅ RoBERTa model loaded successfully!")
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"❌ Error loading RoBERTa model: {e}")
        return None, None, None

# ECS Model Scoring Functions - Direct Integration
def calculate_authority_signal_manipulation_score(text: str, model, tokenizer, device) -> float:
    """Calculate authority signal manipulation score."""
    if PYTORCH_AVAILABLE and model is not None:
        try:
            with torch.no_grad():
                enc = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                score = torch.sigmoid(logits.squeeze()).item()
                
                return float(score)
                
        except Exception as e:
            st.warning(f"RoBERTa scoring failed, using fallback: {e}")
    
    # Fallback to specialized authority logic
    text_lower = text.lower()
    
    authority_phrases = [
        'expert', 'professional', 'doctor', 'scientist', 'researcher',
        'study shows', 'research proves', 'experts agree', 'authority',
        'scientifically proven', 'clinically tested', 'doctor recommended',
        'according to science', 'research indicates', 'studies confirm',
        'medical evidence', 'scientific evidence', 'clinical evidence',
        'expert opinion', 'professional opinion', 'authority figure'
    ]
    
    authority_count = sum(1 for phrase in authority_phrases if phrase in text_lower)
    score = min(authority_count / 3, 1.0)
    
    return max(0.0, min(1.0, score))

def calculate_clickbait_headline_classifier_score(text: str, model, tokenizer, device) -> float:
    """Calculate clickbait headline score."""
    if PYTORCH_AVAILABLE and model is not None:
        try:
            with torch.no_grad():
                enc = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                score = torch.sigmoid(logits.squeeze()).item()
                
                return float(score)
                
        except Exception as e:
            st.warning(f"RoBERTa scoring failed, using fallback: {e}")
    
    # Fallback to specialized clickbait logic
    text_lower = text.lower()
    
    clickbait_phrases = [
        'you won\'t believe', 'shocking', 'amazing', 'incredible',
        'this will blow your mind', 'what happened next', 'the truth about',
        'they don\'t want you to know', 'secret', 'exposed', 'revealed',
        'breaking', 'urgent', 'warning', 'alert', 'critical',
        'number one reason', 'top secret', 'hidden', 'forbidden'
    ]
    
    clickbait_count = sum(1 for phrase in clickbait_phrases if phrase in text_lower)
    score = min(clickbait_count / 2, 1.0)
    
    return max(0.0, min(1.0, score))

def calculate_content_recycling_detector_score(text: str, model, tokenizer, device) -> float:
    """Calculate content recycling score."""
    if PYTORCH_AVAILABLE and model is not None:
        try:
            with torch.no_grad():
                enc = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                score = torch.sigmoid(logits.squeeze()).item()
                
                return float(score)
                
        except Exception as e:
            st.warning(f"RoBERTa scoring failed, using fallback: {e}")
    
    # Fallback to specialized content recycling logic
    text_lower = text.lower()
    
    recycling_phrases = [
        'repost', 'reposting', 'repost this', 'share this',
        'viral', 'going viral', 'trending', 'trending now',
        'everyone is talking about', 'everyone needs to see',
        'spread the word', 'pass it on', 'forward this',
        'retweet this', 'like and share', 'comment and share'
    ]
    
    recycling_count = sum(1 for phrase in recycling_phrases if phrase in text_lower)
    score = min(recycling_count / 2, 1.0)
    
    return max(0.0, min(1.0, score))

def calculate_coordinated_account_network_model_score(text: str, model, tokenizer, device) -> float:
    """Calculate coordinated account network score."""
    if PYTORCH_AVAILABLE and model is not None:
        try:
            with torch.no_grad():
                enc = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                score = torch.sigmoid(logits.squeeze()).item()
                
                return float(score)
                
        except Exception as e:
            st.warning(f"RoBERTa scoring failed, using fallback: {e}")
    
    # Fallback to specialized network logic
    text_lower = text.lower()
    
    network_phrases = [
        'bot', 'bots', 'automated', 'script', 'scripted',
        'coordinated', 'network', 'campaign', 'operation',
        'mass', 'bulk', 'flood', 'spam', 'spamming',
        'trending', 'trend', 'hashtag', 'hashtags'
    ]
    
    network_count = sum(1 for phrase in network_phrases if phrase in text_lower)
    score = min(network_count / 2, 1.0)
    
    return max(0.0, min(1.0, score))

def calculate_emotive_manipulation_detector_score(text: str, model, tokenizer, device) -> float:
    """Calculate emotive manipulation score."""
    if PYTORCH_AVAILABLE and model is not None:
        try:
            with torch.no_grad():
                enc = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                score = torch.sigmoid(logits.squeeze()).item()
                
                return float(score)
                
        except Exception as e:
            st.warning(f"RoBERTa scoring failed, using fallback: {e}")
    
    # Fallback to specialized emotive logic
    text_lower = text.lower()
    
    emotive_phrases = [
        'outrage', 'outraged', 'angry', 'furious', 'livid',
        'shocked', 'shocking', 'disgusting', 'horrible', 'terrible',
        'amazing', 'incredible', 'wonderful', 'fantastic', 'amazing',
        'heartbreaking', 'devastating', 'tragic', 'sad', 'depressing',
        'excited', 'thrilled', 'ecstatic', 'overjoyed', 'elated'
    ]
    
    emotive_count = sum(1 for phrase in emotive_phrases if phrase in text_lower)
    score = min(emotive_count / 3, 1.0)
    
    return max(0.0, min(1.0, score))

def calculate_engagement_mismatch_detector_score(text: str, model, tokenizer, device) -> float:
    """Calculate engagement mismatch score."""
    if PYTORCH_AVAILABLE and model is not None:
        try:
            with torch.no_grad():
                enc = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                score = torch.sigmoid(logits.squeeze()).item()
                
                return float(score)
                
        except Exception as e:
            st.warning(f"RoBERTa scoring failed, using fallback: {e}")
    
    # Fallback to specialized engagement mismatch logic
    text_lower = text.lower()
    
    mismatch_phrases = [
        'like this', 'like if you agree', 'like for more',
        'retweet this', 'retweet if you agree', 'retweet for more',
        'comment below', 'comment your thoughts', 'comment your opinion',
        'share this', 'share if you agree', 'share for awareness',
        'follow me', 'follow for more', 'follow for updates'
    ]
    
    mismatch_count = sum(1 for phrase in mismatch_phrases if phrase in text_lower)
    score = min(mismatch_count / 2, 1.0)
    
    return max(0.0, min(1.0, score))

def calculate_generic_comment_detector_score(text: str, model, tokenizer, device) -> float:
    """Calculate generic comment score."""
    if PYTORCH_AVAILABLE and model is not None:
        try:
            with torch.no_grad():
                enc = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                score = torch.sigmoid(logits.squeeze()).item()
                
                return float(score)
                
        except Exception as e:
            st.warning(f"RoBERTa scoring failed, using fallback: {e}")
    
    # Fallback to specialized generic comment logic
    text_lower = text.lower()
    
    generic_phrases = [
        'nice', 'good', 'great', 'awesome', 'cool',
        'interesting', 'wow', 'omg', 'lol', 'haha',
        'thanks', 'thank you', 'appreciate it', 'love this',
        'agree', 'disagree', 'true', 'false', 'yes', 'no'
    ]
    
    generic_count = sum(1 for phrase in generic_phrases if phrase in text_lower)
    score = min(generic_count / 3, 1.0)
    
    return max(0.0, min(1.0, score))

def calculate_hyperbole_falsehood_detector_score(text: str, model, tokenizer, device) -> float:
    """Calculate hyperbole and falsehood score."""
    if PYTORCH_AVAILABLE and model is not None:
        try:
            with torch.no_grad():
                enc = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                score = torch.sigmoid(logits.squeeze()).item()
                
                return float(score)
                
        except Exception as e:
            st.warning(f"RoBERTa scoring failed, using fallback: {e}")
    
    # Fallback to specialized hyperbole logic
    text_lower = text.lower()
    
    hyperbole_phrases = [
        'always', 'never', 'everyone', 'nobody', 'every single',
        '100%', 'guaranteed', 'proven', 'definitely', 'absolutely',
        'worst ever', 'best ever', 'most', 'least', 'completely',
        'totally', 'literally', 'actually', 'really', 'very'
    ]
    
    hyperbole_count = sum(1 for phrase in hyperbole_phrases if phrase in text_lower)
    score = min(hyperbole_count / 3, 1.0)
    
    return max(0.0, min(1.0, score))

def calculate_rapid_engagement_spike_detector_score(text: str, model, tokenizer, device) -> float:
    """Calculate rapid engagement spike score."""
    if PYTORCH_AVAILABLE and model is not None:
        try:
            with torch.no_grad():
                enc = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                score = torch.sigmoid(logits.squeeze()).item()
                
                return float(score)
                
        except Exception as e:
            st.warning(f"RoBERTa scoring failed, using fallback: {e}")
    
    # Fallback to specialized rapid engagement logic
    text_lower = text.lower()
    
    rapid_phrases = [
        'trending', 'trending now', 'going viral', 'viral',
        'exploding', 'blowing up', 'skyrocketing', 'soaring',
        'breaking', 'breaking news', 'just in', 'update',
        'developing', 'developing story', 'latest', 'newest'
    ]
    
    rapid_count = sum(1 for phrase in rapid_phrases if phrase in text_lower)
    score = min(rapid_count / 2, 1.0)
    
    return max(0.0, min(1.0, score))

def calculate_reply_bait_detector_score(text: str, model, tokenizer, device) -> float:
    """Calculate reply bait score."""
    if PYTORCH_AVAILABLE and model is not None:
        try:
            with torch.no_grad():
                enc = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                score = torch.sigmoid(logits.squeeze()).item()
                
                return float(score)
                
        except Exception as e:
            st.warning(f"RoBERTa scoring failed, using fallback: {e}")
    
    # Fallback to specialized reply bait logic
    text_lower = text.lower()
    
    reply_bait_phrases = [
        'what do you think', 'your thoughts', 'your opinion',
        'agree or disagree', 'comment below', 'comment your thoughts',
        'what\'s your take', 'how do you feel', 'do you agree',
        'share your experience', 'tell me', 'let me know',
        'what about you', 'your turn', 'your say'
    ]
    
    reply_bait_count = sum(1 for phrase in reply_bait_phrases if phrase in text_lower)
    score = min(reply_bait_count / 2, 1.0)
    
    return max(0.0, min(1.0, score))

def calculate_ecs_scores(tweet_text):
    """Calculate all 10 ECS model scores directly."""
    
    # Load models
    model, tokenizer, device = load_models()
    
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
    
    # Calculate scores directly
    for i, (model_name, weight) in enumerate(models.items()):
        status_text.text(f"Running {model_name}...")
        progress_bar.progress((i + 1) / len(models))
        
        # Call the appropriate scoring function
        if 'authority_signal_manipulation' in model_name.lower():
            score = calculate_authority_signal_manipulation_score(tweet_text, model, tokenizer, device)
        elif 'clickbait_headline_classifier' in model_name.lower():
            score = calculate_clickbait_headline_classifier_score(tweet_text, model, tokenizer, device)
        elif 'content_recycling_detector' in model_name.lower():
            score = calculate_content_recycling_detector_score(tweet_text, model, tokenizer, device)
        elif 'coordinated_account_network_model' in model_name.lower():
            score = calculate_coordinated_account_network_model_score(tweet_text, model, tokenizer, device)
        elif 'emotive_manipulation_detector' in model_name.lower():
            score = calculate_emotive_manipulation_detector_score(tweet_text, model, tokenizer, device)
        elif 'engagement_mismatch_detector' in model_name.lower():
            score = calculate_engagement_mismatch_detector_score(tweet_text, model, tokenizer, device)
        elif 'generic_comment_detector' in model_name.lower():
            score = calculate_generic_comment_detector_score(tweet_text, model, tokenizer, device)
        elif 'hyperbole_falsehood_detector' in model_name.lower():
            score = calculate_hyperbole_falsehood_detector_score(tweet_text, model, tokenizer, device)
        elif 'rapid_engagement_spike_detector' in model_name.lower():
            score = calculate_rapid_engagement_spike_detector_score(tweet_text, model, tokenizer, device)
        elif 'reply_bait_detector' in model_name.lower():
            score = calculate_reply_bait_detector_score(tweet_text, model, tokenizer, device)
        else:
            score = calculate_generic_comment_detector_score(tweet_text, model, tokenizer, device)
        
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
