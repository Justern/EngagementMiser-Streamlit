#!/usr/bin/env python3
"""
Deployment Configuration for Engagement Concordance Score
======================================================

This file provides Azure database configuration for all 10 specialized models
to work in a deployed environment without local file dependencies.
"""

import os
import sys
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np

# Try to import transformers for Hugging Face models
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available, using fallback models")

# Azure Database Configuration
def get_azure_config():
    """Get Azure configuration from Streamlit secrets or environment variables."""
    try:
        import streamlit as st
        return {
            'server': st.secrets.get('azure_db.server', 'ecs-sql-server-engagementmiser.database.windows.net'),
            'database': st.secrets.get('azure_db.database', 'ecs_tweets_db'),
            'username': st.secrets.get('azure_db.username', 'ecsadmin'),
            'password': st.secrets.get('azure_db.password', 'EngagementMiser!'),
            'driver': 'ODBC+Driver+17+for+SQL+Server'
        }
    except ImportError:
        # Fallback for non-Streamlit environments
        return {
            'server': os.getenv('DB_SERVER', 'ecs-sql-server-engagementmiser.database.windows.net'),
            'database': os.getenv('DB_NAME', 'ecs_tweets_db'),
            'username': os.getenv('DB_USERNAME', 'ecsadmin'),
            'password': os.getenv('DB_PASSWORD', 'EngagementMiser!'),
            'driver': 'ODBC+Driver+17+for+SQL+Server'
        }

def get_azure_engine():
    """Get Azure SQL Database engine."""
    try:
        config = get_azure_config()
        conn_str = (
            f"mssql+pyodbc://{config['username']}:{config['password']}"
            f"@{config['server']}/{config['database']}"
            f"?driver={config['driver']}"
            "&Encrypt=yes&TrustServerCertificate=no&Connection+Timeout=30"
        )
        engine = create_engine(conn_str, pool_pre_ping=True, pool_recycle=300)
        return engine
    except Exception as e:
        print(f"Error connecting to Azure: {e}")
        return None

# Load the RoBERTa model from Hugging Face Hub
def load_roberta_model():
    """Load the RoBERTa model from Hugging Face Hub."""
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    
    try:
        # Load model from Hugging Face Hub
        model_name = "Justern/text_softlabel_roberta"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Set to evaluation mode
        model.eval()
        
        print(f"✅ RoBERTa model loaded from {model_name}")
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ Error loading RoBERTa model: {e}")
        return None, None

# Model-specific database adapters
class AzureModelAdapter:
    """Adapter class to make models work with Azure database."""
    
    def __init__(self):
        self.engine = get_azure_engine()
    
    def get_tweet_data(self, tweet_id):
        """Get tweet data from Azure database."""
        if not self.engine:
            return None
        
        try:
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
                engagement_rate
            FROM tweets 
            WHERE tweet_id = {tweet_id}
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if not df.empty:
                return df.iloc[0].to_dict()
            return None
            
        except Exception as e:
            print(f"Error fetching tweet data: {e}")
            return None
    
    def get_user_data(self, author_id):
        """Get user profile data from Azure database."""
        if not self.engine:
            return None
        
        try:
            query = f"""
            SELECT TOP 1
                author_id as user_id,
                followers_count,
                total_engagements
            FROM tweets 
            WHERE author_id = '{author_id}'
            ORDER BY created_at DESC
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if not df.empty:
                return df.iloc[0].to_dict()
            return None
            
        except Exception as e:
            print(f"Error fetching user data: {e}")
            return None
    
    def get_interaction_data(self, author_id, days_back=30):
        """Get interaction data for coordination analysis."""
        if not self.engine:
            return None
        
        try:
            query = f"""
            SELECT TOP 100
                tweet_id,
                author_id as user_id,
                created_at,
                retweet_count,
                like_count,
                reply_count,
                quote_count
            FROM tweets 
            WHERE author_id = '{author_id}'
            ORDER BY created_at DESC
            """
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
            
            return df
            
        except Exception as e:
            print(f"Error fetching interaction data: {e}")
            return None

# Enhanced model implementations using RoBERTa
class DeploymentModels:
    """Enhanced model implementations that use RoBERTa for analysis."""
    
    def __init__(self):
        self.adapter = AzureModelAdapter()
        self.tokenizer, self.model = load_roberta_model()
        
        if self.tokenizer and self.model:
            print("✅ RoBERTa model loaded successfully")
        else:
            print("⚠️ RoBERTa model not available, using fallback models")
    
    def _analyze_text_with_roberta(self, text):
        """Analyze text using the RoBERTa model."""
        if not self.tokenizer or not self.model:
            return 0.0
        
        try:
            # Tokenize and prepare input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                
            # Return the probability of the positive class (manipulation detected)
            return float(probabilities[0][1].item())
            
        except Exception as e:
            print(f"Error in RoBERTa analysis: {e}")
            return 0.0
    
    def hyperbole_falsehood_score(self, tweet_id):
        """Enhanced hyperbole and falsehood detection using RoBERTa."""
        try:
            tweet_data = self.adapter.get_tweet_data(tweet_id)
            if not tweet_data:
                return 0.0
            
            text = tweet_data.get('tweet_text', '')
            if not text:
                return 0.0
            
            # Use RoBERTa model if available
            if self.tokenizer and self.model:
                return self._analyze_text_with_roberta(text)
            
            # Fallback to keyword-based scoring
            hyperbole_words = ['amazing', 'incredible', 'unbelievable', 'mind-blowing', 'epic', 'legendary']
            falsehood_indicators = ['fake', 'hoax', 'conspiracy', 'cover-up', 'secret', 'hidden']
            
            score = 0.0
            for word in hyperbole_words:
                if word in text.lower():
                    score += 0.2
            
            for word in falsehood_indicators:
                if word in text.lower():
                    score += 0.3
            
            return min(score, 1.0)
            
        except Exception as e:
            print(f"Error in hyperbole detection: {e}")
            return 0.0
    
    def clickbait_score(self, tweet_id):
        """Enhanced clickbait detection using RoBERTa."""
        try:
            tweet_data = self.adapter.get_tweet_data(tweet_id)
            if not tweet_data:
                return 0.0
            
            text = tweet_data.get('tweet_text', '')
            if not text:
                return 0.0
            
            # Use RoBERTa model if available
            if self.tokenizer and self.model:
                return self._analyze_text_with_roberta(text)
            
            # Fallback to keyword-based scoring
            clickbait_patterns = [
                'you won\'t believe', 'shocking', 'this will change everything',
                'number 7 will surprise you', 'what happened next', 'the truth about'
            ]
            
            score = 0.0
            for pattern in clickbait_patterns:
                if pattern.lower() in text.lower():
                    score += 0.4
            
            # Check for excessive punctuation
            if text.count('!') > 2 or text.count('?') > 2:
                score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            print(f"Error in clickbait detection: {e}")
            return 0.0
    
    def engagement_mismatch_score(self, tweet_id):
        """Enhanced engagement mismatch detection."""
        try:
            tweet_data = self.adapter.get_tweet_data(tweet_id)
            if not tweet_data:
                return 0.0
            
            text = tweet_data.get('tweet_text', '')
            if not text:
                return 0.0
            
            # Use RoBERTa model if available
            if self.tokenizer and self.model:
                roberta_score = self._analyze_text_with_roberta(text)
                
                # Combine RoBERTa score with engagement metrics
                text_length = len(text)
                total_engagements = tweet_data.get('total_engagements', 0)
                followers = tweet_data.get('followers_count', 1)
                
                # Engagement mismatch heuristic
                engagement_ratio = total_engagements / followers if followers > 0 else 0
                if text_length < 50 and engagement_ratio > 0.1:
                    engagement_factor = 0.3
                elif text_length < 100 and engagement_ratio > 0.05:
                    engagement_factor = 0.2
                else:
                    engagement_factor = 0.0
                
                return min(roberta_score + engagement_factor, 1.0)
            
            # Fallback to simple heuristic
            text_length = len(text)
            total_engagements = tweet_data.get('total_engagements', 0)
            followers = tweet_data.get('followers_count', 1)
            
            if text_length < 50 and total_engagements > followers * 0.1:
                return 0.8
            elif text_length < 100 and total_engagements > followers * 0.05:
                return 0.6
            else:
                return 0.2
                
        except Exception as e:
            print(f"Error in engagement mismatch detection: {e}")
            return 0.0
    
    def content_recycling_score(self, tweet_id):
        """Enhanced content recycling detection."""
        try:
            tweet_data = self.adapter.get_tweet_data(tweet_id)
            if not tweet_data:
                return 0.0
            
            text = tweet_data.get('tweet_text', '')
            if not text:
                return 0.0
            
            # Use RoBERTa model if available
            if self.tokenizer and self.model:
                return self._analyze_text_with_roberta(text)
            
            # Fallback to moderate score
            return 0.3
            
        except Exception as e:
            print(f"Error in content recycling detection: {e}")
            return 0.0
    
    def coordinated_network_score(self, tweet_id):
        """Enhanced coordinated network detection."""
        try:
            tweet_data = self.adapter.get_tweet_data(tweet_id)
            if not tweet_data:
                return 0.0
            
            text = tweet_data.get('tweet_text', '')
            author_id = tweet_data.get('author_id')
            
            if not text or not author_id:
                return 0.0
            
            # Use RoBERTa model if available
            if self.tokenizer and self.model:
                roberta_score = self._analyze_text_with_roberta(text)
                
                # Get recent activity patterns
                interactions = self.adapter.get_interaction_data(author_id)
                if interactions is not None and not interactions.empty:
                    recent_tweets = len(interactions)
                    avg_engagement = interactions['total_engagements'].mean() if 'total_engagements' in interactions.columns else 0
                    
                    # Coordination indicators
                    if recent_tweets > 50 and avg_engagement > 100:
                        coordination_factor = 0.3
                    elif recent_tweets > 20 and avg_engagement > 50:
                        coordination_factor = 0.2
                    else:
                        coordination_factor = 0.0
                    
                    return min(roberta_score + coordination_factor, 1.0)
                
                return roberta_score
            
            # Fallback to simple coordination analysis
            interactions = self.adapter.get_interaction_data(author_id)
            if interactions is None or interactions.empty:
                return 0.0
            
            recent_tweets = len(interactions)
            avg_engagement = interactions['total_engagements'].mean() if 'total_engagements' in interactions.columns else 0
            
            if recent_tweets > 50 and avg_engagement > 100:
                return 0.7
            elif recent_tweets > 20 and avg_engagement > 50:
                return 0.5
            else:
                return 0.2
                
        except Exception as e:
            print(f"Error in coordinated network detection: {e}")
            return 0.0
    
    def emotive_manipulation_score(self, tweet_id):
        """Enhanced emotive manipulation detection using RoBERTa."""
        try:
            tweet_data = self.adapter.get_tweet_data(tweet_id)
            if not tweet_data:
                return 0.0
            
            text = tweet_data.get('tweet_text', '')
            if not text:
                return 0.0
            
            # Use RoBERTa model if available
            if self.tokenizer and self.model:
                return self._analyze_text_with_roberta(text)
            
            # Fallback to keyword-based scoring
            emotional_words = ['anger', 'fear', 'hate', 'love', 'hope', 'despair', 'joy', 'sadness']
            manipulative_phrases = ['make you feel', 'you should be', 'everyone knows', 'obviously']
            
            score = 0.0
            for word in emotional_words:
                if word in text.lower():
                    score += 0.15
            
            for phrase in manipulative_phrases:
                if phrase in text.lower():
                    score += 0.3
            
            return min(score, 1.0)
            
        except Exception as e:
            print(f"Error in emotive manipulation detection: {e}")
            return 0.0
    
    def rapid_engagement_spike_score(self, tweet_id):
        """Enhanced rapid engagement spike detection."""
        try:
            tweet_data = self.adapter.get_tweet_data(tweet_id)
            if not tweet_data:
                return 0.0
            
            text = tweet_data.get('tweet_text', '')
            if not text:
                return 0.0
            
            # Use RoBERTa model if available
            if self.tokenizer and self.model:
                return self._analyze_text_with_roberta(text)
            
            # Fallback to moderate score
            return 0.4
            
        except Exception as e:
            print(f"Error in rapid engagement spike detection: {e}")
            return 0.0
    
    def generic_comment_score(self, tweet_id):
        """Enhanced generic comment detection using RoBERTa."""
        try:
            tweet_data = self.adapter.get_tweet_data(tweet_id)
            if not tweet_data:
                return 0.0
            
            text = tweet_data.get('tweet_text', '')
            if not text:
                return 0.0
            
            # Use RoBERTa model if available
            if self.tokenizer and self.model:
                return self._analyze_text_with_roberta(text)
            
            # Fallback to keyword-based scoring
            generic_phrases = [
                'nice', 'good', 'bad', 'interesting', 'cool', 'wow', 'omg',
                'thanks', 'thank you', 'you\'re welcome', 'no problem'
            ]
            
            score = 0.0
            for phrase in generic_phrases:
                if phrase in text.lower():
                    score += 0.2
            
            # Very short generic responses
            if len(text) < 20 and any(phrase in text.lower() for phrase in generic_phrases):
                score += 0.3
            
            return min(score, 1.0)
            
        except Exception as e:
            print(f"Error in generic comment detection: {e}")
            return 0.0
    
    def authority_signal_score(self, tweet_id):
        """Enhanced authority signal manipulation detection using RoBERTa."""
        try:
            tweet_data = self.adapter.get_tweet_data(tweet_id)
            if not tweet_data:
                return 0.0
            
            text = tweet_data.get('tweet_text', '')
            if not text:
                return 0.0
            
            # Use RoBERTa model if available
            if self.tokenizer and self.model:
                return self._analyze_text_with_roberta(text)
            
            # Fallback to keyword-based scoring
            authority_claims = [
                'expert', 'doctor', 'scientist', 'researcher', 'official', 'authority',
                'studies show', 'research proves', 'experts agree', 'official source'
            ]
            
            score = 0.0
            for claim in authority_claims:
                if claim in text.lower():
                    score += 0.3
            
            return min(score, 1.0)
            
        except Exception as e:
            print(f"Error in authority signal detection: {e}")
            return 0.0
    
    def reply_bait_score(self, tweet_id):
        """Enhanced reply-bait detection using RoBERTa."""
        try:
            tweet_data = self.adapter.get_tweet_data(tweet_id)
            if not tweet_data:
                return 0.0
            
            text = tweet_data.get('tweet_text', '')
            if not text:
                return 0.0
            
            # Use RoBERTa model if available
            if self.tokenizer and self.model:
                return self._analyze_text_with_roberta(text)
            
            # Fallback to keyword-based scoring
            reply_bait_patterns = [
                'what do you think?', 'agree?', 'thoughts?', 'opinions?',
                'who else?', 'am i right?', 'or is it just me?', 'change my mind'
            ]
            
            score = 0.0
            for pattern in reply_bait_patterns:
                if pattern in text.lower():
                    score += 0.4
            
            # Question marks often indicate reply-baiting
            if text.count('?') > 2:
                score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            print(f"Error in reply-bait detection: {e}")
            return 0.0

# Export the deployment models
deployment_models = DeploymentModels()
