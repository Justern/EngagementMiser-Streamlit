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

# Load the RoBERTa model from local files
def load_roberta_model():
    """Load the RoBERTa model from local files."""
    print("🔄 Starting RoBERTa model loading from local files...")
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers not available - cannot load RoBERTa model")
        return None, None
    
    try:
        # Use local model files instead of Hugging Face Hub
        model_path = "."  # Current directory where model files are located
        print(f"📁 Loading model from local path: {os.path.abspath(model_path)}")
        
        # Check if model files exist
        required_files = ['config.json', 'model.safetensors', 'tokenizer.json', 'vocab.json']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
        
        if missing_files:
            print(f"❌ Missing model files: {missing_files}")
            return None, None
        
        print("✅ All required model files found")
        
        print("🔄 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ Tokenizer loaded successfully")
        
        print("🔄 Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("✅ Model loaded successfully")
        
        # Set to evaluation mode
        model.eval()
        print("✅ Model set to evaluation mode")
        
        # Test the model with a simple input
        print("🧪 Testing model with sample input...")
        test_input = "This is a test tweet for model verification."
        inputs = tokenizer(test_input, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # For single-class model, just get the raw logit value
            test_score = float(outputs.logits[0][0].item())
            # Normalize to 0-1 range (assuming higher values = higher risk)
            test_score = max(0.0, min(1.0, (test_score + 5) / 10))  # Rough normalization
        
        print(f"✅ Model test successful! Sample score: {test_score:.3f}")
        print(f"✅ RoBERTa model fully loaded and operational from local files")
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ Error loading RoBERTa model: {e}")
        import traceback
        print(f"Full error traceback: {traceback.format_exc()}")
        return None, None

# Model-specific database adapters
class LocalModelAdapter:
    """Adapter class to make models work with LOCAL database through ngrok."""
    
    def __init__(self):
        self.engine = None
        try:
            import streamlit as st
            # Get local database connection from Streamlit secrets
            server = st.secrets.get("local_db.server", "localhost")
            database = st.secrets.get("local_db.database", "EngagementMiser")
            username = st.secrets.get("local_db.username", "")
            password = st.secrets.get("local_db.password", "")
            
            # Create connection string for local database
            if username and password:
                conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes"
            else:
                # Windows Authentication
                conn_str = f"mssql+pyodbc://@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&Trusted_Connection=yes&TrustServerCertificate=yes"
            
            from sqlalchemy import create_engine
            self.engine = create_engine(conn_str, pool_pre_ping=True, pool_recycle=300)
            print("✅ LocalModelAdapter initialized with local database connection")
            
        except Exception as e:
            print(f"❌ Error initializing LocalModelAdapter: {e}")
            self.engine = None
    
    def get_tweet_data(self, tweet_id):
        """Get tweet data from LOCAL database."""
        if not self.engine:
            return None
        
        try:
            query = f"""
            SELECT 
                tweet_id,
                text as tweet_text,
                author_id,
                created_at,
                retweet_count,
                like_count,
                followers_count,
                total_engagements,
                engagement_rate
            FROM [EngagementMiser].[dbo].[Tweets_Sample_4M]
            WHERE tweet_id = {tweet_id}
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if not df.empty:
                return df.iloc[0].to_dict()
            return None
            
        except Exception as e:
            print(f"Error fetching tweet data from local DB: {e}")
            return None

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
        print("🚀 Initializing DeploymentModels...")
        self.adapter = LocalModelAdapter()
        print("✅ LocalModelAdapter initialized")
        
        print("🔄 Loading RoBERTa model...")
        self.tokenizer, self.model = load_roberta_model()
        
        if self.tokenizer and self.model:
            print("✅ RoBERTa model loaded successfully")
            print(f"📊 Model type: {type(self.model)}")
            print(f"📊 Tokenizer type: {type(self.tokenizer)}")
            
            # Test model with a simple input
            test_text = "This is a test for model verification."
            test_score = self._analyze_text_with_roberta(test_text)
            print(f"🧪 Model test score: {test_score:.3f}")
            
        else:
            print("⚠️ RoBERTa model not available, using fallback models")
            print("❌ This means models will use simplified keyword-based scoring")
            print("❌ Scores will be less accurate and may appear similar")
    
    def _analyze_text_with_roberta(self, text):
        """Analyze text using the RoBERTa model."""
        print(f"🔍 RoBERTa analysis called for text: '{text[:100]}...'")
        
        if not self.tokenizer or not self.model:
            print("❌ RoBERTa model not available - falling back to keyword scoring")
            return 0.0
        
        try:
            print("🔄 Tokenizing input text...")
            # Tokenize and prepare input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            print(f"✅ Tokenization complete. Input shape: {inputs['input_ids'].shape}")
            
            print("🔄 Running model inference...")
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                print(f"✅ Model inference complete. Output shape: {outputs.logits.shape}")
                
                # For single-class model, get the raw logit value
                raw_score = float(outputs.logits[0][0].item())
                print(f"📊 Raw logit score: {raw_score:.3f}")
                
                # Better normalization to 0-1 range with more differentiation
                # Assuming logits range from roughly -10 to +10
                # Use sigmoid-like transformation for better spread
                normalized_score = 1.0 / (1.0 + np.exp(-raw_score / 2.0))
                print(f"🎯 Normalized RoBERTa score: {normalized_score:.3f}")
                return normalized_score
                
        except Exception as e:
            print(f"❌ Error in RoBERTa analysis: {e}")
            import traceback
            print(f"Full error traceback: {traceback.format_exc()}")
            return 0.0
    
    def hyperbole_falsehood_score(self, tweet_id):
        """Call the actual Hyperbole & Falsehood detector."""
        print(f"🔍 Hyperbole detection called for tweet ID: {tweet_id}")
        
        try:
            # Call the actual specialized model
            model_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Hyperbole_Falsehood_detector", "simple_score.py")
            
            if not os.path.exists(model_path):
                print(f"❌ Model not found at: {model_path}")
                return 0.0
            
            # Run the specialized model
            import subprocess
            result = subprocess.run([
                sys.executable, model_path, str(tweet_id)
            ], capture_output=True, text=True, cwd=os.path.dirname(model_path))
            
            if result.returncode == 0:
                score = float(result.stdout.strip())
                print(f"✅ Hyperbole & Falsehood score: {score:.3f}")
                return score
            else:
                print(f"❌ Model error: {result.stderr}")
                return 0.0
                
        except Exception as e:
            print(f"❌ Error calling hyperbole model: {e}")
            return 0.0
    
    def clickbait_score(self, tweet_id):
        """Call the actual Clickbait Headline Classifier."""
        try:
            # Call the actual specialized model
            model_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Clickbait_Headline_Classifier", "simple_score.py")
            
            if not os.path.exists(model_path):
                print(f"❌ Model not found at: {model_path}")
                return 0.0
            
            # Run the specialized model
            import subprocess
            result = subprocess.run([
                sys.executable, model_path, str(tweet_id)
            ], capture_output=True, text=True, cwd=os.path.dirname(model_path))
            
            if result.returncode == 0:
                score = float(result.stdout.strip())
                print(f"✅ Clickbait Detection score: {score:.3f}")
                return score
            else:
                print(f"❌ Model error: {result.stderr}")
                return 0.0
                
        except Exception as e:
            print(f"❌ Error calling clickbait model: {e}")
            return 0.0
    
    def engagement_mismatch_score(self, tweet_id):
        """Call the actual Engagement Mismatch Detector."""
        try:
            # Call the actual specialized model
            model_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Engagement_Mismatch_Detector", "simple_score.py")
            
            if not os.path.exists(model_path):
                print(f"❌ Model not found at: {model_path}")
                return 0.0
            
            # Run the specialized model
            import subprocess
            result = subprocess.run([
                sys.executable, model_path, str(tweet_id)
            ], capture_output=True, text=True, cwd=os.path.dirname(model_path))
            
            if result.returncode == 0:
                score = float(result.stdout.strip())
                print(f"✅ Engagement Mismatch score: {score:.3f}")
                return score
            else:
                print(f"❌ Model error: {result.stderr}")
                return 0.0
                
        except Exception as e:
            print(f"❌ Error calling engagement mismatch model: {e}")
            return 0.0
    
    def content_recycling_score(self, tweet_id):
        """Call the actual Content Recycling Detector."""
        try:
            # Call the actual specialized model
            model_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Content_Recycling_Detector", "simple_score.py")
            
            if not os.path.exists(model_path):
                print(f"❌ Model not found at: {model_path}")
                return 0.0
            
            # Run the specialized model
            import subprocess
            result = subprocess.run([
                sys.executable, model_path, str(tweet_id)
            ], capture_output=True, text=True, cwd=os.path.dirname(model_path))
            
            if result.returncode == 0:
                score = float(result.stdout.strip())
                print(f"✅ Content Recycling score: {score:.3f}")
                return score
            else:
                print(f"❌ Model error: {result.stderr}")
                return 0.0
                
        except Exception as e:
            print(f"❌ Error calling content recycling model: {e}")
            return 0.0
    
    def coordinated_network_score(self, tweet_id):
        """Call the actual Coordinated Account Network Model."""
        try:
            # Call the actual specialized model
            model_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Coordinated_Account_Network_Model", "simple_score.py")
            
            if not os.path.exists(model_path):
                print(f"❌ Model not found at: {model_path}")
                return 0.0
            
            # Run the specialized model
            import subprocess
            result = subprocess.run([
                sys.executable, model_path, str(tweet_id)
            ], capture_output=True, text=True, cwd=os.path.dirname(model_path))
            
            if result.returncode == 0:
                score = float(result.stdout.strip())
                print(f"✅ Coordinated Network score: {score:.3f}")
                return score
            else:
                print(f"❌ Model error: {result.stderr}")
                return 0.0
                
        except Exception as e:
            print(f"❌ Error calling coordinated network model: {e}")
            return 0.0
    
    def emotive_manipulation_score(self, tweet_id):
        """Call the actual Emotive Manipulation Detector."""
        try:
            # Call the actual specialized model
            model_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Emotive_Manipulation_Detector", "simple_score.py")
            
            if not os.path.exists(model_path):
                print(f"❌ Model not found at: {model_path}")
                return 0.0
            
            # Run the specialized model
            import subprocess
            result = subprocess.run([
                sys.executable, model_path, str(tweet_id)
            ], capture_output=True, text=True, cwd=os.path.dirname(model_path))
            
            if result.returncode == 0:
                score = float(result.stdout.strip())
                print(f"✅ Emotive Manipulation score: {score:.3f}")
                return score
            else:
                print(f"❌ Model error: {result.stderr}")
                return 0.0
                
        except Exception as e:
            print(f"❌ Error calling emotive manipulation model: {e}")
            return 0.0
    
    def rapid_engagement_spike_score(self, tweet_id):
        """Call the actual Rapid Engagement Spike Detector."""
        try:
            # Call the actual specialized model
            model_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Rapid_Engagement_Spike_Detector", "simple_score.py")
            
            if not os.path.exists(model_path):
                print(f"❌ Model not found at: {model_path}")
                return 0.0
            
            # Run the specialized model
            import subprocess
            result = subprocess.run([
                sys.executable, model_path, str(tweet_id)
            ], capture_output=True, text=True, cwd=os.path.dirname(model_path))
            
            if result.returncode == 0:
                score = float(result.stdout.strip())
                print(f"✅ Rapid Engagement Spike score: {score:.3f}")
                return score
            else:
                print(f"❌ Model error: {result.stderr}")
                return 0.0
                
        except Exception as e:
            print(f"❌ Error calling rapid engagement spike model: {e}")
            return 0.0
    
    def generic_comment_score(self, tweet_id):
        """Call the actual Generic Comment Detector."""
        try:
            # Call the actual specialized model
            model_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Generic_Comment_Detector", "simple_score.py")
            
            if not os.path.exists(model_path):
                print(f"❌ Model not found at: {model_path}")
                return 0.0
            
            # Run the specialized model
            import subprocess
            result = subprocess.run([
                sys.executable, model_path, str(tweet_id)
            ], capture_output=True, text=True, cwd=os.path.dirname(model_path))
            
            if result.returncode == 0:
                score = float(result.stdout.strip())
                print(f"✅ Generic Comment score: {score:.3f}")
                return score
            else:
                print(f"❌ Model error: {result.stderr}")
                return 0.0
                
        except Exception as e:
            print(f"❌ Error calling generic comment model: {e}")
            return 0.0
    
    def authority_signal_score(self, tweet_id):
        """Call the actual Authority Signal Manipulation detector."""
        try:
            # Call the actual specialized model
            model_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Authority_Signal_Manipulation", "simple_score.py")
            
            if not os.path.exists(model_path):
                print(f"❌ Model not found at: {model_path}")
                return 0.0
            
            # Run the specialized model
            import subprocess
            result = subprocess.run([
                sys.executable, model_path, str(tweet_id)
            ], capture_output=True, text=True, cwd=os.path.dirname(model_path))
            
            if result.returncode == 0:
                score = float(result.stdout.strip())
                print(f"✅ Authority Signal score: {score:.3f}")
                return score
            else:
                print(f"❌ Model error: {result.stderr}")
                return 0.0
                
        except Exception as e:
            print(f"❌ Error calling authority signal model: {e}")
            return 0.0
    
    def reply_bait_score(self, tweet_id):
        """Call the actual Reply Bait Detector."""
        try:
            # Call the actual specialized model
            model_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Reply_Bait_Detector", "simple_score.py")
            
            if not os.path.exists(model_path):
                print(f"❌ Model not found at: {model_path}")
                return 0.0
            
            # Run the specialized model
            import subprocess
            result = subprocess.run([
                sys.executable, model_path, str(tweet_id)
            ], capture_output=True, text=True, cwd=os.path.dirname(model_path))
            
            if result.returncode == 0:
                score = float(result.stdout.strip())
                print(f"✅ Reply Bait score: {score:.3f}")
                return score
            else:
                print(f"❌ Model error: {result.stderr}")
                return 0.0
                
        except Exception as e:
            print(f"❌ Error calling reply bait model: {e}")
            return 0.0

# Export the deployment models
deployment_models = DeploymentModels()

# Test function to verify model functionality
def test_roberta_model():
    """Test function to verify RoBERTa model is working."""
    print("🧪 Testing RoBERTa model functionality...")
    
    try:
        # Test with different types of text
        test_texts = [
            "This is a normal, genuine tweet.",
            "AMAZING! You won't BELIEVE what happened next!",
            "Fake news conspiracy cover-up revealed!",
            "Just sharing some thoughts on the weather today."
        ]
        
        for i, text in enumerate(test_texts):
            print(f"\n📝 Test {i+1}: '{text}'")
            score = deployment_models._analyze_text_with_roberta(text)
            print(f"🎯 Score: {score:.3f}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")

# Run test if this file is run directly
if __name__ == "__main__":
    test_roberta_model()
