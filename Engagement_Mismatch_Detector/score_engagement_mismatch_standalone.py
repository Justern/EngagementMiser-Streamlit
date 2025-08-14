#!/usr/bin/env python3
"""
Standalone Engagement Mismatch Detector Scorer
=============================================

This script loads a pre-trained engagement mismatch detection model and scores
individual tweets by their tweet_id to identify unexpectedly high engagement.

Usage:
    python score_engagement_mismatch_standalone.py <tweet_id>
    python score_engagement_mismatch_standalone.py --interactive
"""

import os
import sys
import argparse
import torch
import pandas as pd
from sqlalchemy import create_engine, text as sql_text
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Union, Tuple
import numpy as np

# Automatically navigate to the script's directory
SCRIPT_DIR = r"C:\Users\justi\OneDrive\Desktop\MSc. Data Science\DS Capstone\Supporting_Files\Models\Engagement_Mismatch_Detector"
os.chdir(SCRIPT_DIR)
print(f"📁 Working directory set to: {SCRIPT_DIR}")

# Add script directory to Python path
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Configuration - use absolute paths based on script location
MODEL_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "engagement_mismatch_detector")
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SQL connection details
SQL_SERVER = os.getenv("SQL_SERVER", "localhost")
SQL_DB = os.getenv("SQL_DB", "EngagementMiser")
SQL_DRIVER = os.getenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server")

# Connection string for SQL Server over ODBC with Windows Authentication
CONN_STR = (
    f"mssql+pyodbc://@{SQL_SERVER}/{SQL_DB}"
    f"?driver={SQL_DRIVER.replace(' ', '+')}"
    "&Trusted_Connection=yes"
    "&TrustServerCertificate=yes"
)

def load_model_and_tokenizer():
    """Load the pre-trained model and tokenizer."""
    try:
        # Load tokenizer and model from the checkpoints directory
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        model.eval()
        print(f"✅ Model loaded successfully: {os.path.basename(MODEL_PATH)}")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Make sure you have trained the model first using engagement_mismatch_detector.py")
        sys.exit(1)

def create_database_connection():
    """Create and test database connection."""
    try:
        engine = create_engine(CONN_STR, fast_executemany=True)
        # Test connection
        with engine.connect() as conn:
            conn.execute(sql_text("SELECT 1"))
        return engine
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)

def calculate_engagement_metrics(row) -> dict:
    """Calculate engagement metrics for a single tweet."""
    # SQLAlchemy Row objects can be accessed by index or converted to dict
    try:
        # Try to access by column name first (if it's a Row object)
        followers = max(row.followers_count, 1)
        total_engagement = row.like_count + row.retweet_count + row.reply_count + row.quote_count
    except AttributeError:
        # If that fails, try to access by index
        try:
            followers = max(row[6], 1)  # followers_count is at index 6
            total_engagement = row[2] + row[3] + row[4] + row[5]  # like, retweet, reply, quote counts
        except (IndexError, TypeError):
            # Last resort: convert to dict if possible
            row_dict = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
            followers = max(row_dict['followers_count'], 1)
            total_engagement = row_dict['like_count'] + row_dict['retweet_count'] + row_dict['reply_count'] + row_dict['quote_count']
    
    engagement_rate = total_engagement / followers
    engagement_per_1k = (total_engagement / followers) * 1000
    
    # Get individual rates using the same approach
    try:
        like_rate = row.like_count / followers
        retweet_rate = row.retweet_count / followers
        reply_rate = row.reply_count / followers
        quote_rate = row.quote_count / followers
    except AttributeError:
        try:
            like_rate = row[2] / followers
            retweet_rate = row[3] / followers
            reply_rate = row[4] / followers
            quote_rate = row[5] / followers
        except (IndexError, TypeError):
            row_dict = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
            like_rate = row_dict['like_count'] / followers
            retweet_rate = row_dict['retweet_count'] / followers
            reply_rate = row_dict['reply_count'] / followers
            quote_rate = row_dict['quote_count'] / followers
    
    return {
        'total_engagement': total_engagement,
        'engagement_rate': engagement_rate,
        'engagement_per_1k': engagement_per_1k,
        'like_rate': like_rate,
        'retweet_rate': retweet_rate,
        'reply_rate': reply_rate,
        'quote_rate': quote_rate
    }

def predict_engagement_mismatch(text: str, model, tokenizer) -> float:
    """Predict engagement mismatch score for a single text input."""
    model.eval()
    
    with torch.no_grad():
        enc = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        score = torch.sigmoid(logits.squeeze()).item()
        
        return float(score)

def get_mismatch_interpretation(score: float) -> str:
    """Get interpretation of the engagement mismatch score."""
    if score < 0.3:
        return "Normal engagement pattern"
    elif score < 0.5:
        return "Slightly elevated engagement"
    elif score < 0.7:
        return "High engagement mismatch - potential viral content"
    else:
        return "Very high engagement mismatch - likely manipulation or viral"

def score_tweet_id(tweet_id: Union[str, int], model, tokenizer, engine) -> Tuple[float, str, dict]:
    """Look up tweet by tweet_id and return mismatch score, text, and engagement metrics."""
    query = sql_text(
        """
        SELECT TOP 1 text, clean_text, like_count, retweet_count, reply_count, quote_count,
               followers_count, hashtag1, hashtag2, hashtag3, has_popular_entity, eng_bucket_3
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M]
        WHERE CAST(tweet_id AS VARCHAR(32)) = :tid
        """
    )
    
    try:
        with engine.connect() as conn:
            row = conn.execute(query, {"tid": str(tweet_id)}).fetchone()
        
        if not row:
            raise ValueError(f"tweet_id {tweet_id} not found.")
        
        # Try to access text fields using different methods
        try:
            # Try attribute access first
            tweet_text = str(row.clean_text if row.clean_text else row.text)
        except AttributeError:
            try:
                # Try index access
                tweet_text = str(row[1] if row[1] else row[0])
            except (IndexError, TypeError):
                # Last resort: convert to dict
                row_dict = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
                tweet_text = str(row_dict.get('clean_text', row_dict.get('text', '')))
        
        if not tweet_text:
            raise ValueError(f"tweet_id {tweet_id} has no text content.")
        
        # Get engagement metrics
        engagement_metrics = calculate_engagement_metrics(row)
        
        # Get mismatch score
        score = predict_engagement_mismatch(tweet_text, model, tokenizer)
        
        return score, tweet_text, engagement_metrics
        
    except Exception as e:
        raise ValueError(f"Database error for tweet_id {tweet_id}: {e}")

def main():
    """Main function to score tweets for engagement mismatch."""
    parser = argparse.ArgumentParser(description='Score tweet for engagement mismatch likelihood')
    parser.add_argument('tweet_id', nargs='?', help='Tweet ID to score (optional, will prompt if not provided)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    try:
        # Load model and tokenizer
        print("🔄 Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer()
        print("✅ Model loaded successfully!")
        
        # Create database connection
        print("🔄 Connecting to database...")
        engine = create_database_connection()
        print("✅ Database connected!")
        
        if args.tweet_id:
            # Use command-line argument
            tweet_id = args.tweet_id
            print(f"📱 Scoring tweet ID: {tweet_id}")
            
            try:
                score, text, metrics = score_tweet_id(tweet_id, model, tokenizer, engine)
                interpretation = get_mismatch_interpretation(score)
                
                print(f"\n🎯 Engagement Mismatch Score: {score:.3f}")
                print(f"📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
                print(f"📊 Interpretation: {interpretation}")
                print(f"🔢 Engagement Metrics:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"   {key}: {value:.4f}")
                    else:
                        print(f"   {key}: {value}")
                        
            except ValueError as e:
                print(f"❌ Error: {e}")
                
        elif args.interactive:
            # Interactive mode
            while True:
                try:
                    tweet_id = input("\n📱 Enter tweet ID (or 'quit' to exit): ").strip()
                    
                    if tweet_id.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break
                    
                    if not tweet_id:
                        print("❌ Please enter a valid tweet ID.")
                        continue
                    
                    print(f"🔄 Scoring tweet ID: {tweet_id}...")
                    
                    score, text, metrics = score_tweet_id(tweet_id, model, tokenizer, engine)
                    interpretation = get_mismatch_interpretation(score)
                    
                    print(f"\n🎯 Engagement Mismatch Score: {score:.3f}")
                    print(f"📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
                    print(f"📊 Interpretation: {interpretation}")
                    print(f"🔢 Engagement Metrics:")
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            print(f"   {key}: {value:.4f}")
                        else:
                            print(f"   {key}: {value}")
                            
                except ValueError as e:
                    print(f"❌ Error: {e}")
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except EOFError:
                    print("\n👋 Goodbye!")
                    break
        else:
            # No arguments provided, show usage
            print("📖 Usage:")
            print("  python score_engagement_mismatch_standalone.py <tweet_id>")
            print("  python score_engagement_mismatch_standalone.py --interactive")
            print("\nExample:")
            print("  python score_engagement_mismatch_standalone.py 1503813284")
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
