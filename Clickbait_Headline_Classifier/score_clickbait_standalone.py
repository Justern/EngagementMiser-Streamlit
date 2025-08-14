#!/usr/bin/env python3
"""
Standalone Clickbait Classifier Scorer
=====================================

This script loads a pre-trained clickbait classification model and scores
individual tweets by their tweet_id.

Usage:
    python score_clickbait_standalone.py
    # Then enter a tweet_id when prompted
"""

import os
import sys
import torch
import pandas as pd
from sqlalchemy import create_engine, text as sql_text
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Union, Tuple, List, Dict
import re

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration - use absolute paths based on script location
MODEL_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "clickbait_classifier")
MAX_LEN = 128  # Updated to match new training configuration
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
        print("💡 Make sure you have trained the model first using clickbait_classifier.py")
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

def load_popular_entities(engine) -> Dict[str, float]:
    """Load popular entities and their frequency scores."""
    try:
        query = sql_text(
            """
            SELECT entity_name, frequency_count, confidence_score
            FROM [EngagementMiser].[dbo].[Popular_Entities_Corpus]
            WHERE confidence_score > 0.5
            ORDER BY frequency_count DESC
            """
        )
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        # Create entity score mapping
        entity_scores = {}
        for _, row in df.iterrows():
            # Normalize score based on frequency and confidence
            score = (row['frequency_count'] * row['confidence_score']) / 1000
            entity_scores[row['entity_name'].lower()] = min(score, 1.0)
        
        return entity_scores
        
    except Exception as e:
        print(f"⚠️ Warning: Could not load popular entities: {e}")
        return {}

def load_colloquial_phrases(engine) -> List[str]:
    """Load colloquial phrases that might indicate clickbait."""
    try:
        query = sql_text(
            """
            SELECT phrase, meaning
            FROM [EngagementMiser].[dbo].[Colloquial_Phrasing_Corpus]
            """
        )
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        phrases = df['phrase'].tolist()
        return phrases
        
    except Exception as e:
        print(f"⚠️ Warning: Could not load colloquial phrases: {e}")
        return []

def predict_clickbait(text: str, model, tokenizer) -> float:
    """Predict clickbait score for a single text input."""
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

def get_clickbait_interpretation(score: float) -> str:
    """Get interpretation of the clickbait score."""
    if score < 0.3:
        return "Low clickbait likelihood"
    elif score < 0.5:
        return "Moderate clickbait likelihood"
    elif score < 0.7:
        return "High clickbait likelihood"
    else:
        return "Very high clickbait likelihood"

def score_tweet_id(tweet_id: Union[str, int], model, tokenizer, engine) -> Tuple[float, str]:
    """Look up tweet text by tweet_id and return clickbait score and text."""
    query = sql_text(
        """
        SELECT TOP 1 text, clean_text
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M]
        WHERE CAST(tweet_id AS VARCHAR(32)) = :tid
        """
    )
    
    try:
        with engine.connect() as conn:
            row = conn.execute(query, {"tid": str(tweet_id)}).fetchone()
        
        if not row or not row[0]:
            raise ValueError(f"tweet_id {tweet_id} not found or has no text.")
        
        # Use clean_text if available, otherwise use text
        tweet_text = str(row[1] if row[1] else row[0])
        
        # Get clickbait score
        score = predict_clickbait(tweet_text, model, tokenizer)
        
        return score, tweet_text
        
    except Exception as e:
        raise ValueError(f"Database error for tweet_id {tweet_id}: {e}")

def main():
    """Main function to run the scoring interface."""
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Create database connection
    engine = create_database_connection()
    
    print("=" * 60)
    
    while True:
        try:
            # Get tweet_id from user
            tweet_id_input = input("\nEnter tweet_id (or 'quit' to exit): ").strip()
            
            if tweet_id_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not tweet_id_input:
                print("⚠️ Please enter a valid tweet_id")
                continue
            
            # Score the tweet
            score, text = score_tweet_id(tweet_id_input, model, tokenizer, engine)
            
            # Display simplified results
            print(f"\n📊 Results for tweet_id: {tweet_id_input}")
            print(f"Tweet text: {text[:100]}{'...' if len(text) > 100 else ''}")
            print(f"Clickbait Score: {score:.4f}")
            print(f"Interpretation: {get_clickbait_interpretation(score)}")
            print("-" * 60)
            
        except ValueError as e:
            print(f"❌ Error: {e}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
