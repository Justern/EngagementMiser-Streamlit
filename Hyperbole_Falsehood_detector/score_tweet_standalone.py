#!/usr/bin/env python3
"""
Simplified Tweet Scoring Script
==============================

This script loads a pre-trained Hyperbole/Falsehood detection model and scores
individual tweets by their tweet_id.

Usage:
    python score_tweet_standalone.py
    # Then enter a tweet_id when prompted
"""

import os
import sys
import torch
import pandas as pd
from sqlalchemy import create_engine, text as sql_text
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Union, Tuple

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration - use absolute paths based on script location
MODEL_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "text_softlabel_roberta")
MAX_LEN = 220
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
        print(f"Model: {MODEL_PATH}")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
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

def predict_single_text(text: str, model, tokenizer) -> float:
    """Predict score for a single text input."""
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

def score_tweet_id(tweet_id: Union[str, int], model, tokenizer, engine) -> Tuple[float, str]:
    """Look up tweet text by tweet_id and return score and text."""
    query = sql_text(
        """
        SELECT TOP 1 text
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M]
        WHERE CAST(tweet_id AS VARCHAR(32)) = :tid
        """
    )
    
    try:
        with engine.connect() as conn:
            row = conn.execute(query, {"tid": str(tweet_id)}).fetchone()
        
        if not row or not row[0]:
            raise ValueError(f"tweet_id {tweet_id} not found or has no text.")
        
        tweet_text = str(row[0])
        score = predict_single_text(tweet_text, model, tokenizer)
        return score, tweet_text
        
    except Exception as e:
        raise ValueError(f"Database error for tweet_id {tweet_id}: {e}")

def main():
    """Main function to run the scoring interface."""
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Create database connection
    engine = create_database_connection()
    
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
            
            # Display results - simplified output
            print(f"Score: {score:.4f}")
            
        except ValueError as e:
            print(f"❌ Error: {e}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
