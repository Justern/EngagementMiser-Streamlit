#!/usr/bin/env python3
"""
Simple Score Script for Content Recycling Detector
===============================================

Takes a tweet ID as input and returns a single 0-1 score.
Usage: python simple_score.py <tweet_id>
Output: Single line with score (0.0 to 1.0)
"""

import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sqlalchemy import create_engine, text as sql_text

def load_trained_model():
    """Load the trained content recycling detection model."""
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "content_recycling_detector")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device

def score_tweet_id(tweet_id: str, model, tokenizer, device, max_len: int = 128):
    """Score a tweet by ID using the trained model."""
    # Database connection
    SQL_SERVER = "ecs-sql-server-engagementmiser.database.windows.net"
    SQL_DB = "ecs_tweets_db"
    SQL_DRIVER = "ODBC Driver 18 for SQL Server"
    
    CONN_STR = (
        f"mssql+pyodbc://ecsadmin:EngagementMiser!@{SQL_SERVER}/{SQL_DB}"
        f"?driver={SQL_DRIVER.replace(' ', '+')}"
        "&UID=ecsadmin;PWD=EngagementMiser!"
        "&TrustServerCertificate=yes"
    )
    
    engine = create_engine(CONN_STR)
    
    # Get tweet text from database
    query = sql_text("""
        SELECT TOP 1 text, clean_text
        FROM [ecs_tweets_db].[dbo].[Tweets_Sample_4M]
        WHERE CAST(tweet_id AS VARCHAR(32)) = :tid
    """)
    
    with engine.connect() as conn:
        row = conn.execute(query, {"tid": str(tweet_id)}).fetchone()
    
    if not row or not row[0]:
        raise ValueError(f"tweet_id {tweet_id} not found or has no text.")
    
    # Use clean_text if available, otherwise use text
    tweet_text = str(row[1] if row[1] else row[0])
    
    # Tokenize and predict
    with torch.no_grad():
        enc = tokenizer(
            tweet_text,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        score = torch.sigmoid(logits.squeeze()).item()
        
        return float(score)

def main():
    """Main function to get content recycling score for a tweet."""
    if len(sys.argv) != 2:
        print("Usage: python simple_score.py <tweet_id>")
        sys.exit(1)
    
    tweet_id = sys.argv[1].strip()
    
    try:
        # Load the trained model
        model, tokenizer, device = load_trained_model()
        
        # Score the tweet
        score = score_tweet_id(tweet_id, model, tokenizer, device)
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, float(score)))
        
        # Output the score
        print(f"{score:.3f}")
        
    except Exception as e:
        # On any error, return 0.0
        print("0.0")

if __name__ == "__main__":
    main()
