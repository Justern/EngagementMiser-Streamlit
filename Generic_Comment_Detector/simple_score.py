#!/usr/bin/env python3
"""
Simple Score Script for Generic Comment Detector
==============================================

Takes a tweet ID as input and returns a single 0-1 score.
Usage: python simple_score.py <tweet_id>
Output: Single line with score (0.0 to 1.0)
"""

import sys
import os
import pandas as pd
from sqlalchemy import create_engine, text as sql_text

def get_tweet_text(tweet_id: str) -> str:
    """Get tweet text from database."""
    try:
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
            SELECT TOP 1 text 
            FROM [ecs_tweets_db].[dbo].[Tweets_Sample_4M] 
            WHERE CAST(tweet_id AS VARCHAR(32)) = :tid
        """)
        
        with engine.connect() as conn:
            row = conn.execute(query, {"tid": str(tweet_id)}).fetchone()
        
        if not row or not row[0]:
            raise ValueError(f"tweet_id {tweet_id} not found or has no text.")
        
        return str(row[0])
        
    except Exception as e:
        raise ValueError(f"Database error: {e}")

def calculate_simple_generic_score(tweet_text: str) -> float:
    """Calculate a simple generic content score based on text analysis."""
    if not tweet_text:
        return 0.0
    
    try:
        # Simple heuristic based on text characteristics
        text_lower = tweet_text.lower()
        
        # Generic phrases that indicate low-quality content
        generic_phrases = [
            'nice', 'good', 'great', 'awesome', 'amazing', 'cool', 'wow', 'omg',
            'lol', 'haha', 'thanks', 'thank you', 'congrats', 'congratulations',
            'well done', 'good job', 'keep it up', 'you got this', 'stay strong',
            'sending love', 'thoughts and prayers', 'stay safe', 'take care',
            'good luck', 'best wishes', 'have a great day', 'happy friday',
            'mood', 'same', 'relatable', 'this', 'that', 'yes', 'no', 'okay',
            'sure', 'maybe', 'idk', 'i don\'t know', 'whatever', 'fine', 'ok'
        ]
        
        # Count generic phrases
        generic_count = sum(1 for phrase in generic_phrases if phrase in text_lower)
        
        # Text length factor
        text_length = len(tweet_text.split())
        
        # Calculate score: more generic phrases and shorter text = higher generic score
        if text_length == 0:
            return 0.0
        
        generic_ratio = generic_count / text_length
        length_factor = max(0, (50 - text_length) / 50)  # Shorter text gets higher score
        
        # Combine factors
        score = (generic_ratio * 0.7) + (length_factor * 0.3)
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        return 0.0

def main():
    """Main function to get generic comment score for a tweet."""
    if len(sys.argv) != 2:
        print("Usage: python simple_score.py <tweet_id>")
        sys.exit(1)
    
    tweet_id = sys.argv[1].strip()
    
    try:
        # Get tweet text from database
        tweet_text = get_tweet_text(tweet_id)
        
        if not tweet_text:
            print("0.0")
            return
        
        # Calculate simple generic score
        score = calculate_simple_generic_score(tweet_text)
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, float(score)))
        
        # Output the score
        print(f"{score:.3f}")
        
    except Exception as e:
        # On any error, return 0.0
        print("0.0")

if __name__ == "__main__":
    main()
