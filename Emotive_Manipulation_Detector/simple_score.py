#!/usr/bin/env python3
"""
Simple Score Script for Emotive Manipulation Detector
==================================================

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

def calculate_simple_emotive_score(tweet_text: str) -> float:
    """Calculate a simple emotive manipulation score based on text analysis."""
    if not tweet_text:
        return 0.0
    
    try:
        text_lower = tweet_text.lower()
        
        # Emotional manipulation patterns
        emotional_patterns = {
            'urgency': [
                'urgent', 'immediate', 'now', 'hurry', 'quick', 'fast', 'limited time',
                'deadline', 'expires', 'last chance', 'don\'t miss out', 'act fast',
                'right now', 'this instant', 'asap', 'rush', 'hurry up', 'don\'t wait'
            ],
            'scarcity': [
                'limited', 'exclusive', 'rare', 'unique', 'one of a kind', 'only',
                'last one', 'while supplies last', 'limited edition', 'rare opportunity',
                'final chance', 'last opportunity', 'never again', 'once in a lifetime'
            ],
            'fear': [
                'scared', 'afraid', 'terrified', 'panic', 'danger', 'threat', 'risk',
                'warning', 'caution', 'beware', 'scary', 'frightening', 'horrifying',
                'terrifying', 'shocking', 'alarming', 'disturbing'
            ],
            'guilt': [
                'should', 'must', 'have to', 'need to', 'responsible', 'duty',
                'obligation', 'owe it to', 'let down', 'disappoint', 'fail',
                'you\'re wrong if', 'you\'ll regret', 'you\'re missing out'
            ],
            'flattery': [
                'amazing', 'incredible', 'brilliant', 'genius', 'expert', 'master',
                'professional', 'special', 'elite', 'premium', 'vip', 'exclusive',
                'outstanding', 'extraordinary', 'phenomenal', 'revolutionary'
            ]
        }
        
        # Count pattern matches
        total_patterns = 0
        for pattern_type, patterns in emotional_patterns.items():
            count = sum(1 for pattern in patterns if pattern in text_lower)
            total_patterns += count
        
        # Calculate score based on pattern density
        text_length = len(tweet_text.split())
        if text_length == 0:
            return 0.0
        
        pattern_density = total_patterns / text_length
        score = min(1.0, pattern_density * 10)  # Scale factor
        
        return score
        
    except Exception as e:
        return 0.0

def main():
    """Main function to get emotive manipulation score for a tweet."""
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
        
        # Calculate simple emotive score
        score = calculate_simple_emotive_score(tweet_text)
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, float(score)))
        
        # Output ONLY the score - no extra text
        print(f"{score:.3f}")
        
    except Exception as e:
        # On any error, return 0.0
        print("0.0")

if __name__ == "__main__":
    main()
