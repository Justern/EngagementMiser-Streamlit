#!/usr/bin/env python3
"""
Simple Score Script for Reply-Bait Detector
==========================================

Takes a tweet ID as input and returns a single 0-1 score.
Usage: python simple_score.py <tweet_id>
Output: Single line with score (0.0 to 1.0)
"""

import sys
import os
import pandas as pd
from sqlalchemy import create_engine, text as sql_text

def get_tweet_data(tweet_id: str) -> dict:
    """Get tweet data from database."""
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
        
        # Get tweet data from database
        query = sql_text("""
            SELECT TOP 1 
                t.text,
                t.author_id,
                t.in_reply_to_user_id,
                t.conversation_id,
                t.like_count,
                t.retweet_count,
                t.reply_count,
                t.quote_count
            FROM [ecs_tweets_db].[dbo].[Tweets_Sample_4M] t
            WHERE CAST(t.tweet_id AS VARCHAR(32)) = :tid
        """)
        
        with engine.connect() as conn:
            row = conn.execute(query, {"tid": str(tweet_id)}).fetchone()
        
        if not row:
            return {}
        
        return {
            'text': str(row[0]),
            'author_id': str(row[1]),
            'in_reply_to_user_id': str(row[2]) if row[2] else None,
            'conversation_id': str(row[3]) if row[3] else None,
            'like_count': int(row[4]) if row[4] else 0,
            'retweet_count': int(row[5]) if row[5] else 0,
            'reply_count': int(row[6]) if row[6] else 0,
            'quote_count': int(row[7]) if row[7] else 0
        }
        
    except Exception as e:
        return {}

def calculate_simple_reply_bait_score(tweet_data: dict) -> float:
    """Calculate a simple reply-bait score based on tweet characteristics."""
    if not tweet_data:
        return 0.0
    
    try:
        text = tweet_data['text'].lower()
        is_reply = tweet_data['in_reply_to_user_id'] is not None
        reply_count = tweet_data['reply_count']
        
        # Reply-bait indicators
        reply_bait_phrases = [
            'what do you think?', 'thoughts?', 'agree?', 'disagree?',
            'your opinion?', 'what\'s your take?', 'how about you?',
            'anyone else?', 'am i right?', 'am i wrong?', 'thoughts?',
            'agree or disagree?', 'what say you?', 'your thoughts?',
            'anyone?', 'thoughts on this?', 'what do you think?',
            'agree?', 'disagree?', 'thoughts?', 'anyone else?'
        ]
        
        # Count reply-bait phrases
        reply_bait_count = sum(1 for phrase in reply_bait_phrases if phrase in text)
        
        # Question marks (indicate seeking engagement)
        question_marks = text.count('?')
        
        # Calculate score
        phrase_score = min(reply_bait_count / 3, 1.0)  # Normalize to 0-1
        question_score = min(question_marks / 5, 1.0)  # Normalize to 0-1
        
        # Combine scores
        final_score = (phrase_score * 0.6) + (question_score * 0.4)
        
        return max(0.0, min(1.0, final_score))
        
    except Exception as e:
        return 0.0

def main():
    """Main function to get reply-bait score for a tweet."""
    if len(sys.argv) != 2:
        print("Usage: python simple_score.py <tweet_id>")
        sys.exit(1)
    
    tweet_id = sys.argv[1].strip()
    
    try:
        # Get tweet data from database
        tweet_data = get_tweet_data(tweet_id)
        
        if not tweet_data:
            print("0.0")
            return
        
        # Calculate simple reply-bait score
        score = calculate_simple_reply_bait_score(tweet_data)
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, float(score)))
        
        # Output the score
        print(f"{score:.3f}")
        
    except Exception as e:
        # On any error, return 0.0
        print("0.0")

if __name__ == "__main__":
    main()
