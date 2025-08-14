#!/usr/bin/env python3
"""
Simple Score Script for Authority Signal Manipulation Detector
==========================================================

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
                t.like_count,
                t.retweet_count,
                t.reply_count,
                t.quote_count,
                u.followers_count,
                u.following_count,
                u.verified,
                u.description
            FROM [ecs_tweets_db].[dbo].[Tweets_Sample_4M] t
            JOIN [ecs_tweets_db].[dbo].[TwitterUsers] u ON t.author_id = u.id
            WHERE CAST(t.tweet_id AS VARCHAR(32)) = :tid
        """)
        
        with engine.connect() as conn:
            row = conn.execute(query, {"tid": str(tweet_id)}).fetchone()
        
        if not row:
            return {}
        
        return {
            'text': str(row[0]),
            'author_id': str(row[1]),
            'like_count': int(row[2]) if row[2] else 0,
            'retweet_count': int(row[3]) if row[3] else 0,
            'reply_count': int(row[4]) if row[4] else 0,
            'quote_count': int(row[5]) if row[5] else 0,
            'followers_count': int(row[6]) if row[6] else 0,
            'following_count': int(row[7]) if row[7] else 0,
            'verified': bool(row[8]) if row[8] else False,
            'description': str(row[9]) if row[9] else ''
        }
        
    except Exception as e:
        return {}

def calculate_simple_authority_score(tweet_data: dict) -> float:
    """Calculate a simple authority signal manipulation score."""
    if not tweet_data:
        return 0.0
    
    try:
        text = tweet_data['text'].lower()
        followers = tweet_data['followers_count']
        following = tweet_data['following_count']
        verified = tweet_data['verified']
        
        # Authority manipulation indicators
        authority_phrases = [
            'expert', 'professional', 'doctor', 'scientist', 'researcher',
            'study shows', 'research proves', 'experts agree', 'authority',
            'scientifically proven', 'clinically tested', 'doctor recommended',
            'according to science', 'research indicates', 'studies confirm',
            'medical evidence', 'scientific evidence', 'clinical evidence',
            'expert opinion', 'professional opinion', 'authority figure'
        ]
        
        # Count authority phrases
        authority_count = sum(1 for phrase in authority_phrases if phrase in text)
        
        # Profile mismatch indicators
        profile_mismatch = 0
        
        # High authority language but low follower count
        if authority_count > 0 and followers < 1000:
            profile_mismatch += 0.3
        
        # High authority language but not verified
        if authority_count > 0 and not verified:
            profile_mismatch += 0.2
        
        # Very high following to follower ratio (suspicious)
        if following > 0 and followers > 0:
            ratio = following / followers
            if ratio > 10:  # Following 10x more than followers
                profile_mismatch += 0.2
        
        # Calculate score
        authority_ratio = min(authority_count / 5, 1.0)  # Normalize to 0-1
        final_score = (authority_ratio * 0.6) + (profile_mismatch * 0.4)
        
        return max(0.0, min(1.0, final_score))
        
    except Exception as e:
        return 0.0

def main():
    """Main function to get authority signal manipulation score for a tweet."""
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
        
        # Calculate simple authority score
        score = calculate_simple_authority_score(tweet_data)
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, float(score)))
        
        # Output the score
        print(f"{score:.3f}")
        
    except Exception as e:
        # On any error, return 0.0
        print("0.0")

if __name__ == "__main__":
    main()
