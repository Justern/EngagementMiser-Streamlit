#!/usr/bin/env python3
"""
Simple Score Script for Rapid Engagement Spike Detector
====================================================

Takes a tweet ID as input and returns a single 0-1 score.
Usage: python simple_score.py <tweet_id>
Output: Single line with score (0.0 to 1.0)
"""

import sys
import os
import pandas as pd
from sqlalchemy import create_engine, text as sql_text

def get_tweet_engagement_data(tweet_id: str) -> pd.DataFrame:
    """Get basic engagement data for a tweet."""
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
        
        # Get tweet engagement data from database
        query = sql_text("""
            SELECT TOP 1 
                tweet_id,
                created_at,
                like_count,
                retweet_count,
                reply_count,
                quote_count,
                (like_count + retweet_count + reply_count + quote_count) as total_engagements
            FROM [ecs_tweets_db].[dbo].[Tweets_Sample_4M] 
            WHERE CAST(tweet_id AS VARCHAR(32)) = :tid
        """)
        
        with engine.connect() as conn:
            row = conn.execute(query, {"tid": str(tweet_id)}).fetchone()
        
        if not row:
            return pd.DataFrame()
        
        # Create a simple DataFrame with the data
        data = pd.DataFrame([{
            'timestamp': row[1],
            'total_engagements': row[6],
            'likes': row[2],
            'retweets': row[3],
            'replies': row[4],
            'quotes': row[5]
        }])
        
        return data
        
    except Exception as e:
        return pd.DataFrame()

def calculate_simple_spike_score(engagement_data: pd.DataFrame) -> float:
    """Calculate a simple spike score based on engagement data."""
    if engagement_data.empty:
        return 0.0
    
    try:
        total_engagements = engagement_data.iloc[0]['total_engagements']
        
        # Simple heuristic: higher engagement = higher chance of spikes
        # Normalize to 0-1 range
        if total_engagements == 0:
            return 0.0
        elif total_engagements < 10:
            return 0.1
        elif total_engagements < 50:
            return 0.3
        elif total_engagements < 100:
            return 0.5
        elif total_engagements < 500:
            return 0.7
        else:
            return 0.9
            
    except Exception as e:
        return 0.0

def main():
    """Main function to get engagement spike score for a tweet."""
    if len(sys.argv) != 2:
        print("Usage: python simple_score.py <tweet_id>")
        sys.exit(1)
    
    tweet_id = sys.argv[1].strip()
    
    try:
        # Get engagement data from database
        engagement_data = get_tweet_engagement_data(tweet_id)
        
        if engagement_data.empty:
            print("0.0")
            return
        
        # Calculate simple spike score
        score = calculate_simple_spike_score(engagement_data)
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, float(score)))
        
        # Output the score
        print(f"{score:.3f}")
        
    except Exception as e:
        # On any error, return 0.0
        print("0.0")

if __name__ == "__main__":
    main()
