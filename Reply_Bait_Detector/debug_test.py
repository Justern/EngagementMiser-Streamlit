#!/usr/bin/env python3
"""
Debug script to test Reply-Bait Detector data retrieval
"""

import pandas as pd
from sqlalchemy import create_engine, text as sql_text

def get_tweet_data(tweet_id: str) -> dict:
    """Get tweet data from database."""
    try:
        # Database connection
        SQL_SERVER = "localhost"
        SQL_DB = "EngagementMiser"
        SQL_DRIVER = "ODBC Driver 18 for SQL Server"
        
        CONN_STR = (
            f"mssql+pyodbc://@{SQL_SERVER}/{SQL_DB}"
            f"?driver={SQL_DRIVER.replace(' ', '+')}"
            "&Trusted_Connection=yes"
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
            FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
            WHERE CAST(t.tweet_id AS VARCHAR(32)) = :tid
        """)
        
        with engine.connect() as conn:
            row = conn.execute(query, {"tid": str(tweet_id)}).fetchone()
        
        if not row:
            print("No data found!")
            return {}
        
        data = {
            'text': str(row[0]),
            'author_id': str(row[1]),
            'in_reply_to_user_id': str(row[2]) if row[2] else None,
            'conversation_id': str(row[3]) if row[3] else None,
            'like_count': int(row[4]) if row[4] else 0,
            'retweet_count': int(row[5]) if row[5] else 0,
            'reply_count': int(row[6]) if row[6] else 0,
            'quote_count': int(row[7]) if row[7] else 0
        }
        
        print("Retrieved data:")
        for key, value in data.items():
            print(f"  {key}: {value}")
        
        return data
        
    except Exception as e:
        print(f"Error: {e}")
        return {}

if __name__ == "__main__":
    tweet_id = "1233064764357726209"
    print(f"Testing tweet ID: {tweet_id}")
    data = get_tweet_data(tweet_id)
