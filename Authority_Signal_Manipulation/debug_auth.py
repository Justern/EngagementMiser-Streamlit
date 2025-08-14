#!/usr/bin/env python3
"""
Debug script to test Authority Signal Manipulation
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
                t.like_count,
                t.retweet_count,
                t.reply_count,
                t.quote_count,
                u.followers_count,
                u.following_count,
                u.verified,
                u.description
            FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
            JOIN [EngagementMiser].[dbo].[TwitterUsers] u ON t.author_id = u.id
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
            'like_count': int(row[2]) if row[2] else 0,
            'retweet_count': int(row[3]) if row[3] else 0,
            'reply_count': int(row[4]) if row[4] else 0,
            'quote_count': int(row[5]) if row[5] else 0,
            'followers_count': int(row[6]) if row[6] else 0,
            'following_count': int(row[7]) if row[7] else 0,
            'verified': bool(row[8]) if row[8] else False,
            'description': str(row[9]) if row[9] else ''
        }
        
        print("Retrieved data:")
        for key, value in data.items():
            print(f"  {key}: {value}")
        
        return data
        
    except Exception as e:
        print(f"Error: {e}")
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
        
        print(f"\nAnalyzing text: {text[:100]}...")
        print(f"Followers: {followers}")
        print(f"Following: {following}")
        print(f"Verified: {verified}")
        
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
        print(f"Authority phrases found: {authority_count}")
        
        # Profile mismatch indicators
        profile_mismatch = 0
        
        # High authority language but low follower count
        if authority_count > 0 and followers < 1000:
            profile_mismatch += 0.3
            print("Profile mismatch: High authority language but low followers")
        
        # High authority language but not verified
        if authority_count > 0 and not verified:
            profile_mismatch += 0.2
            print("Profile mismatch: High authority language but not verified")
        
        # Very high following to follower ratio (suspicious)
        if following > 0 and followers > 0:
            ratio = following / followers
            if ratio > 10:  # Following 10x more than followers
                profile_mismatch += 0.2
                print(f"Profile mismatch: High following/follower ratio: {ratio:.2f}")
        
        # Calculate score
        authority_ratio = min(authority_count / 5, 1.0)  # Normalize to 0-1
        final_score = (authority_ratio * 0.6) + (profile_mismatch * 0.4)
        
        print(f"Authority ratio: {authority_ratio}")
        print(f"Profile mismatch: {profile_mismatch}")
        print(f"Final score: {final_score}")
        
        return max(0.0, min(1.0, final_score))
        
    except Exception as e:
        print(f"Error in scoring: {e}")
        return 0.0

if __name__ == "__main__":
    tweet_id = "1233064764357726209"
    print(f"Testing tweet ID: {tweet_id}")
    data = get_tweet_data(tweet_id)
    
    if data:
        print("\nCalculating score:")
        print("=" * 50)
        score = calculate_simple_authority_score(data)
        print(f"\nFinal result: {score}")
