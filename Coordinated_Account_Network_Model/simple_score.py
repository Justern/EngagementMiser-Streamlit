#!/usr/bin/env python3
"""
Simple Score Script for Coordinated Account Network Model
======================================================

Takes a tweet ID as input and returns a single 0-1 score.
Usage: python simple_score.py <tweet_id>
Output: Single line with score (0.0 to 1.0)
"""

import sys
import os
import pandas as pd
from sqlalchemy import create_engine, text as sql_text

def get_tweet_network_data(tweet_id: str) -> tuple:
    """Get network data for a tweet to analyze coordination."""
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
        
        # First get the tweet author ID
        tweet_query = sql_text("""
            SELECT TOP 1 author_id, created_at
            FROM [ecs_tweets_db].[dbo].[Tweets_Sample_4M] 
            WHERE CAST(tweet_id AS VARCHAR(32)) = :tid
        """)
        
        with engine.connect() as conn:
            tweet_row = conn.execute(tweet_query, {"tid": str(tweet_id)}).fetchone()
        
        if not tweet_row:
            return pd.DataFrame(), pd.DataFrame()
        
        author_id = str(tweet_row[0])
        tweet_created = tweet_row[1]
        
        # Get user profile data
        user_query = sql_text("""
            SELECT TOP 1
                id as user_id,
                username,
                name,
                description,
                created_at,
                followers_count,
                following_count,
                tweet_count,
                listed_count,
                verified,
                location,
                url,
                profile_image_url
            FROM [ecs_tweets_db].[dbo].[TwitterUsers] 
            WHERE id = :uid
        """)
        
        with engine.connect() as conn:
            user_data = pd.read_sql(user_query, conn, params={"uid": author_id})
        
        # Get recent interactions by this user (last 30 days)
        interactions_query = sql_text("""
            SELECT TOP 100
                t.tweet_id,
                t.author_id as user_id,
                t.in_reply_to_user_id as interacted_with_id,
                t.created_at,
                t.text,
                t.like_count,
                t.retweet_count,
                t.reply_count,
                t.quote_count,
                'reply' as interaction_type
            FROM [ecs_tweets_db].[dbo].[Tweets_Sample_4M] t
            WHERE t.author_id = :uid 
                AND t.created_at >= :start_date
                AND t.in_reply_to_user_id IS NOT NULL
            
            UNION ALL
            
            SELECT TOP 100
                t.tweet_id,
                t.author_id as user_id,
                t.tweet_id as interacted_with_id,
                t.created_at,
                t.text,
                t.like_count,
                t.retweet_count,
                t.reply_count,
                t.quote_count,
                'own_tweet' as interaction_type
            FROM [ecs_tweets_db].[dbo].[Tweets_Sample_4M] t
            WHERE t.author_id = :uid 
                AND t.created_at >= :start_date
            
            ORDER BY created_at DESC
        """)
        
        start_date = tweet_created - pd.Timedelta(days=30)
        
        with engine.connect() as conn:
            interactions_data = pd.read_sql(interactions_query, conn, 
                                          params={"uid": author_id, "start_date": start_date})
        
        return user_data, interactions_data
        
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame()

def calculate_simple_coordination_score(user_data: pd.DataFrame, interactions_data: pd.DataFrame) -> float:
    """Calculate a simple coordination score based on network patterns."""
    if user_data.empty:
        return 0.0
    
    try:
        user = user_data.iloc[0]
        
        # Basic coordination indicators
        coordination_score = 0.0
        
        # 1. Suspicious follower/following ratio
        followers = user.get('followers_count', 0) or 0
        following = user.get('following_count', 0) or 0
        
        if following > 0 and followers > 0:
            ratio = following / followers
            if ratio > 10:  # Following 10x more than followers
                coordination_score += 0.3
            elif ratio > 5:  # Following 5x more than followers
                coordination_score += 0.2
            elif ratio > 2:  # Following 2x more than followers
                coordination_score += 0.1
        
        # 2. Account age vs tweet count (suspicious activity)
        tweet_count = user.get('tweet_count', 0) or 0
        if tweet_count > 10000:  # Very high tweet count
            coordination_score += 0.2
        
        # 3. Interaction patterns
        if not interactions_data.empty:
            # Check for rapid interaction patterns
            if len(interactions_data) > 50:  # High interaction volume
                coordination_score += 0.2
            
            # Check for repetitive interaction times
            if 'created_at' in interactions_data.columns:
                time_diffs = interactions_data['created_at'].diff().dropna()
                if len(time_diffs) > 10:
                    avg_time_diff = time_diffs.mean().total_seconds()
                    if avg_time_diff < 300:  # Less than 5 minutes between interactions
                        coordination_score += 0.2
        
        # 4. Profile characteristics
        if not user.get('verified', False):
            if followers > 10000:  # High followers but not verified
                coordination_score += 0.1
        
        # Normalize score to 0-1 range
        return min(1.0, coordination_score)
        
    except Exception as e:
        return 0.0

def main():
    """Main function to get coordination score for a tweet."""
    if len(sys.argv) != 2:
        print("Usage: python simple_score.py <tweet_id>")
        sys.exit(1)
    
    tweet_id = sys.argv[1].strip()
    
    try:
        # Get network data from database
        user_data, interactions_data = get_tweet_network_data(tweet_id)
        
        if user_data.empty:
            print("0.0")
            return
        
        # Calculate simple coordination score
        score = calculate_simple_coordination_score(user_data, interactions_data)
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, float(score)))
        
        # Output the score
        print(f"{score:.3f}")
        
    except Exception as e:
        # On any error, return 0.0
        print("0.0")

if __name__ == "__main__":
    main()
