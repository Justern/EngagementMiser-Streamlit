#!/usr/bin/env python3
"""
Database connector for Generic Comment Detector
Connects to EngagementMiser database to retrieve tweet data by ID
"""

import pyodbc
import pandas as pd
from typing import Dict, Optional, List
import os
from database_config import get_connection_string, get_test_connection_string

class TwitterDataConnector:
    """
    Connects to Twitter database to retrieve tweet data for analysis.
    """
    
    def __init__(self, connection_string: str = None):
        """
        Initialize the database connector.
        
        Args:
            connection_string: Database connection string (optional)
        """
        if connection_string:
            self.connection_string = connection_string
        else:
            # Use configuration file for connection string
            self.connection_string = get_connection_string()
    
    def _get_connection(self):
        """Get database connection."""
        try:
            return pyodbc.connect(self.connection_string)
        except Exception as e:
            raise Exception(f"Database connection failed: {str(e)}")
    
    def get_tweet_data(self, tweet_id: str) -> Optional[Dict]:
        """
        Retrieve tweet data by ID for content quality analysis.
        
        Args:
            tweet_id: Tweet ID to retrieve
            
        Returns:
            Dict: Tweet data with text and metadata
        """
        query = """
        SELECT TOP 1
            t.tweet_id,
            t.author_id,
            t.text,
            t.clean_text,
            t.created_at,
            t.reply_count,
            t.like_count,
            t.retweet_count,
            t.quote_count,
            t.lang,
            t.possibly_sensitive,
            t.source,
            t.reply_settings,
            t.followers_count,
            t.total_engagements,
            t.engagement_rate,
            t.hashtag1,
            t.hashtag2,
            t.hashtag3,
            t.mention_screen_name1,
            t.mention_screen_name2,
            t.mention_screen_name3,
            t.url1,
            t.url2,
            t.url3,
            -- User profile data
            u.username,
            u.name,
            u.description,
            u.location,
            u.profile_image_url,
            u.verified,
            u.followers_count as user_followers_count,
            u.following_count,
            u.tweet_count,
            u.created_at as user_created_at
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
        LEFT JOIN [EngagementMiser].[dbo].[TwitterUsers] u ON t.author_id = u.id
        WHERE t.tweet_id = ?
        """
        
        try:
            with self._get_connection() as conn:
                df = pd.read_sql(query, conn, params=[tweet_id])
                
                if df.empty:
                    return None
                
                row = df.iloc[0]
                
                # Convert to dictionary
                tweet_data = {
                    'tweet_id': str(row['tweet_id']),
                    'author_id': str(row['author_id']) if row['author_id'] else None,
                    'text': row['text'] if pd.notna(row['text']) else '',
                    'clean_text': row['clean_text'] if pd.notna(row['clean_text']) else '',
                    'created_at': row['created_at'] if pd.notna(row['created_at']) else None,
                    'reply_count': int(row['reply_count']) if pd.notna(row['reply_count']) else 0,
                    'like_count': int(row['like_count']) if pd.notna(row['like_count']) else 0,
                    'retweet_count': int(row['retweet_count']) if pd.notna(row['retweet_count']) else 0,
                    'quote_count': int(row['quote_count']) if pd.notna(row['quote_count']) else 0,
                    'lang': row['lang'] if pd.notna(row['lang']) else None,
                    'possibly_sensitive': bool(row['possibly_sensitive']) if pd.notna(row['possibly_sensitive']) else False,
                    'source': row['source'] if pd.notna(row['source']) else None,
                    'reply_settings': row['reply_settings'] if pd.notna(row['reply_settings']) else None,
                    'followers_count': int(row['followers_count']) if pd.notna(row['followers_count']) else 0,
                    'total_engagements': int(row['total_engagements']) if pd.notna(row['total_engagements']) else 0,
                    'engagement_rate': float(row['engagement_rate']) if pd.notna(row['engagement_rate']) else 0.0,
                    'hashtags': [h for h in [row['hashtag1'], row['hashtag2'], row['hashtag3']] if pd.notna(h)],
                    'mentions': [m for m in [row['mention_screen_name1'], row['mention_screen_name2'], row['mention_screen_name3']] if pd.notna(m)],
                    'urls': [u for u in [row['url1'], row['url2'], row['url3']] if pd.notna(u)],
                    'user_profile': {
                        'username': row['username'] if pd.notna(row['username']) else None,
                        'name': row['name'] if pd.notna(row['name']) else None,
                        'description': row['description'] if pd.notna(row['description']) else None,
                        'location': row['location'] if pd.notna(row['location']) else None,
                        'profile_image_url': row['profile_image_url'] if pd.notna(row['profile_image_url']) else None,
                        'verified': bool(row['verified']) if pd.notna(row['verified']) else False,
                        'followers_count': int(row['user_followers_count']) if pd.notna(row['user_followers_count']) else 0,
                        'following_count': int(row['following_count']) if pd.notna(row['following_count']) else 0,
                        'tweet_count': int(row['tweet_count']) if pd.notna(row['tweet_count']) else 0,
                        'created_at': row['user_created_at'] if pd.notna(row['user_created_at']) else None
                    }
                }
                
                return tweet_data
                
        except Exception as e:
            raise Exception(f"Error retrieving tweet data: {str(e)}")
    
    def get_sample_tweets_for_analysis(self, limit: int = 10) -> List[Dict]:
        """
        Get sample tweets for testing and analysis.
        
        Args:
            limit: Number of tweets to retrieve
            
        Returns:
            List: Sample tweet data
        """
        query = f"""
        SELECT TOP {limit}
            t.tweet_id,
            t.text,
            t.clean_text,
            t.author_id,
            t.source,
            t.created_at,
            t.reply_count,
            t.like_count,
            t.retweet_count,
            t.quote_count,
            t.followers_count,
            t.total_engagements,
            t.engagement_rate
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
        WHERE t.text IS NOT NULL 
        AND LEN(t.text) > 10
        ORDER BY NEWID()
        """
        
        try:
            with self._get_connection() as conn:
                df = pd.read_sql(query, conn)
                
                tweets = []
                for _, row in df.iterrows():
                    tweet_data = {
                        'tweet_id': str(row['tweet_id']),
                        'text': row['text'],
                        'clean_text': row['clean_text'] if pd.notna(row['clean_text']) else row['text'],
                        'author_id': str(row['author_id']) if pd.notna(row['author_id']) else None,
                        'source': row['source'] if pd.notna(row['source']) else None,
                        'created_at': row['created_at'] if pd.notna(row['created_at']) else None,
                        'reply_count': int(row['reply_count']) if pd.notna(row['reply_count']) else 0,
                        'like_count': int(row['like_count']) if pd.notna(row['like_count']) else 0,
                        'retweet_count': int(row['retweet_count']) if pd.notna(row['retweet_count']) else 0,
                        'quote_count': int(row['quote_count']) if pd.notna(row['quote_count']) else 0,
                        'followers_count': int(row['followers_count']) if pd.notna(row['followers_count']) else 0,
                        'total_engagements': int(row['total_engagements']) if pd.notna(row['total_engagements']) else 0,
                        'engagement_rate': float(row['engagement_rate']) if pd.notna(row['engagement_rate']) else 0.0
                    }
                    tweets.append(tweet_data)
                
                return tweets
                
        except Exception as e:
            raise Exception(f"Error retrieving sample tweets: {str(e)}")
    
    def get_user_tweets_for_analysis(self, author_id: str, limit: int = 20) -> List[Dict]:
        """
        Get tweets from a specific user for analysis.
        
        Args:
            author_id: User ID to retrieve tweets from
            limit: Number of tweets to retrieve
            
        Returns:
            List: User's tweet data
        """
        query = f"""
        SELECT TOP {limit}
            t.tweet_id,
            t.text,
            t.clean_text,
            t.created_at,
            t.reply_count,
            t.like_count,
            t.retweet_count,
            t.quote_count,
            t.source,
            t.followers_count,
            t.total_engagements,
            t.engagement_rate
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
        WHERE t.author_id = ?
        AND t.text IS NOT NULL
        ORDER BY t.created_at DESC
        """
        
        try:
            with self._get_connection() as conn:
                df = pd.read_sql(query, conn, params=[author_id])
                
                tweets = []
                for _, row in df.iterrows():
                    tweet_data = {
                        'tweet_id': str(row['tweet_id']),
                        'text': row['text'],
                        'clean_text': row['clean_text'] if pd.notna(row['clean_text']) else row['text'],
                        'created_at': row['created_at'] if pd.notna(row['created_at']) else None,
                        'reply_count': int(row['reply_count']) if pd.notna(row['reply_count']) else 0,
                        'like_count': int(row['like_count']) if pd.notna(row['like_count']) else 0,
                        'retweet_count': int(row['retweet_count']) if pd.notna(row['retweet_count']) else 0,
                        'quote_count': int(row['quote_count']) if pd.notna(row['quote_count']) else 0,
                        'source': row['source'] if pd.notna(row['source']) else None,
                        'followers_count': int(row['followers_count']) if pd.notna(row['followers_count']) else 0,
                        'total_engagements': int(row['total_engagements']) if pd.notna(row['total_engagements']) else 0,
                        'engagement_rate': float(row['engagement_rate']) if pd.notna(row['engagement_rate']) else 0.0
                    }
                    tweets.append(tweet_data)
                
                return tweets
                
        except Exception as e:
            raise Exception(f"Error retrieving user tweets: {str(e)}")
    
    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return result[0] == 1
        except Exception as e:
            print(f"Connection test failed: {str(e)}")
            return False


def main():
    """Test the database connector."""
    print("Testing Twitter Database Connector...")
    
    # Initialize connector
    connector = TwitterDataConnector()
    
    # Test connection
    if connector.test_connection():
        print("✅ Database connection successful!")
        
        # Test sample tweet retrieval
        try:
            sample_tweets = connector.get_sample_tweets_for_analysis(3)
            print(f"\nRetrieved {len(sample_tweets)} sample tweets:")
            
            for i, tweet in enumerate(sample_tweets, 1):
                print(f"\nTweet {i}:")
                print(f"  ID: {tweet['tweet_id']}")
                print(f"  Text: {tweet['text'][:100]}...")
                print(f"  Author: {tweet['author_id']}")
                print(f"  Source: {tweet['source']}")
                print(f"  Engagement: {tweet['total_engagements']}")
        
        except Exception as e:
            print(f"❌ Error retrieving sample tweets: {str(e)}")
    
    else:
        print("❌ Database connection failed!")
        print("\nPlease check your connection string and ensure:")
        print("1. SQL Server is running")
        print("2. ODBC Driver 17 is installed")
        print("3. Database 'EngagementMiser' exists")
        print("4. Your connection string is correct")


if __name__ == "__main__":
    main()
