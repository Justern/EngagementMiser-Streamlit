"""
Database Connector for Rapid Engagement Spike Detector

This module handles database connections and data retrieval for engagement time series analysis.
"""

import pyodbc
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EngagementDataConnector:
    """
    Connects to the database to retrieve engagement time series data for spike detection.
    """
    
    def __init__(self):
        """Initialize the database connector."""
        self.connection_string = self._get_connection_string()
    
    def _get_connection_string(self) -> str:
        """
        Get the database connection string.
        
        Returns:
            str: Database connection string
        """
        # You can customize these connection parameters
        server = "localhost"
        database = "EngagementMiser"
        driver = "ODBC Driver 17 for SQL Server"
        trusted_connection = "yes"
        
        return (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"Trusted_Connection={trusted_connection};"
        )
    
    def test_connection(self) -> bool:
        """
        Test the database connection.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                return True
        except Exception as e:
            print(f"Database connection failed: {str(e)}")
            return False
    
    def get_tweet_engagement_timeline(self, tweet_id: str, hours_back: int = 48) -> pd.DataFrame:
        """
        Get engagement timeline data for a specific tweet.
        
        Args:
            tweet_id: ID of the tweet to analyze
            hours_back: Number of hours to look back from tweet creation
            
        Returns:
            pd.DataFrame: Time series engagement data
        """
        try:
            with pyodbc.connect(self.connection_string) as conn:
                # First get the tweet creation time
                tweet_query = """
                SELECT created_at, author_id, text
                FROM [EngagementMiser].[dbo].[Tweets_Sample_4M]
                WHERE tweet_id = ?
                """
                
                tweet_data = pd.read_sql(tweet_query, conn, params=[str(tweet_id)])
                
                if tweet_data.empty:
                    print(f"Tweet {tweet_id} not found")
                    return pd.DataFrame()
                
                tweet_created = tweet_data.iloc[0]['created_at']
                start_time = tweet_created - timedelta(hours=hours_back)
                end_time = tweet_created + timedelta(hours=hours_back)
                
                # Get engagement data over time
                engagement_query = """
                SELECT 
                    t.created_at as timestamp,
                    t.like_count as likes,
                    t.retweet_count as retweets,
                    t.reply_count as replies,
                    t.quote_count as quotes,
                    (t.like_count + t.retweet_count + t.reply_count + t.quote_count) as total_engagements,
                    t.author_id,
                    t.text
                FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
                WHERE t.created_at BETWEEN ? AND ?
                ORDER BY t.created_at
                """
                
                engagement_data = pd.read_sql(
                    engagement_query, 
                    conn, 
                    params=[start_time, end_time]
                )
                
                if engagement_data.empty:
                    print(f"No engagement data found for tweet {tweet_id}")
                    return pd.DataFrame()
                
                # Add tweet ID for reference
                engagement_data['tweet_id'] = tweet_id
                
                return engagement_data
                
        except Exception as e:
            print(f"Error retrieving engagement timeline: {str(e)}")
            return pd.DataFrame()
    
    def get_user_engagement_timeline(self, user_id: str, hours_back: int = 72) -> pd.DataFrame:
        """
        Get engagement timeline data for a specific user.
        
        Args:
            user_id: ID of the user to analyze
            hours_back: Number of hours to look back
            
        Returns:
            pd.DataFrame: Time series engagement data for user's tweets
        """
        try:
            with pyodbc.connect(self.connection_string) as conn:
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=hours_back)
                
                # Get user's tweets and engagement over time
                user_engagement_query = """
                SELECT 
                    t.created_at as timestamp,
                    t.tweet_id,
                    t.like_count as likes,
                    t.retweet_count as retweets,
                    t.reply_count as replies,
                    t.quote_count as quotes,
                    (t.like_count + t.retweet_count + t.reply_count + t.quote_count) as total_engagements,
                    t.author_id,
                    t.text
                FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
                WHERE t.author_id = ? 
                AND t.created_at BETWEEN ? AND ?
                ORDER BY t.created_at
                """
                
                user_engagement = pd.read_sql(
                    user_engagement_query,
                    conn,
                    params=[user_id, start_time, end_time]
                )
                
                if user_engagement.empty:
                    print(f"No engagement data found for user {user_id}")
                    return pd.DataFrame()
                
                return user_engagement
                
        except Exception as e:
            print(f"Error retrieving user engagement timeline: {str(e)}")
            return pd.DataFrame()
    
    def get_conversation_engagement_timeline(self, conversation_id: str, hours_back: int = 48) -> pd.DataFrame:
        """
        Get engagement timeline data for an entire conversation thread.
        
        Args:
            conversation_id: ID of the conversation to analyze
            hours_back: Number of hours to look back from conversation start
            
        Returns:
            pd.DataFrame: Time series engagement data for the conversation
        """
        try:
            with pyodbc.connect(self.connection_string) as conn:
                # Get conversation start time
                conv_query = """
                SELECT MIN(created_at) as conversation_start
                FROM [EngagementMiser].[dbo].[Tweets_Sample_4M]
                WHERE conversation_id = ?
                """
                
                conv_data = pd.read_sql(conv_query, conn, params=[str(conversation_id)])
                
                if conv_data.empty:
                    print(f"Conversation {conversation_id} not found")
                    return pd.DataFrame()
                
                conv_start = conv_data.iloc[0]['conversation_start']
                start_time = conv_start - timedelta(hours=hours_back)
                end_time = conv_start + timedelta(hours=hours_back)
                
                # Get all tweets in conversation with engagement
                conversation_query = """
                SELECT 
                    t.created_at as timestamp,
                    t.tweet_id,
                    t.conversation_id,
                    t.like_count as likes,
                    t.retweet_count as retweets,
                    t.reply_count as replies,
                    t.quote_count as quotes,
                    (t.like_count + t.retweet_count + t.reply_count + t.quote_count) as total_engagements,
                    t.author_id,
                    t.text
                FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
                WHERE t.conversation_id = ?
                AND t.created_at BETWEEN ? AND ?
                ORDER BY t.created_at
                """
                
                conversation_data = pd.read_sql(
                    conversation_query,
                    conn,
                    params=[conversation_id, start_time, end_time]
                )
                
                if conversation_data.empty:
                    print(f"No conversation data found for {conversation_id}")
                    return pd.DataFrame()
                
                return conversation_data
                
        except Exception as e:
            print(f"Error retrieving conversation engagement timeline: {str(e)}")
            return pd.DataFrame()
    
    def get_sample_engagement_data(self, limit: int = 1000) -> pd.DataFrame:
        """
        Get sample engagement data for testing and analysis.
        
        Args:
            limit: Maximum number of records to retrieve
            
        Returns:
            pd.DataFrame: Sample engagement data
        """
        try:
            with pyodbc.connect(self.connection_string) as conn:
                sample_query = """
                SELECT TOP (?) 
                    t.created_at as timestamp,
                    t.tweet_id,
                    t.conversation_id,
                    t.like_count as likes,
                    t.retweet_count as retweets,
                    t.reply_count as replies,
                    t.quote_count as quotes,
                    (t.like_count + t.retweet_count + t.reply_count + t.quote_count) as total_engagements,
                    t.author_id,
                    t.text
                FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
                ORDER BY t.created_at DESC
                """
                
                sample_data = pd.read_sql(sample_query, conn, params=[limit])
                
                if sample_data.empty:
                    print("No sample engagement data found")
                    return pd.DataFrame()
                
                return sample_data
                
        except Exception as e:
            print(f"Error retrieving sample engagement data: {str(e)}")
            return pd.DataFrame()
    
    def get_high_engagement_tweets(self, min_engagement: int = 1000, hours_back: int = 168) -> pd.DataFrame:
        """
        Get tweets with high engagement for spike analysis.
        
        Args:
            min_engagement: Minimum total engagement threshold
            hours_back: Number of hours to look back
            
        Returns:
            pd.DataFrame: High engagement tweets
        """
        try:
            with pyodbc.connect(self.connection_string) as conn:
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=hours_back)
                
                high_engagement_query = """
                SELECT 
                    t.created_at as timestamp,
                    t.tweet_id,
                    t.conversation_id,
                    t.like_count as likes,
                    t.retweet_count as retweets,
                    t.reply_count as replies,
                    t.quote_count as quotes,
                    (t.like_count + t.retweet_count + t.reply_count + t.quote_count) as total_engagements,
                    t.author_id,
                    t.text
                FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
                WHERE (t.like_count + t.retweet_count + t.reply_count + t.quote_count) >= ?
                AND t.created_at BETWEEN ? AND ?
                ORDER BY (t.like_count + t.retweet_count + t.reply_count + t.quote_count) DESC
                """
                
                high_engagement_data = pd.read_sql(
                    high_engagement_query,
                    conn,
                    params=[min_engagement, start_time, end_time]
                )
                
                if high_engagement_data.empty:
                    print(f"No high engagement tweets found (threshold: {min_engagement})")
                    return pd.DataFrame()
                
                return high_engagement_data
                
        except Exception as e:
            print(f"Error retrieving high engagement tweets: {str(e)}")
            return pd.DataFrame()


def main():
    """
    Test the database connector functionality.
    """
    print("Testing Engagement Data Connector...")
    
    connector = EngagementDataConnector()
    
    # Test connection
    if connector.test_connection():
        print("✅ Database connection successful!")
        
        # Test sample data retrieval
        print("\n📥 Retrieving sample engagement data...")
        sample_data = connector.get_sample_engagement_data(limit=100)
        
        if not sample_data.empty:
            print(f"✅ Retrieved {len(sample_data)} sample records")
            print(f"📊 Data range: {sample_data['timestamp'].min()} to {sample_data['timestamp'].max()}")
            print(f"📈 Engagement range: {sample_data['total_engagements'].min()} to {sample_data['total_engagements'].max()}")
            
            # Show sample records
            print("\n📋 Sample records:")
            print(sample_data[['timestamp', 'tweet_id', 'total_engagements', 'text']].head())
        else:
            print("❌ No sample data retrieved")
    else:
        print("❌ Database connection failed!")


if __name__ == "__main__":
    main()
