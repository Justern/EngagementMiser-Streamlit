"""
Data Connector for Reply-Bait Detector

Handles database connections and data extraction from SQL Server
for the Reply-Bait Detector model.

Author: DS Capstone Project
Date: 2025
"""

import pandas as pd
import pyodbc
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwitterDataConnector:
    """
    Connects to SQL Server database and extracts Twitter data
    for reply-baiting analysis.
    """
    
    def __init__(self, server: str, database: str, trusted_connection: bool = True,
                 username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize the database connector.
        
        Args:
            server: SQL Server instance name
            database: Database name
            trusted_connection: Use Windows authentication
            username: SQL Server username (if not using trusted connection)
            password: SQL Server password (if not using trusted connection)
        """
        self.server = server
        self.database = database
        self.trusted_connection = trusted_connection
        self.username = username
        self.password = password
        self.connection_string = self._build_connection_string()
        
    def _build_connection_string(self) -> str:
        """Build the database connection string."""
        if self.trusted_connection:
            return f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.server};DATABASE={self.database};Trusted_Connection=yes;"
        else:
            return f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.server};DATABASE={self.database};UID={self.username};PWD={self.password};"
    
    def test_connection(self) -> bool:
        """Test the database connection."""
        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                logger.info("Database connection successful!")
                return True
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            return False
    
    def get_conversation_thread(self, conversation_id: str) -> pd.DataFrame:
        """
        Get all tweets in a specific conversation thread.
        
        Args:
            conversation_id: The conversation ID to retrieve
            
        Returns:
            DataFrame with conversation data
        """
        query = """
        SELECT 
            tweet_id,
            author_id,
            conversation_id,
            created_at,
            in_reply_to_user_id,
            text,
            reply_count,
            like_count,
            retweet_count,
            quote_count,
            lang,
            possibly_sensitive,
            source,
            reply_settings,
            hashtag1,
            hashtag2,
            hashtag3,
            mention_screen_name1,
            mention_screen_name2,
            mention_screen_name3,
            url1,
            url2,
            url3,
            followers_count,
            total_engagements,
            engagement_rate,
            clean_text
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M]
        WHERE conversation_id = ?
        ORDER BY created_at ASC
        """
        
        try:
            with pyodbc.connect(self.connection_string) as conn:
                df = pd.read_sql(query, conn, params=[conversation_id])
                logger.info(f"Retrieved {len(df)} tweets for conversation {conversation_id}")
                return df
        except Exception as e:
            logger.error(f"Error retrieving conversation {conversation_id}: {str(e)}")
            return pd.DataFrame()
    
    def get_user_conversations(self, user_id: str, limit: int = 100) -> pd.DataFrame:
        """
        Get all conversations started by a specific user.
        
        Args:
            user_id: The user ID to analyze
            limit: Maximum number of conversations to retrieve
            
        Returns:
            DataFrame with user's conversation data
        """
        query = """
        SELECT 
            t.tweet_id,
            t.author_id,
            t.conversation_id,
            t.created_at,
            t.in_reply_to_user_id,
            t.text,
            t.reply_count,
            t.like_count,
            t.retweet_count,
            t.quote_count,
            t.lang,
            t.possibly_sensitive,
            t.source,
            t.reply_settings,
            t.hashtag1,
            t.hashtag2,
            t.hashtag3,
            t.mention_screen_name1,
            t.mention_screen_name2,
            t.mention_screen_name3,
            t.url1,
            t.url2,
            t.url3,
            t.followers_count,
            t.total_engagements,
            t.engagement_rate,
            t.clean_text
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
        WHERE t.conversation_id IN (
            SELECT DISTINCT conversation_id 
            FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] 
            WHERE author_id = ? 
            AND in_reply_to_user_id IS NULL
        )
        ORDER BY t.conversation_id, t.created_at ASC
        """
        
        try:
            with pyodbc.connect(self.connection_string) as conn:
                df = pd.read_sql(query, conn, params=[user_id])
                
                # Limit to specified number of conversations
                if limit and len(df) > 0:
                    unique_conversations = df['conversation_id'].unique()[:limit]
                    df = df[df['conversation_id'].isin(unique_conversations)]
                
                logger.info(f"Retrieved {len(df)} tweets across {df['conversation_id'].nunique()} conversations for user {user_id}")
                return df
        except Exception as e:
            logger.error(f"Error retrieving conversations for user {user_id}: {str(e)}")
            return pd.DataFrame()
    
    def get_conversations_with_own_replies(self, min_own_replies: int = 2, 
                                          limit: int = 100) -> pd.DataFrame:
        """
        Get conversations where the author has replied to their own posts.
        
        Args:
            min_own_replies: Minimum number of own replies required
            limit: Maximum number of conversations to retrieve
            
        Returns:
            DataFrame with conversations containing own replies
        """
        query = """
        WITH ConversationStats AS (
            SELECT 
                conversation_id,
                author_id,
                COUNT(*) as own_reply_count
            FROM [EngagementMiser].[dbo].[Tweets_Sample_4M]
            WHERE in_reply_to_user_id IS NOT NULL
            GROUP BY conversation_id, author_id
        ),
        MainPosts AS (
            SELECT 
                t1.conversation_id,
                t1.author_id,
                t1.tweet_id as main_tweet_id,
                t1.text as main_text,
                t1.created_at as main_created_at,
                cs.own_reply_count
            FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t1
            INNER JOIN ConversationStats cs ON t1.conversation_id = cs.conversation_id 
                AND t1.author_id = cs.author_id
            WHERE t1.in_reply_to_user_id IS NULL
                AND cs.own_reply_count >= ?
        )
        SELECT 
            t.tweet_id,
            t.author_id,
            t.conversation_id,
            t.created_at,
            t.in_reply_to_user_id,
            t.text,
            t.reply_count,
            t.like_count,
            t.retweet_count,
            t.quote_count,
            t.lang,
            t.possibly_sensitive,
            t.source,
            t.reply_settings,
            t.hashtag1,
            t.hashtag2,
            t.hashtag3,
            t.mention_screen_name1,
            t.mention_screen_name2,
            t.mention_screen_name3,
            t.url1,
            t.url2,
            t.url3,
            t.followers_count,
            t.total_engagements,
            t.engagement_rate,
            t.clean_text,
            mp.own_reply_count,
            mp.main_tweet_id,
            mp.main_text,
            mp.main_created_at
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
        INNER JOIN MainPosts mp ON t.conversation_id = mp.conversation_id
        ORDER BY t.conversation_id, t.created_at ASC
        """
        
        try:
            with pyodbc.connect(self.connection_string) as conn:
                df = pd.read_sql(query, conn, params=[min_own_replies])
                
                # Limit to specified number of conversations
                if limit and len(df) > 0:
                    unique_conversations = df['conversation_id'].unique()[:limit]
                    df = df[df['conversation_id'].isin(unique_conversations)]
                
                logger.info(f"Retrieved {len(df)} tweets from {df['conversation_id'].nunique()} conversations with own replies")
                return df
        except Exception as e:
            logger.error(f"Error retrieving conversations with own replies: {str(e)}")
            return pd.DataFrame()
    
    def get_user_profile(self, user_id: str) -> Dict:
        """
        Get user profile information.
        
        Args:
            user_id: The user ID to retrieve
            
        Returns:
            Dictionary with user profile data
        """
        query = """
        SELECT 
            id,
            username,
            name,
            description,
            location,
            url,
            profile_image_url,
            created_at,
            pinned_tweet_id,
            protected,
            verified,
            withheld,
            followers_count,
            following_count,
            tweet_count,
            listed_count
        FROM [EngagementMiser].[dbo].[TwitterUsers]
        WHERE id = ?
        """
        
        try:
            with pyodbc.connect(self.connection_string) as conn:
                df = pd.read_sql(query, conn, params=[user_id])
                if not df.empty:
                    user_data = df.iloc[0].to_dict()
                    logger.info(f"Retrieved profile for user {user_id}")
                    return user_data
                else:
                    logger.warning(f"No profile found for user {user_id}")
                    return {}
        except Exception as e:
            logger.error(f"Error retrieving profile for user {user_id}: {str(e)}")
            return {}
    
    def get_sample_conversations(self, sample_size: int = 50) -> pd.DataFrame:
        """
        Get a random sample of conversations for testing and analysis.
        
        Args:
            sample_size: Number of conversations to sample
            
        Returns:
            DataFrame with sample conversation data
        """
        query = """
        WITH SampledConversations AS (
            SELECT TOP (?) conversation_id
            FROM [EngagementMiser].[dbo].[Tweets_Sample_4M]
            WHERE in_reply_to_user_id IS NULL
            ORDER BY NEWID()
        )
        SELECT
            t.tweet_id,
            t.author_id,
            t.conversation_id,
            t.created_at,
            t.in_reply_to_user_id,
            t.text,
            t.reply_count,
            t.like_count,
            t.retweet_count,
            t.quote_count,
            t.lang,
            t.possibly_sensitive,
            t.source,
            t.reply_settings,
            t.hashtag1,
            t.hashtag2,
            t.hashtag3,
            t.mention_screen_name1,
            t.mention_screen_name2,
            t.mention_screen_name3,
            t.url1,
            t.url2,
            t.url3,
            t.followers_count,
            t.total_engagements,
            t.engagement_rate,
            t.clean_text
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
        INNER JOIN SampledConversations sc ON t.conversation_id = sc.conversation_id
        ORDER BY t.conversation_id, t.created_at ASC
        """
        
        try:
            with pyodbc.connect(self.connection_string) as conn:
                df = pd.read_sql(query, conn, params=[sample_size])
                logger.info(f"Retrieved {len(df)} tweets from {df['conversation_id'].nunique()} sampled conversations")
                return df
        except Exception as e:
            logger.error(f"Error retrieving sample conversations: {str(e)}")
            return pd.DataFrame()
    
    def get_conversation_metrics(self, conversation_id: str) -> Dict:
        """
        Get high-level metrics for a conversation.
        
        Args:
            conversation_id: The conversation ID to analyze
            
        Returns:
            Dictionary with conversation metrics
        """
        query = """
        SELECT 
            conversation_id,
            COUNT(*) as total_tweets,
            COUNT(DISTINCT author_id) as unique_authors,
            SUM(reply_count) as total_replies,
            SUM(like_count) as total_likes,
            SUM(retweet_count) as total_retweets,
            SUM(quote_count) as total_quotes,
            MIN(created_at) as first_tweet_time,
            MAX(created_at) as last_tweet_time,
            AVG(CAST(followers_count AS FLOAT)) as avg_followers,
            AVG(CAST(engagement_rate AS FLOAT)) as avg_engagement_rate
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M]
        WHERE conversation_id = ?
        GROUP BY conversation_id
        """
        
        try:
            with pyodbc.connect(self.connection_string) as conn:
                df = pd.read_sql(query, conn, params=[conversation_id])
                if not df.empty:
                    metrics = df.iloc[0].to_dict()
                    logger.info(f"Retrieved metrics for conversation {conversation_id}")
                    return metrics
                else:
                    logger.warning(f"No metrics found for conversation {conversation_id}")
                    return {}
        except Exception as e:
            logger.error(f"Error retrieving metrics for conversation {conversation_id}: {str(e)}")
            return {}


def main():
    """
    Example usage of the Twitter Data Connector.
    """
    print("Twitter Data Connector for Reply-Bait Detector")
    print("=" * 50)
    
    # Example connection (modify as needed)
    connector = TwitterDataConnector(
        server="localhost",  # or your SQL Server instance
        database="EngagementMiser"
    )
    
    # Test connection
    if connector.test_connection():
        print("✓ Database connection successful!")
        
        # Example: Get sample conversations
        print("\nRetrieving sample conversations...")
        sample_data = connector.get_sample_conversations(sample_size=5)
        
        if not sample_data.empty:
            print(f"✓ Retrieved {len(sample_data)} tweets from {sample_data['conversation_id'].nunique()} conversations")
            
            # Show conversation IDs
            conv_ids = sample_data['conversation_id'].unique()
            print(f"Conversation IDs: {conv_ids[:3]}...")  # Show first 3
        else:
            print("✗ No sample data retrieved")
    else:
        print("✗ Database connection failed!")
        print("Please check your connection parameters and ensure SQL Server is running.")


if __name__ == "__main__":
    main()
