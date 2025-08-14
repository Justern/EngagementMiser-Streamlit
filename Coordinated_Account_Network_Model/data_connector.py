import pandas as pd
import pyodbc
import numpy as np
from datetime import datetime, timedelta

class NetworkDataConnector:
    """
    Database connector for retrieving network and interaction data
    for coordinated behavior detection.
    """
    
    def __init__(self, server, database, driver='ODBC Driver 18 for SQL Server'):
        self.server = server
        self.database = database
        self.driver = driver
        self.connection_string = f'DRIVER={{{driver}}};SERVER={server};DATABASE={database};Trusted_Connection=yes;Encrypt=no;TrustServerCertificate=yes;'
    
    def _get_connection(self):
        """Get database connection."""
        try:
            return pyodbc.connect(self.connection_string)
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return None
    
    def test_connection(self):
        """Test database connection."""
        try:
            conn = self._get_connection()
            if conn:
                conn.close()
                return True
            return False
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
    
    def get_user_network_data(self, user_id):
        """
        Get comprehensive network data for a specific user.
        
        Args:
            user_id (str): User ID to analyze
            
        Returns:
            tuple: (user_profile_data, interactions_data)
        """
        try:
            conn = self._get_connection()
            if not conn:
                return pd.DataFrame(), pd.DataFrame()
            
            # Get user profile data
            user_query = """
                SELECT TOP 1000
                    u.id as user_id,
                    u.username,
                    u.name,
                    u.description,
                    u.created_at,
                    u.followers_count,
                    u.following_count,
                    u.tweet_count,
                    u.listed_count,
                    u.verified,
                    u.location,
                    u.url,
                    u.profile_image_url
                FROM [EngagementMiser].[dbo].[TwitterUsers] u
                WHERE u.id = ?
            """
            
            user_data = pd.read_sql(user_query, conn, params=[str(user_id)])
            
            # Get user's interactions (replies, retweets, likes)
            interactions_query = """
                SELECT TOP 1000
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
                FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
                WHERE t.author_id = ? AND t.in_reply_to_user_id IS NOT NULL
                
                UNION ALL
                
                SELECT TOP 1000
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
                FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
                WHERE t.author_id = ?
                
                ORDER BY created_at DESC
            """
            
            interactions_data = pd.read_sql(interactions_query, conn, params=[str(user_id), str(user_id)])
            
            # Get users who interacted with this user's tweets
            incoming_interactions_query = """
                SELECT TOP 1000
                    t.id as tweet_id,
                    t.author_id as user_id,
                    t.id as interacted_with_id,
                    t.created_at,
                    t.text,
                    t.like_count,
                    t.retweet_count,
                    t.reply_count,
                    t.quote_count,
                    'incoming' as interaction_type
                FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
                WHERE t.in_reply_to_user_id = ? OR t.retweeted_tweet_id IN (
                    SELECT id FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] 
                    WHERE author_id = ?
                )
                ORDER BY created_at DESC
            """
            
            incoming_data = pd.read_sql(incoming_interactions_query, conn, params=[str(user_id), str(user_id)])
            
            # Combine all interactions
            all_interactions = pd.concat([interactions_data, incoming_data], ignore_index=True)
            
            conn.close()
            
            return user_data, all_interactions
            
        except Exception as e:
            print(f"Error retrieving user network data: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def get_tweet_network_data(self, tweet_id):
        """
        Get network data for a specific tweet and its interactions.
        
        Args:
            tweet_id (str): Tweet ID to analyze
            
        Returns:
            tuple: (tweet_data, user_data, interactions_data)
        """
        try:
            conn = self._get_connection()
            if not conn:
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
            # Get tweet data
            tweet_query = """
                SELECT TOP 1000
                    t.tweet_id,
                    t.author_id,
                    t.text,
                    t.created_at,
                    t.like_count,
                    t.retweet_count,
                    t.reply_count,
                    t.quote_count,
                    t.conversation_id,
                    t.in_reply_to_user_id,
                    t.retweeted_tweet_id
                FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
                WHERE t.tweet_id = ?
            """
            
            tweet_data = pd.read_sql(tweet_query, conn, params=[str(tweet_id)])
            
            if tweet_data.empty:
                conn.close()
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
            # Get conversation participants
            conversation_id = tweet_data.iloc[0]['conversation_id']
            author_id = tweet_data.iloc[0]['author_id']
            
            # Get all users in the conversation
            users_query = """
                SELECT DISTINCT TOP 1000
                    u.id as user_id,
                    u.username,
                    u.name,
                    u.description,
                    u.created_at,
                    u.followers_count,
                    u.following_count,
                    u.tweet_count,
                    u.listed_count,
                    u.verified,
                    u.location,
                    u.url,
                    u.profile_image_url
                FROM [EngagementMiser].[dbo].[TwitterUsers] u
                WHERE u.id IN (
                    SELECT DISTINCT author_id 
                    FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] 
                    WHERE conversation_id = ?
                )
            """
            
            user_data = pd.read_sql(users_query, conn, params=[str(conversation_id)])
            
            # Get all interactions in the conversation
            interactions_query = """
                SELECT TOP 1000
                    t.id as tweet_id,
                    t.author_id as user_id,
                    t.in_reply_to_user_id as interacted_with_id,
                    t.created_at,
                    t.text,
                    t.like_count,
                    t.retweet_count,
                    t.reply_count,
                    t.quote_count,
                    'reply' as interaction_type
                FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
                WHERE t.conversation_id = ? AND t.in_reply_to_user_id IS NOT NULL
                
                UNION ALL
                
                SELECT TOP 1000
                    t.id as tweet_id,
                    t.author_id as user_id,
                    t.id as interacted_with_id,
                    t.created_at,
                    t.text,
                    t.like_count,
                    t.retweet_count,
                    t.reply_count,
                    t.quote_count,
                    'own_tweet' as interaction_type
                FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
                WHERE t.conversation_id = ?
                
                ORDER BY created_at DESC
            """
            
            interactions_data = pd.read_sql(interactions_query, conn, params=[str(conversation_id), str(conversation_id)])
            
            conn.close()
            
            return tweet_data, user_data, interactions_data
            
        except Exception as e:
            print(f"Error retrieving tweet network data: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def get_sample_network_data(self, limit=100):
        """
        Get sample network data for analysis and testing.
        
        Args:
            limit (int): Number of sample records to retrieve
            
        Returns:
            tuple: (user_data, interactions_data)
        """
        try:
            conn = self._get_connection()
            if not conn:
                return pd.DataFrame(), pd.DataFrame()
            
            # Get sample users
            users_query = f"""
                SELECT TOP {limit}
                    u.id as user_id,
                    u.username,
                    u.name,
                    u.description,
                    u.created_at,
                    u.followers_count,
                    u.following_count,
                    u.tweet_count,
                    u.listed_count,
                    u.verified,
                    u.location,
                    u.url,
                    u.profile_image_url
                FROM [EngagementMiser].[dbo].[TwitterUsers] u
                WHERE u.followers_count > 0 OR u.following_count > 0
                ORDER BY NEWID()
            """
            
            user_data = pd.read_sql(users_query, conn)
            
            if user_data.empty:
                conn.close()
                return pd.DataFrame(), pd.DataFrame()
            
            # Get sample interactions for these users
            user_ids = user_data['user_id'].astype(str).tolist()
            placeholders = ','.join(['?' for _ in user_ids])
            
            interactions_query = f"""
                                SELECT TOP {limit * 2}
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
                FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
                WHERE t.author_id IN ({placeholders}) AND t.in_reply_to_user_id IS NOT NULL

                UNION ALL

                SELECT TOP {limit * 2}
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
                FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
                WHERE t.author_id IN ({placeholders})

                ORDER BY created_at DESC
            """
            
            interactions_data = pd.read_sql(interactions_query, conn, params=user_ids + user_ids)
            
            conn.close()
            
            return user_data, interactions_data
            
        except Exception as e:
            print(f"Error retrieving sample network data: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def get_conversation_network_data(self, conversation_id):
        """
        Get network data for an entire conversation thread.
        
        Args:
            conversation_id (str): Conversation ID to analyze
            
        Returns:
            tuple: (user_data, interactions_data)
        """
        try:
            conn = self._get_connection()
            if not conn:
                return pd.DataFrame(), pd.DataFrame()
            
            # Get all users in the conversation
            users_query = """
                SELECT DISTINCT TOP 1000
                    u.id as user_id,
                    u.username,
                    u.name,
                    u.description,
                    u.created_at,
                    u.followers_count,
                    u.following_count,
                    u.tweet_count,
                    u.listed_count,
                    u.verified,
                    u.location,
                    u.url,
                    u.profile_image_url
                FROM [EngagementMiser].[dbo].[TwitterUsers] u
                WHERE u.id IN (
                    SELECT DISTINCT author_id 
                    FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] 
                    WHERE conversation_id = ?
                )
            """
            
            user_data = pd.read_sql(users_query, conn, params=[str(conversation_id)])
            
            # Get all interactions in the conversation
            interactions_query = """
                SELECT TOP 1000
                    t.id as tweet_id,
                    t.author_id as user_id,
                    t.in_reply_to_user_id as interacted_with_id,
                    t.created_at,
                    t.text,
                    t.like_count,
                    t.retweet_count,
                    t.reply_count,
                    t.quote_count,
                    'reply' as interaction_type
                FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
                WHERE t.conversation_id = ? AND t.in_reply_to_user_id IS NOT NULL
                
                UNION ALL
                
                SELECT TOP 1000
                    t.id as tweet_id,
                    t.author_id as user_id,
                    t.id as interacted_with_id,
                    t.created_at,
                    t.text,
                    t.like_count,
                    t.retweet_count,
                    t.reply_count,
                    t.quote_count,
                    'own_tweet' as interaction_type
                FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
                WHERE t.conversation_id = ?
                
                ORDER BY created_at DESC
            """
            
            interactions_data = pd.read_sql(interactions_query, conn, params=[str(conversation_id), str(conversation_id)])
            
            conn.close()
            
            return user_data, interactions_data
            
        except Exception as e:
            print(f"Error retrieving conversation network data: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def get_high_coordination_suspicion_users(self, limit=50):
        """
        Get users with high suspicion of coordination based on profile patterns.
        
        Args:
            limit (int): Number of users to retrieve
            
        Returns:
            pd.DataFrame: Users with suspicious patterns
        """
        try:
            conn = self._get_connection()
            if not conn:
                return pd.DataFrame()
            
            # Query for users with suspicious patterns
            suspicious_query = f"""
                SELECT TOP {limit}
                    u.id as user_id,
                    u.username,
                    u.name,
                    u.description,
                    u.created_at,
                    u.followers_count,
                    u.following_count,
                    u.tweet_count,
                    u.listed_count,
                    u.verified,
                    u.location,
                    u.url,
                    u.profile_image_url
                FROM [EngagementMiser].[dbo].[TwitterUsers] u
                WHERE (
                    -- New accounts
                    u.created_at >= DATEADD(day, -90, GETDATE())
                    OR
                    -- Suspicious follower ratios
                    (u.followers_count > 0 AND u.following_count > 0 AND 
                     u.followers_count / NULLIF(u.following_count, 0) > 10)
                    OR
                    -- High activity
                    u.tweet_count > 1000
                    OR
                    -- Missing profile info
                    (u.description IS NULL OR u.description = '')
                )
                AND (u.followers_count > 0 OR u.following_count > 0)
                ORDER BY u.created_at ASC
            """
            
            suspicious_users = pd.read_sql(suspicious_query, conn)
            conn.close()
            
            return suspicious_users
            
        except Exception as e:
            print(f"Error retrieving suspicious users: {e}")
            return pd.DataFrame()

    def get_training_data_from_full_tweets(self, limit=10000):
        """
        Get training data from the full Tweets table for model training.
        This is slower but provides more comprehensive data for training.
        
        Args:
            limit (int): Number of records to retrieve for training
            
        Returns:
            tuple: (user_data, interactions_data)
        """
        try:
            conn = self._get_connection()
            if not conn:
                return pd.DataFrame(), pd.DataFrame()
            
            print(f"📥 Retrieving training data from full Tweets table (limit: {limit})...")
            
            # Get users with sufficient activity for training (limit to avoid too many placeholders)
            max_users = min(limit // 10, 100)  # Limit to 100 users max to avoid placeholder issues
            users_query = f"""
                SELECT TOP {max_users}
                    u.id as user_id,
                    u.username,
                    u.name,
                    u.description,
                    u.created_at,
                    u.followers_count,
                    u.following_count,
                    u.tweet_count,
                    u.listed_count,
                    u.verified,
                    u.location,
                    u.url,
                    u.profile_image_url
                FROM [EngagementMiser].[dbo].[TwitterUsers] u
                WHERE u.tweet_count > 10 AND u.followers_count > 0
                ORDER BY NEWID()
            """
            
            user_data = pd.read_sql(users_query, conn)
            
            if user_data.empty:
                conn.close()
                return pd.DataFrame(), pd.DataFrame()
            
            # Get interactions using a more efficient approach - get interactions for these users
            # but limit the total interactions to avoid overwhelming the database
            interactions_query = f"""
                SELECT TOP {limit}
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
                FROM [EngagementMiser].[dbo].[Tweets] t
                WHERE t.author_id IN (
                    SELECT TOP {max_users} id FROM [EngagementMiser].[dbo].[TwitterUsers] 
                    WHERE tweet_count > 10 AND followers_count > 0
                ) AND t.in_reply_to_user_id IS NOT NULL

                UNION ALL

                SELECT TOP {limit}
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
                FROM [EngagementMiser].[dbo].[Tweets] t
                WHERE t.author_id IN (
                    SELECT TOP {max_users} id FROM [EngagementMiser].[dbo].[TwitterUsers] 
                    WHERE tweet_count > 10 AND followers_count > 0
                )

                ORDER BY created_at DESC
            """
            
            interactions_data = pd.read_sql(interactions_query, conn)
            
            conn.close()
            
            print(f"✅ Retrieved {len(user_data)} users and {len(interactions_data)} interactions from full Tweets table")
            return user_data, interactions_data
            
        except Exception as e:
            print(f"Error retrieving training data from full Tweets table: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def get_optimized_sample_data(self, limit=100):
        """
        Get optimized sample data using smaller table for quick analysis.
        
        Args:
            limit (int): Number of sample records to retrieve
            
        Returns:
            tuple: (user_data, interactions_data)
        """
        try:
            conn = self._get_connection()
            if not conn:
                return pd.DataFrame(), pd.DataFrame()
            
            # Limit users to avoid too many placeholders
            max_users = min(limit, 50)  # Limit to 50 users max
            
            # Get sample users with optimized query
            users_query = f"""
                SELECT TOP {max_users}
                    u.id as user_id,
                    u.username,
                    u.name,
                    u.description,
                    u.created_at,
                    u.followers_count,
                    u.following_count,
                    u.tweet_count,
                    u.listed_count,
                    u.verified,
                    u.location,
                    u.url,
                    u.profile_image_url
                FROM [EngagementMiser].[dbo].[TwitterUsers] u
                WHERE u.followers_count > 0 OR u.following_count > 0
                ORDER BY NEWID()
            """
            
            user_data = pd.read_sql(users_query, conn)
            
            if user_data.empty:
                conn.close()
                return pd.DataFrame(), pd.DataFrame()
            
            # Get optimized interactions using subquery approach
            interactions_query = f"""
                SELECT TOP {limit * 2}
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
                FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
                WHERE t.author_id IN (
                    SELECT TOP {max_users} id FROM [EngagementMiser].[dbo].[TwitterUsers] 
                    WHERE followers_count > 0 OR following_count > 0
                ) AND t.in_reply_to_user_id IS NOT NULL

                UNION ALL

                SELECT TOP {limit * 2}
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
                FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
                WHERE t.author_id IN (
                    SELECT TOP {max_users} id FROM [EngagementMiser].[dbo].[TwitterUsers] 
                    WHERE followers_count > 0 OR following_count > 0
                )

                ORDER BY created_at DESC
            """
            
            interactions_data = pd.read_sql(interactions_query, conn)
            
            conn.close()
            
            return user_data, interactions_data
            
        except Exception as e:
            print(f"Error retrieving optimized sample data: {e}")
            return pd.DataFrame(), pd.DataFrame()
