"""
Data Connector for Authority Signal Manipulation (ASM) Model

Provides database connectivity and data retrieval methods for ASM analysis.
"""

import pandas as pd
import pyodbc
from typing import Dict, Optional, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwitterDataConnector:
    """
    Connects to Twitter database and provides data retrieval methods for ASM analysis.
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
        
        # Build connection string
        if trusted_connection:
            self.connection_string = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={server};"
                f"DATABASE={database};"
                f"Trusted_Connection=yes;"
            )
        else:
            self.connection_string = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={server};"
                f"DATABASE={database};"
                f"UID={username};"
                f"PWD={password};"
            )
    
    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
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
    
    def get_tweet_data(self, tweet_id: str) -> Optional[Dict]:
        """
        Get comprehensive tweet data for ASM analysis.
        
        Args:
            tweet_id: Tweet ID to retrieve
            
        Returns:
            Dict: Tweet data with all necessary fields for ASM analysis
        """
        query = """
        SELECT TOP 1
            t.tweet_id,
            t.author_id,
            t.text,
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
            t.clean_text,
            -- User profile data
            u.following_count,
            u.tweet_count,
            u.verified,
            u.created_at as user_created_at,
            u.description,
            u.profile_image_url,
            u.url,
            -- Additional profile fields
            u.name,
            u.username,
            u.location
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
        LEFT JOIN [EngagementMiser].[dbo].[TwitterUsers] u ON t.author_id = u.id
        WHERE t.tweet_id = ?
        """
        
        try:
            with pyodbc.connect(self.connection_string) as conn:
                df = pd.read_sql(query, conn, params=[tweet_id])
                
                if df.empty:
                    logger.warning(f"No tweet found with ID: {tweet_id}")
                    return None
                
                # Convert to dictionary
                tweet_data = df.iloc[0].to_dict()
                
                # Handle NaN values
                for key, value in tweet_data.items():
                    if pd.isna(value):
                        if key in ['verified', 'protected']:
                            tweet_data[key] = False
                        elif key in ['followers_count', 'following_count', 'tweet_count']:
                            tweet_data[key] = 0
                        else:
                            tweet_data[key] = ''
                
                # Calculate account age in days if we have user creation date
                if 'user_created_at' in tweet_data and tweet_data['user_created_at']:
                    try:
                        from datetime import datetime
                        user_created = pd.to_datetime(tweet_data['user_created_at'])
                        current_time = pd.Timestamp.now()
                        account_age_days = (current_time - user_created).days
                        tweet_data['account_age_days'] = account_age_days
                    except:
                        tweet_data['account_age_days'] = 365  # Default to 1 year
                else:
                    tweet_data['account_age_days'] = 365
                
                # Calculate description length
                if 'description' in tweet_data and tweet_data['description']:
                    tweet_data['description_length'] = len(str(tweet_data['description']))
                else:
                    tweet_data['description_length'] = 0
                
                # Handle profile image and banner image
                tweet_data['profile_image'] = bool(tweet_data.get('profile_image_url'))
                tweet_data['banner_image'] = False  # Not available in current schema
                
                # Handle title and organization (not available, use defaults)
                tweet_data['title'] = ''
                tweet_data['organization'] = ''
                
                logger.info(f"Retrieved tweet data for ID: {tweet_id}")
                return tweet_data
                
        except Exception as e:
            logger.error(f"Error retrieving tweet data: {str(e)}")
            return None
    
    def get_authority_corpus(self) -> pd.DataFrame:
        """
        Get authority figures corpus for legitimate authority signal comparison.
        
        Returns:
            DataFrame: Authority figures data
        """
        query = """
        SELECT TOP (1000)
            [id],
            [name],
            [title],
            [organization]
        FROM [EngagementMiser].[dbo].[Authority_Figures_Corpus]
        """
        
        try:
            with pyodbc.connect(self.connection_string) as conn:
                df = pd.read_sql(query, conn)
                logger.info(f"Retrieved {len(df)} authority figures from corpus")
                return df
                
        except Exception as e:
            logger.error(f"Error retrieving authority corpus: {str(e)}")
            return pd.DataFrame()
    
    def get_sample_tweets_for_asm(self, sample_size: int = 100) -> pd.DataFrame:
        """
        Get sample tweets for ASM model testing and validation.
        
        Args:
            sample_size: Number of tweets to sample
            
        Returns:
            DataFrame: Sample tweet data
        """
        query = """
        SELECT TOP (?) 
            t.tweet_id,
            t.author_id,
            t.text,
            t.created_at,
            t.reply_count,
            t.like_count,
            t.retweet_count,
            t.quote_count,
            t.followers_count,
            t.total_engagements,
            t.engagement_rate,
            t.clean_text,
            u.following_count,
            u.tweet_count,
            u.verified,
            u.created_at as user_created_at,
            u.description,
            u.profile_image_url,
            u.url,
            u.name,
            u.username,
            u.location
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
        LEFT JOIN [EngagementMiser].[dbo].[TwitterUsers] u ON t.author_id = u.id
        WHERE t.text IS NOT NULL 
        AND LEN(t.text) > 50  -- Only tweets with substantial text
        ORDER BY NEWID()
        """
        
        try:
            with pyodbc.connect(self.connection_string) as conn:
                df = pd.read_sql(query, conn, params=[sample_size])
                logger.info(f"Retrieved {len(df)} sample tweets for ASM analysis")
                return df
                
        except Exception as e:
            logger.error(f"Error retrieving sample tweets: {str(e)}")
            return pd.DataFrame()
    
    def get_user_tweets_for_asm(self, author_id: str, limit: int = 50) -> pd.DataFrame:
        """
        Get all tweets from a specific user for ASM analysis.
        
        Args:
            author_id: User ID to analyze
            limit: Maximum number of tweets to retrieve
            
        Returns:
            DataFrame: User's tweets data
        """
        query = """
        SELECT TOP (?) 
            t.tweet_id,
            t.author_id,
            t.text,
            t.created_at,
            t.reply_count,
            t.like_count,
            t.retweet_count,
            t.quote_count,
            t.followers_count,
            t.total_engagements,
            t.engagement_rate,
            t.clean_text,
            u.following_count,
            u.tweet_count,
            u.verified,
            u.created_at as user_created_at,
            u.description,
            u.profile_image_url,
            u.url,
            u.name,
            u.username,
            u.location
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
        LEFT JOIN [EngagementMiser].[dbo].[TwitterUsers] u ON t.author_id = u.id
        WHERE t.author_id = ?
        ORDER BY t.created_at DESC
        """
        
        try:
            with pyodbc.connect(self.connection_string) as conn:
                df = pd.read_sql(query, conn, params=[author_id])
                logger.info(f"Retrieved {len(df)} tweets for user {author_id}")
                return df
                
        except Exception as e:
            logger.error(f"Error retrieving user tweets: {str(e)}")
            return pd.DataFrame()


def main():
    """
    Test the data connector functionality.
    """
    print("Twitter Data Connector for ASM Model")
    print("=" * 40)
    
    # Initialize connector
    connector = TwitterDataConnector('localhost', 'EngagementMiser')
    
    # Test connection
    if connector.test_connection():
        print("✅ Database connection successful")
        
        # Test authority corpus retrieval
        authority_data = connector.get_authority_corpus()
        if not authority_data.empty:
            print(f"✅ Authority corpus loaded: {len(authority_data)} figures")
            print("Sample authority figures:")
            print(authority_data.head(3))
        else:
            print("⚠️  Authority corpus not available")
        
        # Test sample tweet retrieval
        sample_tweets = connector.get_sample_tweets_for_asm(5)
        if not sample_tweets.empty:
            print(f"✅ Sample tweets loaded: {len(sample_tweets)} tweets")
        else:
            print("⚠️  Sample tweets not available")
            
    else:
        print("❌ Database connection failed")


if __name__ == "__main__":
    main()
