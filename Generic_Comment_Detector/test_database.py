#!/usr/bin/env python3
"""
Test database connection and functionality for Generic Comment Detector
"""

from data_connector import TwitterDataConnector
from database_config import get_connection_string, DATABASE_CONFIG

def test_database_connection():
    """Test the database connection."""
    print("=" * 60)
    print("DATABASE CONNECTION TEST")
    print("=" * 60)
    
    print(f"Configuration:")
    print(f"  Server: {DATABASE_CONFIG['server']}")
    print(f"  Database: {DATABASE_CONFIG['database']}")
    print(f"  Driver: {DATABASE_CONFIG['driver']}")
    print(f"  Authentication: {'Windows' if DATABASE_CONFIG['trusted_connection'] else 'SQL'}")
    
    print(f"\nConnection string:")
    print(get_connection_string())
    
    print(f"\nTesting connection...")
    
    try:
        # Initialize connector
        connector = TwitterDataConnector()
        
        # Test connection
        if connector.test_connection():
            print("✅ Database connection successful!")
            
            # Test sample tweet retrieval
            print(f"\nTesting sample tweet retrieval...")
            try:
                sample_tweets = connector.get_sample_tweets_for_analysis(3)
                print(f"✅ Retrieved {len(sample_tweets)} sample tweets:")
                
                for i, tweet in enumerate(sample_tweets, 1):
                    print(f"\nTweet {i}:")
                    print(f"  ID: {tweet['tweet_id']}")
                    print(f"  Text: {tweet['text'][:80]}...")
                    print(f"  Author: {tweet['author_id']}")
                    print(f"  Source: {tweet['source']}")
                    print(f"  Engagement: {tweet['total_engagements']}")
                
                # Test specific tweet retrieval
                if sample_tweets:
                    test_tweet_id = sample_tweets[0]['tweet_id']
                    print(f"\nTesting specific tweet retrieval for ID: {test_tweet_id}")
                    
                    tweet_data = connector.get_tweet_data(test_tweet_id)
                    if tweet_data:
                        print(f"✅ Tweet retrieved successfully!")
                        print(f"  Text: {tweet_data['text'][:100]}...")
                        print(f"  Author: {tweet_data['user_profile']['username'] or tweet_data['author_id']}")
                        print(f"  Created: {tweet_data['created_at']}")
                        print(f"  Engagement: {tweet_data['total_engagements']}")
                    else:
                        print(f"❌ Failed to retrieve specific tweet")
                
            except Exception as e:
                print(f"❌ Error retrieving tweets: {str(e)}")
        
        else:
            print("❌ Database connection failed!")
            print("\nTroubleshooting tips:")
            print("1. Ensure SQL Server is running")
            print("2. Check if ODBC Driver 17 is installed")
            print("3. Verify database 'EngagementMiser' exists")
            print("4. Check your connection string in database_config.py")
            print("5. Ensure Windows Authentication is enabled (if using trusted connection)")
    
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        print("\nCommon issues:")
        print("- pyodbc not installed: pip install pyodbc")
        print("- Wrong server name or instance")
        print("- Database doesn't exist")
        print("- Authentication issues")

def test_sample_tweet_analysis():
    """Test analyzing a sample tweet from the database."""
    print(f"\n" + "=" * 60)
    print("SAMPLE TWEET ANALYSIS TEST")
    print("=" * 60)
    
    try:
        connector = TwitterDataConnector()
        
        if not connector.test_connection():
            print("❌ Database connection failed. Cannot test tweet analysis.")
            return
        
        # Get a sample tweet
        sample_tweets = connector.get_sample_tweets_for_analysis(1)
        if not sample_tweets:
            print("❌ No sample tweets found.")
            return
        
        tweet = sample_tweets[0]
        print(f"Analyzing sample tweet:")
        print(f"  ID: {tweet['tweet_id']}")
        print(f"  Text: {tweet['text']}")
        print(f"  Author: {tweet['author_id']}")
        print(f"  Source: {tweet['source']}")
        
        # Now you can analyze this tweet with your model
        print(f"\nThis tweet is ready for content quality analysis!")
        print(f"Use the Generic Comment Detector to analyze it.")
        
    except Exception as e:
        print(f"❌ Error during sample analysis: {str(e)}")

if __name__ == "__main__":
    test_database_connection()
    test_sample_tweet_analysis()
