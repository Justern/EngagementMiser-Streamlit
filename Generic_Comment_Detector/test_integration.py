#!/usr/bin/env python3
"""
Test the full integration: Database + Generic Comment Detector
"""

from generic_comment_detector import GenericCommentDetector
from data_connector import TwitterDataConnector
import os

def test_full_integration():
    """Test the complete database + model integration."""
    
    print("=" * 70)
    print("FULL INTEGRATION TEST: DATABASE + GENERIC COMMENT DETECTOR")
    print("=" * 70)
    
    # Check if trained model exists
    model_path = "generic_comment_model.joblib"
    if not os.path.exists(model_path):
        print("❌ Trained model not found!")
        print("Please run 'python train_model.py' first to train the model.")
        return
    
    print("✅ Trained model found!")
    
    # Initialize components
    try:
        detector = GenericCommentDetector(model_path)
        connector = TwitterDataConnector()
        print("✅ Components initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing components: {str(e)}")
        return
    
    # Test database connection
    print(f"\n🔌 Testing database connection...")
    if not connector.test_connection():
        print("❌ Database connection failed!")
        return
    
    print("✅ Database connection successful!")
    
    # Get sample tweets for analysis
    print(f"\n📥 Retrieving sample tweets for analysis...")
    try:
        sample_tweets = connector.get_sample_tweets_for_analysis(3)
        print(f"✅ Retrieved {len(sample_tweets)} sample tweets")
    except Exception as e:
        print(f"❌ Error retrieving sample tweets: {str(e)}")
        return
    
    # Analyze each tweet
    print(f"\n🔍 Analyzing tweets with Generic Comment Detector...")
    print("=" * 70)
    
    for i, tweet in enumerate(sample_tweets, 1):
        print(f"\nTweet {i}:")
        print(f"  ID: {tweet['tweet_id']}")
        print(f"  Text: {tweet['text'][:100]}...")
        print(f"  Author: {tweet['author_id']}")
        print(f"  Source: {tweet['source']}")
        print(f"  Engagement: {tweet['total_engagements']}")
        
        try:
            # Analyze the tweet
            results = detector.analyze_text(tweet['text'])
            
            print(f"\n📊 Analysis Results:")
            print(f"  Quality Score: {results['content_quality_score']:.3f}")
            print(f"  Quality Level: {results['quality_level']}")
            print(f"  Score Source: {results.get('score_source', 'unknown')}")
            
            # Show detected patterns
            patterns = results['pattern_analysis']
            if patterns:
                print(f"  Patterns Detected:")
                for pattern_type, pattern_data in patterns.items():
                    print(f"    {pattern_type}: {pattern_data['count']} indicators")
            
            # Show key features
            features = results['features']
            print(f"  Key Features:")
            print(f"    Lexical Diversity: {features.get('lexical_diversity', 0):.3f}")
            print(f"    Generic Phrase Density: {features.get('generic_phrase_density', 0):.3f}")
            print(f"    High-Quality Indicators: {features.get('high_quality_indicator_count', 0)}")
            
        except Exception as e:
            print(f"  ❌ Analysis error: {str(e)}")
        
        print("-" * 70)
    
    # Test specific tweet retrieval and analysis
    print(f"\n🎯 Testing specific tweet retrieval and analysis...")
    if sample_tweets:
        test_tweet_id = sample_tweets[0]['tweet_id']
        print(f"Retrieving tweet ID: {test_tweet_id}")
        
        try:
            tweet_data = connector.get_tweet_data(test_tweet_id)
            if tweet_data:
                print(f"✅ Tweet retrieved successfully!")
                print(f"  Text: {tweet_data['text'][:100]}...")
                print(f"  Author: {tweet_data['user_profile']['username'] or tweet_data['author_id']}")
                print(f"  Created: {tweet_data['created_at']}")
                
                # Analyze the retrieved tweet
                results = detector.analyze_text(tweet_data['text'])
                print(f"\n📊 Analysis Results:")
                print(f"  Quality Score: {results['content_quality_score']:.3f}")
                print(f"  Quality Level: {results['quality_level']}")
                print(f"  Score Source: {results.get('score_source', 'unknown')}")
                
            else:
                print(f"❌ Failed to retrieve specific tweet")
        
        except Exception as e:
            print(f"❌ Error during specific tweet analysis: {str(e)}")
    
    print(f"\n🎉 Integration test completed!")
    print(f"The Generic Comment Detector is now fully integrated with your database!")
    print(f"You can analyze any tweet by ID using the simple_usage.py script.")

if __name__ == "__main__":
    test_full_integration()
