#!/usr/bin/env python3
"""
Simple Usage Example for Generic Comment Detector
Input: Tweet ID from database
Output: Content quality score from 0-1
"""

from generic_comment_detector import GenericCommentDetector
from data_connector import TwitterDataConnector
import os

def analyze_tweet_by_id(tweet_id: str) -> float:
    """
    Analyze a tweet by ID using database integration.
    
    Args:
        tweet_id: Tweet ID to analyze
        
    Returns:
        float: Content quality score from 0-1
    """
    print(f"🔍 Tweet ID: {tweet_id}")
    
    try:
        # Initialize database connector
        connector = TwitterDataConnector()
        
        # Test connection first
        if not connector.test_connection():
            print("❌ Database connection failed!")
            print("Please check your connection string and ensure SQL Server is running.")
            return 0.0
        
        print("✅ Database connection successful!")
        
        # Retrieve tweet data
        print("📥 Retrieving tweet data from database...")
        tweet_data = connector.get_tweet_data(tweet_id)
        
        if not tweet_data:
            print(f"❌ Tweet with ID {tweet_id} not found in database.")
            return 0.0
        
        print("✅ Tweet data retrieved successfully!")
        print(f"📝 Text: {tweet_data['text'][:100]}...")
        print(f"👤 Author: {tweet_data['user_profile']['username'] or tweet_data['author_id']}")
        print(f"📅 Created: {tweet_data['created_at']}")
        print(f"📊 Engagement: {tweet_data['total_engagements']} total")
        
        # Analyze the tweet text
        print(f"\n🔍 Analyzing content quality...")
        
        # Try to load pre-trained model first
        model_path = "generic_comment_model.joblib"
        detector = None
        
        # Debug: Show current working directory and model path
        print(f"🔍 Current working directory: {os.getcwd()}")
        print(f"🔍 Looking for model at: {os.path.abspath(model_path)}")
        print(f"🔍 Model file exists: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            try:
                print(f"🔍 Attempting to load model from: {os.path.abspath(model_path)}")
                detector = GenericCommentDetector(model_path)
                if detector.is_trained:
                    print("✅ Loaded pre-trained ML model")
                else:
                    print("⚠️  Model file exists but not properly trained, using rule-based scoring")
                    detector = GenericCommentDetector()
            except Exception as e:
                print(f"⚠️  Error loading model: {str(e)}, using rule-based scoring")
                detector = GenericCommentDetector()
        else:
            # Try with absolute path as fallback
            script_dir = os.path.dirname(os.path.abspath(__file__))
            absolute_model_path = os.path.join(script_dir, "generic_comment_model.joblib")
            print(f"🔍 Trying absolute path: {absolute_model_path}")
            
            if os.path.exists(absolute_model_path):
                try:
                    print(f"🔍 Attempting to load model from absolute path")
                    detector = GenericCommentDetector(absolute_model_path)
                    if detector.is_trained:
                        print("✅ Loaded pre-trained ML model from absolute path")
                    else:
                        print("⚠️  Model file exists but not properly trained, using rule-based scoring")
                        detector = GenericCommentDetector()
                except Exception as e:
                    print(f"⚠️  Error loading model from absolute path: {str(e)}, using rule-based scoring")
                    detector = GenericCommentDetector()
            else:
                # Use untrained model with rule-based scoring
                detector = GenericCommentDetector()
                print("⚠️  No pre-trained model found, using rule-based scoring")
        
        # Get the score
        if detector.is_trained:
            results = detector.analyze_text(tweet_data['text'])
            score = results['generic_content_score']
            source = results.get('score_source', 'unknown')
        else:
            results = detector.analyze_text(tweet_data['text'])
            score = results['generic_content_score']
            source = 'rule_based'
        
        print(f"📊 Generic Content Score: {score:.3f}")
        print(f"🔍 Score Source: {source}")
        
        # Interpret the score (inverted: higher = more generic)
        if score >= 0.7:
            interpretation = "VERY_GENERIC - Very low-quality, repetitive content"
        elif score >= 0.5:
            interpretation = "GENERIC - Low-quality, generic content"
        elif score >= 0.3:
            interpretation = "MODERATE - Moderate quality content"
        else:
            interpretation = "HIGH_QUALITY - High-quality, substantive content"
        
        print(f"🔍 Interpretation: {interpretation}")
        
        # Show score source explanation
        if source == 'rule_based_override':
            print(f"💡 Note: Rule-based scoring used due to ML model underperformance")
        elif source == 'ml_model':
            print(f"💡 Note: ML model prediction used")
        elif source == 'rule_based':
            print(f"💡 Note: Rule-based scoring used (no trained model)")
        
        return score
        
    except Exception as e:
        print(f"❌ Error analyzing tweet by ID: {str(e)}")
        return 0.0

def main():
    """Main function for tweet ID analysis."""
    print("GENERIC COMMENT DETECTOR - TWEET ID ANALYSIS")
    print("=" * 60)
    print("Input: Tweet ID from database")
    print("Output: Generic content score from 0-1 (higher = more generic)")
    print("=" * 60)
    
    tweet_id = input("\nEnter tweet ID to analyze: ").strip()
    
    if tweet_id:
        score = analyze_tweet_by_id(tweet_id)
        print(f"\n🎯 Final Result: {score:.3f}")
    else:
        print("No tweet ID provided.")

if __name__ == "__main__":
    main()
