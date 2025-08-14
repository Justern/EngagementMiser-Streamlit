#!/usr/bin/env python3
"""
Simple Usage Example for Reply-Bait Detector
Input: Tweet ID
Output: Single score from 0-1
"""

from reply_bait_detector import ReplyBaitDetector
from data_connector import TwitterDataConnector

def analyze_single_tweet(tweet_id: str) -> float:
    """
    Analyze a single tweet and return reply-bait score.
    
    Args:
        tweet_id: The tweet ID to analyze
        
    Returns:
        float: Reply-bait score from 0-1
    """
    try:
        # Initialize detector and connector
        detector = ReplyBaitDetector()
        connector = TwitterDataConnector('localhost', 'EngagementMiser')
        
        # Test connection
        if not connector.test_connection():
            print("❌ Database connection failed")
            return 0.0
        
        print(f"✅ Analyzing tweet: {tweet_id}")
        
        # Get the score
        score = detector.analyze_tweet(tweet_id, connector)
        
        print(f"📊 Reply-Bait Score: {score:.3f}")
        
        # Interpret the score
        if score >= 0.8:
            interpretation = "HIGH - Strong evidence of reply-baiting"
        elif score >= 0.6:
            interpretation = "MEDIUM - Moderate evidence of reply-baiting"
        elif score >= 0.4:
            interpretation = "LOW - Some evidence of reply-baiting"
        else:
            interpretation = "MINIMAL - No significant reply-baiting detected"
        
        print(f"🔍 Interpretation: {interpretation}")
        
        return score
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 0.0

def main():
    """Main function for simple tweet analysis."""
    print("REPLY-BAIT DETECTOR - SIMPLE TWEET ANALYSIS")
    print("=" * 50)
    print("Input: Tweet ID")
    print("Output: Single score from 0-1")
    print("=" * 50)
    
    # Example usage
    tweet_id = input("Enter tweet ID to analyze: ").strip()
    
    if tweet_id:
        score = analyze_single_tweet(tweet_id)
        print(f"\n🎯 Final Result: {score:.3f}")
    else:
        print("No tweet ID provided.")

if __name__ == "__main__":
    main()
