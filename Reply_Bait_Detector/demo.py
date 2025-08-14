#!/usr/bin/env python3
"""
Reply-Bait Detector Demo
Simple demonstration: Input tweet ID, output single score 0-1
"""

from reply_bait_detector import ReplyBaitDetector
from data_connector import TwitterDataConnector

def main():
    """Demonstrate the simple tweet analysis interface."""
    print("REPLY-BAIT DETECTOR DEMO")
    print("=" * 40)
    print("Input: Tweet ID")
    print("Output: Single score from 0-1")
    print("=" * 40)
    
    # Initialize the model and connector
    detector = ReplyBaitDetector()
    connector = TwitterDataConnector('localhost', 'EngagementMiser')
    
    # Test connection
    if not connector.test_connection():
        print("❌ Database connection failed")
        return
    
    print("✅ Model initialized and connected to database")
    print()
    
    # Example usage - exactly as requested
    tweet_id = "1545494512"  # Example tweet ID
    
    print(f"🔍 Analyzing tweet: {tweet_id}")
    score = detector.analyze_tweet(tweet_id, connector)
    
    print(f"📊 Reply-Bait Score: {score:.3f}")
    print()
    
    # Score interpretation
    if score >= 0.8:
        level = "HIGH"
        description = "Strong evidence of reply-baiting"
    elif score >= 0.6:
        level = "MEDIUM"
        description = "Moderate evidence of reply-baiting"
    elif score >= 0.4:
        level = "LOW"
        description = "Some evidence of reply-baiting"
    else:
        level = "MINIMAL"
        description = "No significant reply-baiting detected"
    
    print(f"🎯 Result: {level} - {description}")
    print()
    print("✅ Demo completed successfully!")

if __name__ == "__main__":
    main()
