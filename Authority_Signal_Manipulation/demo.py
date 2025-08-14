#!/usr/bin/env python3
"""
Authority Signal Manipulation (ASM) Detector Demo
Simple demonstration: Input tweet ID, output single score 0-1
"""

from authority_signal_detector import AuthoritySignalDetector
from data_connector import TwitterDataConnector

def main():
    print("AUTHORITY SIGNAL MANIPULATION (ASM) DETECTOR DEMO")
    print("=" * 50)
    print("Input: Tweet ID")
    print("Output: Single score from 0-1")
    print("=" * 50)
    
    # Initialize detector and connector
    detector = AuthoritySignalDetector(
        lexical_threshold=0.6,
        coherence_threshold=0.5,
        engagement_threshold=0.4
    )
    connector = TwitterDataConnector('localhost', 'EngagementMiser')
    
    if not connector.test_connection():
        print("❌ Database connection failed")
        return
    
    # Load authority corpus if available
    try:
        authority_data = connector.get_authority_corpus()
        if not authority_data.empty:
            detector.load_authority_corpus(authority_data)
            print(f"✅ Authority corpus loaded with {len(authority_data)} figures")
        else:
            print("⚠️  Authority corpus not available")
    except Exception as e:
        print(f"⚠️  Authority corpus not available: {str(e)}")
    
    print("✅ Model initialized and connected to database")
    print()
    
    # Example tweet ID (you can change this)
    tweet_id = "1545494512"  # Example tweet ID
    
    print(f"🔍 Analyzing tweet: {tweet_id}")
    score = detector.analyze_tweet_by_id(tweet_id, connector)
    
    print(f"📊 ASM Score: {score:.3f}")
    print()
    
    # Score interpretation
    if score >= 0.8:
        level = "HIGH"
        description = "Strong evidence of authority signal manipulation"
    elif score >= 0.6:
        level = "MEDIUM"
        description = "Moderate evidence of authority signal manipulation"
    elif score >= 0.4:
        level = "LOW"
        description = "Some evidence of authority signal manipulation"
    else:
        level = "MINIMAL"
        description = "No significant manipulation detected"
    
    print(f"🎯 Result: {level} - {description}")
    print()
    print("✅ Demo completed successfully!")
    print()
    print("💡 To analyze your own tweets, use:")
    print("   python simple_usage.py")

if __name__ == "__main__":
    main()
