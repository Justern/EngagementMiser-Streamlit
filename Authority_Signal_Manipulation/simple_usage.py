#!/usr/bin/env python3
"""
Simple Usage Example for Authority Signal Manipulation (ASM) Detector
Input: Tweet ID
Output: Single score from 0-1
"""

from authority_signal_detector import AuthoritySignalDetector
from data_connector import TwitterDataConnector
import pandas as pd

def analyze_single_tweet(tweet_id: str) -> float:
    """
    Analyze a single tweet and return ASM score.
    
    Args:
        tweet_id: The tweet ID to analyze
        
    Returns:
        float: ASM score from 0-1
    """
    try:
        # Initialize detector and connector
        detector = AuthoritySignalDetector(
            lexical_threshold=0.6,
            coherence_threshold=0.5,
            engagement_threshold=0.4
        )
        connector = TwitterDataConnector('localhost', 'EngagementMiser')
        
        # Test connection
        if not connector.test_connection():
            print("❌ Database connection failed")
            return 0.0
        
        # Load authority corpus if available
        try:
            authority_data = connector.get_authority_corpus()
            if not authority_data.empty:
                detector.load_authority_corpus(authority_data)
        except:
            print("⚠️  Authority corpus not available, proceeding without it")
        
        print(f"✅ Analyzing tweet: {tweet_id}")
        
        # Get the score
        score = detector.analyze_tweet_by_id(tweet_id, connector)
        
        print(f"📊 ASM Score: {score:.3f}")
        
        # Interpret the score
        if score >= 0.8:
            interpretation = "HIGH - Strong evidence of authority signal manipulation"
        elif score >= 0.6:
            interpretation = "MEDIUM - Moderate evidence of authority signal manipulation"
        elif score >= 0.4:
            interpretation = "LOW - Some evidence of authority signal manipulation"
        else:
            interpretation = "MINIMAL - No significant manipulation detected"
        
        print(f"🔍 Interpretation: {interpretation}")
        
        return score
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 0.0

def main():
    """Main function for simple tweet analysis."""
    print("AUTHORITY SIGNAL MANIPULATION (ASM) DETECTOR - SIMPLE TWEET ANALYSIS")
    print("=" * 60)
    print("Input: Tweet ID")
    print("Output: Single score from 0-1")
    print("=" * 60)
    
    # Example usage
    tweet_id = input("Enter tweet ID to analyze: ").strip()
    
    if tweet_id:
        score = analyze_single_tweet(tweet_id)
        print(f"\n🎯 Final Result: {score:.3f}")
    else:
        print("No tweet ID provided.")

if __name__ == "__main__":
    main()
