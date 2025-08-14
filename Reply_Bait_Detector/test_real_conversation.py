#!/usr/bin/env python3
"""
Test the analyze_tweet method with real conversation data
"""

from reply_bait_detector import ReplyBaitDetector
from data_connector import TwitterDataConnector

def test_real_conversation():
    """Test with a real conversation that has self-replies."""
    
    print("TESTING WITH REAL CONVERSATION DATA")
    print("=" * 50)
    
    # Initialize
    detector = ReplyBaitDetector(similarity_threshold=0.5, sentiment_threshold=0.2)
    connector = TwitterDataConnector('localhost', 'EngagementMiser')
    
    if not connector.test_connection():
        print("❌ Database connection failed")
        return
    
    print("✅ Connected to database")
    
    # Test with a conversation that has self-replies
    tweet_id = "1406280563372302340"  # This conversation had 8 self-replies and score 0.360
    
    print(f"\n🔍 Testing tweet: {tweet_id}")
    
    # Test the simple analyze_tweet method
    score = detector.analyze_tweet(tweet_id, connector)
    
    print(f"📊 Reply-Bait Score: {score:.3f}")
    
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
    elif score >= 0.1:
        level = "MINIMAL"
        description = "Slight evidence of reply-baiting"
    else:
        level = "NONE"
        description = "No significant reply-baiting detected"
    
    print(f"🎯 Result: {level} - {description}")
    
    # Also run the full analysis to compare
    print(f"\n🔬 Running full analysis for comparison...")
    conv_data = connector.get_conversation_thread(tweet_id)
    
    if not conv_data.empty:
        full_analysis = detector.analyze_conversation_thread(conv_data)
        
        if 'error' not in full_analysis:
            print(f"📊 Full analysis score: {full_analysis['reply_bait_score']:.3f}")
            print(f"  Own replies: {full_analysis['own_replies_count']}")
            print(f"  Total replies: {full_analysis['total_replies']}")
            
            # Check if scores match
            if abs(score - full_analysis['reply_bait_score']) < 0.001:
                print("✅ Scores match perfectly!")
            else:
                print(f"⚠️  Score mismatch: simple={score:.3f}, full={full_analysis['reply_bait_score']:.3f}")
        else:
            print(f"❌ Full analysis error: {full_analysis['error']}")
    
    return score

if __name__ == "__main__":
    test_real_conversation()
