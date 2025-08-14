#!/usr/bin/env python3
"""
Test script with synthetic data to verify Reply-Bait Detector scoring
"""

from reply_bait_detector import ReplyBaitDetector
import pandas as pd

def test_synthetic_conversation():
    """Test with synthetic data that should produce a non-zero score."""
    
    print("TESTING WITH SYNTHETIC REPLY-BAITING DATA")
    print("=" * 50)
    
    # Create synthetic conversation with clear reply-baiting patterns
    synthetic_data = pd.DataFrame([
        {
            'tweet_id': 'main_001',
            'author_id': 'user123',
            'conversation_id': 'conv_001',
            'created_at': pd.Timestamp('2025-01-01 10:00:00'),
            'in_reply_to_user_id': None,  # Main post
            'text': 'I hate this new policy! It\'s absolutely terrible and will ruin everything!',
            'reply_count': 5,
            'like_count': 10,
            'retweet_count': 2,
            'quote_count': 1,
            'lang': 'en',
            'possibly_sensitive': False,
            'source': 'Twitter Web App',
            'reply_settings': 'everyone',
            'followers_count': 1000,
            'total_engagements': 18,
            'engagement_rate': 0.018,
            'clean_text': 'i hate this new policy its absolutely terrible and will ruin everything'
        },
        {
            'tweet_id': 'reply_001',
            'author_id': 'user123',
            'conversation_id': 'conv_001',
            'created_at': pd.Timestamp('2025-01-01 10:05:00'),
            'in_reply_to_user_id': 'user123',  # Replying to own main post
            'text': 'Actually, this policy might have some good aspects to consider.',
            'reply_count': 0,
            'like_count': 2,
            'retweet_count': 0,
            'quote_count': 0,
            'lang': 'en',
            'possibly_sensitive': False,
            'source': 'Twitter Web App',
            'reply_settings': 'everyone',
            'followers_count': 1000,
            'total_engagements': 2,
            'engagement_rate': 0.002,
            'clean_text': 'actually this policy might have some good aspects to consider'
        },
        {
            'tweet_id': 'reply_002',
            'author_id': 'user123',
            'conversation_id': 'conv_001',
            'created_at': pd.Timestamp('2025-01-01 10:10:00'),
            'in_reply_to_user_id': 'user123',  # Replying to own main post
            'text': 'Actually, this policy might have some good aspects to consider.',
            'reply_count': 0,
            'like_count': 1,
            'retweet_count': 0,
            'quote_count': 0,
            'lang': 'en',
            'possibly_sensitive': False,
            'source': 'Twitter Web App',
            'reply_settings': 'everyone',
            'followers_count': 1000,
            'total_engagements': 1,
            'engagement_rate': 0.001,
            'clean_text': 'actually this policy might have some good aspects to consider'
        },
        {
            'tweet_id': 'reply_003',
            'author_id': 'user123',
            'conversation_id': 'conv_001',
            'created_at': pd.Timestamp('2025-01-01 10:15:00'),
            'in_reply_to_user_id': 'user123',  # Replying to own main post
            'text': 'I think we should look at the positive side of this policy.',
            'reply_count': 0,
            'like_count': 0,
            'retweet_count': 0,
            'quote_count': 0,
            'lang': 'en',
            'possibly_sensitive': False,
            'source': 'Twitter Web App',
            'reply_settings': 'everyone',
            'followers_count': 1000,
            'total_engagements': 0,
            'engagement_rate': 0.0,
            'clean_text': 'i think we should look at the positive side of this policy'
        }
    ])
    
    print("📝 Synthetic conversation created:")
    print(f"  Total tweets: {len(synthetic_data)}")
    print(f"  Main post: {synthetic_data.iloc[0]['text'][:60]}...")
    print(f"  Own replies: {len(synthetic_data[synthetic_data['in_reply_to_user_id'] == 'user123'])}")
    
    # Initialize detector with lower thresholds for testing
    detector = ReplyBaitDetector(similarity_threshold=0.5, sentiment_threshold=0.2)
    
    print(f"\n🔬 Analyzing synthetic conversation...")
    analysis = detector.analyze_conversation_thread(synthetic_data)
    
    if 'error' in analysis:
        print(f"❌ Analysis error: {analysis['error']}")
        return
    
    print(f"📊 Analysis results:")
    print(f"  Reply-bait score: {analysis['reply_bait_score']:.3f}")
    print(f"  Is reply-bait: {analysis['is_reply_bait']}")
    print(f"  Total replies: {analysis['total_replies']}")
    print(f"  Own replies: {analysis['own_replies_count']}")
    
    # Check repetitive analysis
    rep_analysis = analysis['repetitive_analysis']
    print(f"\n🔄 Repetitive analysis:")
    print(f"  Is repetitive: {rep_analysis['is_repetitive']}")
    print(f"  Overall similarity: {rep_analysis['overall_similarity']:.3f}")
    print(f"  Repetitive pairs: {len(rep_analysis.get('repetitive_pairs', []))}")
    
    # Check sentiment analysis
    sent_analysis = analysis['sentiment_analysis']
    print(f"\n😊 Sentiment analysis:")
    print(f"  Main sentiment: {sent_analysis['main_sentiment']:.3f}")
    print(f"  Avg reply sentiment: {sent_analysis['avg_reply_sentiment']:.3f}")
    print(f"  Has inversion: {sent_analysis['has_inversion']}")
    print(f"  Inversion score: {sent_analysis['inversion_score']:.3f}")
    
    # Generate report
    print(f"\n📋 Generated report:")
    report = detector.generate_report(analysis)
    print(report)
    
    return analysis

if __name__ == "__main__":
    test_synthetic_conversation()
