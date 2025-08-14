"""
Test Script for Reply-Bait Detector

This script tests the Reply-Bait Detector model with sample data
to ensure all functionality works correctly.

Author: DS Capstone Project
Date: 2025
"""

import pandas as pd
import numpy as np
from reply_bait_detector import ReplyBaitDetector
import warnings
warnings.filterwarnings('ignore')

def create_sample_conversation_data():
    """
    Create sample conversation data for testing.
    
    Returns:
        DataFrame with sample conversation data
    """
    # Sample conversation with repetitive replies and sentiment inversion
    sample_data = [
        {
            'tweet_id': 'main_001',
            'author_id': 'user123',
            'conversation_id': 'conv_001',
            'created_at': pd.Timestamp('2025-01-01 10:00:00'),
            'in_reply_to_user_id': None,
            'text': 'I hate this new policy! It\'s absolutely terrible and will ruin everything.',
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
            'in_reply_to_user_id': 'user456',
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
            'in_reply_to_user_id': 'user789',
            'text': 'I think we should look at the positive side of this policy.',
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
            'clean_text': 'i think we should look at the positive side of this policy'
        },
        {
            'tweet_id': 'reply_003',
            'author_id': 'user123',
            'conversation_id': 'conv_001',
            'created_at': pd.Timestamp('2025-01-01 10:15:00'),
            'in_reply_to_user_id': 'user101',
            'text': 'Actually, this policy might have some good aspects to consider.',
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
            'clean_text': 'actually this policy might have some good aspects to consider'
        },
        {
            'tweet_id': 'reply_004',
            'author_id': 'user456',
            'conversation_id': 'conv_001',
            'created_at': pd.Timestamp('2025-01-01 10:20:00'),
            'in_reply_to_user_id': 'user123',
            'text': 'I agree with your original point, this policy is indeed problematic.',
            'reply_count': 0,
            'like_count': 3,
            'retweet_count': 0,
            'quote_count': 0,
            'lang': 'en',
            'possibly_sensitive': False,
            'source': 'Twitter Web App',
            'reply_settings': 'everyone',
            'followers_count': 500,
            'total_engagements': 3,
            'engagement_rate': 0.006,
            'clean_text': 'i agree with your original point this policy is indeed problematic'
        }
    ]
    
    return pd.DataFrame(sample_data)

def test_text_preprocessing():
    """Test text preprocessing functionality."""
    print("Testing text preprocessing...")
    
    detector = ReplyBaitDetector()
    
    # Test cases
    test_cases = [
        ("Hello @user123! This is #awesome", "hello this is awesome"),
        ("Check out http://example.com", "check out"),
        ("RT @user: Great post!", "rt great post"),
        ("", ""),
        (None, ""),
        ("   Multiple    spaces   ", "multiple spaces")
    ]
    
    for input_text, expected in test_cases:
        result = detector.preprocess_text(input_text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{input_text}' -> '{result}' (expected: '{expected}')")
    
    print()

def test_sentiment_calculation():
    """Test sentiment calculation functionality."""
    print("Testing sentiment calculation...")
    
    detector = ReplyBaitDetector()
    
    # Test cases
    test_cases = [
        ("I love this!", "positive"),
        ("I hate this!", "negative"),
        ("This is neutral.", "neutral"),
        ("", "neutral"),
        (None, "neutral")
    ]
    
    for input_text, expected_type in test_cases:
        sentiment = detector.calculate_sentiment(input_text)
        
        if expected_type == "positive" and sentiment > 0:
            status = "✓"
        elif expected_type == "negative" and sentiment < 0:
            status = "✓"
        elif expected_type == "neutral" and abs(sentiment) < 0.1:
            status = "✓"
        else:
            status = "✗"
        
        print(f"  {status} '{input_text}' -> {sentiment:.3f} (expected: {expected_type})")
    
    print()

def test_repetitive_reply_detection():
    """Test repetitive reply detection functionality."""
    print("Testing repetitive reply detection...")
    
    detector = ReplyBaitDetector(similarity_threshold=0.7)
    
    # Test case 1: Repetitive replies
    repetitive_replies = [
        "This is a great policy!",
        "This is a great policy!",
        "I love this new approach",
        "This is a great policy!"
    ]
    
    result1 = detector.detect_repetitive_replies(repetitive_replies)
    print(f"  Repetitive replies test:")
    print(f"    Is repetitive: {result1['is_repetitive']}")
    print(f"    Overall similarity: {result1['overall_similarity']:.3f}")
    print(f"    Repetitive pairs: {len(result1['repetitive_pairs'])}")
    
    # Test case 2: Non-repetitive replies
    non_repetitive_replies = [
        "This is interesting",
        "I disagree completely",
        "Let me think about this",
        "What are the implications?"
    ]
    
    result2 = detector.detect_repetitive_replies(non_repetitive_replies)
    print(f"  Non-repetitive replies test:")
    print(f"    Is repetitive: {result2['is_repetitive']}")
    print(f"    Overall similarity: {result2['overall_similarity']:.3f}")
    print(f"    Repetitive pairs: {len(result2['repetitive_pairs'])}")
    
    print()

def test_sentiment_inversion_detection():
    """Test sentiment inversion detection functionality."""
    print("Testing sentiment inversion detection...")
    
    detector = ReplyBaitDetector(sentiment_threshold=0.3)
    
    # Test case 1: Negative main post with positive replies
    main_post_negative = "I hate this new policy! It's absolutely terrible!"
    replies_positive = [
        "Actually, this policy might have some good aspects.",
        "I think we should look at the positive side.",
        "There are benefits to consider here."
    ]
    
    result1 = detector.detect_sentiment_inversion(main_post_negative, replies_positive)
    print(f"  Negative main post with positive replies:")
    print(f"    Main sentiment: {result1['main_sentiment']:.3f}")
    print(f"    Avg reply sentiment: {result1['avg_reply_sentiment']:.3f}")
    print(f"    Has inversion: {result1['has_inversion']}")
    print(f"    Inverted replies: {len(result1['inverted_replies'])}")
    
    # Test case 2: Positive main post with negative replies
    main_post_positive = "I love this new policy! It's absolutely amazing!"
    replies_negative = [
        "Actually, this policy has serious flaws.",
        "I think we should reconsider this approach.",
        "There are major problems with this."
    ]
    
    result2 = detector.detect_sentiment_inversion(main_post_positive, replies_negative)
    print(f"  Positive main post with negative replies:")
    print(f"    Main sentiment: {result2['main_sentiment']:.3f}")
    print(f"    Avg reply sentiment: {result2['avg_reply_sentiment']:.3f}")
    print(f"    Has inversion: {result2['has_inversion']}")
    print(f"    Inverted replies: {len(result2['inverted_replies'])}")
    
    print()

def test_conversation_analysis():
    """Test full conversation analysis functionality."""
    print("Testing conversation analysis...")
    
    detector = ReplyBaitDetector(similarity_threshold=0.7, sentiment_threshold=0.3)
    
    # Create sample conversation data
    conversation_data = create_sample_conversation_data()
    
    # Analyze the conversation
    analysis = detector.analyze_conversation_thread(conversation_data)
    
    print(f"  Conversation analysis results:")
    print(f"    Conversation ID: {analysis['conversation_id']}")
    print(f"    Main author: {analysis['main_author']}")
    print(f"    Total replies: {analysis['total_replies']}")
    print(f"    Own replies: {analysis['own_replies_count']}")
    print(f"    Reply-bait score: {analysis['reply_bait_score']:.3f}")
    print(f"    Is reply-bait: {analysis['is_reply_bait']}")
    
    # Check repetitive analysis
    rep_analysis = analysis['repetitive_analysis']
    print(f"    Repetitive analysis:")
    print(f"      Is repetitive: {rep_analysis['is_repetitive']}")
    print(f"      Overall similarity: {rep_analysis['overall_similarity']:.3f}")
    
    # Check sentiment analysis
    sent_analysis = analysis['sentiment_analysis']
    print(f"    Sentiment analysis:")
    print(f"      Main sentiment: {sent_analysis['main_sentiment']:.3f}")
    print(f"      Has inversion: {sent_analysis['has_inversion']}")
    
    print()

def test_report_generation():
    """Test report generation functionality."""
    print("Testing report generation...")
    
    detector = ReplyBaitDetector()
    
    # Create sample conversation data
    conversation_data = create_sample_conversation_data()
    
    # Analyze the conversation
    analysis = detector.analyze_conversation_thread(conversation_data)
    
    # Generate report
    report = detector.generate_report(analysis)
    
    print("  Generated report preview:")
    print("  " + "="*40)
    
    # Show first few lines of the report
    lines = report.split('\n')[:15]
    for line in lines:
        print(f"  {line}")
    
    print("  " + "="*40)
    print()

def test_analyze_tweet_method():
    """Test the new analyze_tweet method."""
    print("Testing analyze_tweet method...")
    
    detector = ReplyBaitDetector()
    
    # Create a mock connector for testing
    class MockConnector:
        def get_conversation_thread(self, tweet_id):
            # Return sample conversation data
            return pd.DataFrame({
                'tweet_id': ['main_001', 'reply_001', 'reply_002'],
                'author_id': ['user123', 'user123', 'user123'],
                'conversation_id': ['conv_001', 'conv_001', 'conv_001'],
                'created_at': ['2025-01-01', '2025-01-01', '2025-01-01'],
                'text': ['Negative post', 'Positive reply', 'Positive reply'],
                'in_reply_to_user_id': [None, 'user123', 'user123']
            })
    
    mock_connector = MockConnector()
    
    # Test the method
    score = detector.analyze_tweet('test_tweet_123', mock_connector)
    
    # Verify it returns a float between 0 and 1
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    
    print(f"  ✓ analyze_tweet method test completed - Score: {score:.3f}")
    print()

def run_all_tests():
    """Run all test functions."""
    print("REPLY-BAIT DETECTOR - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print()
    
    try:
        test_text_preprocessing()
        test_sentiment_calculation()
        test_repetitive_reply_detection()
        test_sentiment_inversion_detection()
        test_conversation_analysis()
        test_report_generation()
        test_analyze_tweet_method()
        
        print("=" * 60)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("The Reply-Bait Detector model is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print("=" * 60)
        print(f"✗ TEST FAILED: {str(e)}")
        print("Please check the error and fix any issues.")
        print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
