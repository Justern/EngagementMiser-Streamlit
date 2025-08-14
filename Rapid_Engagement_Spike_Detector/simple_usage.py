#!/usr/bin/env python3
"""
Simple Usage Example for Rapid Engagement Spike Detector
Input: Tweet ID from database
Output: Engagement spike detection score from 0-1 (higher = more likely to have spikes)
"""

from rapid_engagement_spike_detector import RapidEngagementSpikeDetector
from data_connector import EngagementDataConnector
import os

def analyze_tweet_engagement_spikes(tweet_id: str) -> float:
    """
    Analyze engagement patterns for a specific tweet to detect rapid spikes.
    
    Args:
        tweet_id: Tweet ID to analyze
        
    Returns:
        float: Spike detection score from 0-1 (higher = more likely to have spikes)
    """
    print(f"🔍 Tweet ID: {tweet_id}")
    
    try:
        # Initialize database connector
        connector = EngagementDataConnector()
        
        # Test connection first
        if not connector.test_connection():
            print("❌ Database connection failed!")
            print("Please check your connection string and ensure SQL Server is running.")
            return 0.0
        
        print("✅ Database connection successful!")
        
        # Retrieve engagement timeline data
        print("📥 Retrieving engagement timeline data...")
        engagement_data = connector.get_tweet_engagement_timeline(tweet_id, hours_back=48)
        
        if engagement_data.empty:
            print(f"❌ No engagement data found for tweet {tweet_id}")
            return 0.0
        
        print("✅ Engagement timeline data retrieved successfully!")
        print(f"📊 Data points: {len(engagement_data)}")
        print(f"📅 Time range: {engagement_data['timestamp'].min()} to {engagement_data['timestamp'].max()}")
        print(f"📈 Engagement range: {engagement_data['total_engagements'].min()} to {engagement_data['total_engagements'].max()}")
        
        # Initialize spike detector
        print(f"\n🔍 Initializing spike detector...")
        detector = RapidEngagementSpikeDetector()
        
        # Analyze the engagement patterns
        print(f"🔍 Analyzing engagement patterns for spike detection...")
        results = detector.analyze_tweet_engagement(tweet_id, engagement_data)
        
        if 'error' in results:
            print(f"❌ Analysis error: {results['error']}")
            return 0.0
        
        # Extract key results
        spike_score = results['spike_detection_score']
        spike_level = results['spike_level']
        anomaly_count = results['anomaly_detection']['anomaly_count']
        spike_count = results['spike_detection']['spike_count']
        changepoint_count = results['changepoint_detection']['changepoint_count']
        
        # Display results
        print(f"\n📊 SPIKE DETECTION RESULTS:")
        print(f"🎯 Spike Detection Score: {spike_score:.3f}")
        print(f"🔍 Spike Level: {spike_level}")
        print(f"📈 Anomalies Detected: {anomaly_count}")
        print(f"⚡ Spikes Detected: {spike_count}")
        print(f"🔄 Changepoints Detected: {changepoint_count}")
        
        # Interpret the score
        if spike_score >= 0.7:
            interpretation = "HIGH_SPIKE - Very likely to have rapid engagement spikes"
        elif spike_score >= 0.4:
            interpretation = "MODERATE_SPIKE - Likely to have some engagement spikes"
        elif spike_score >= 0.2:
            interpretation = "LOW_SPIKE - Minimal engagement spikes detected"
        else:
            interpretation = "NO_SPIKE - No significant engagement spikes detected"
        
        print(f"🔍 Interpretation: {interpretation}")
        
        # Show detailed analysis if available
        if results['anomaly_detection']['anomalies']:
            print(f"\n🚨 ANOMALIES DETECTED:")
            for i, anomaly in enumerate(results['anomaly_detection']['anomalies'][:3]):  # Show first 3
                print(f"  {i+1}. Time: {anomaly['timestamp']}, Score: {anomaly['anomaly_score']:.3f}")
        
        if results['spike_detection']['spikes']:
            print(f"\n⚡ SPIKES DETECTED:")
            for i, spike in enumerate(results['spike_detection']['spikes'][:3]):  # Show first 3
                print(f"  {i+1}. Time: {spike['timestamp']}, Method: {spike['method']}, Score: {spike['spike_score']:.3f}")
        
        if results['changepoint_detection']['changepoints']:
            print(f"\n🔄 CHANGEPOINTS DETECTED:")
            for i, changepoint in enumerate(results['changepoint_detection']['changepoints'][:3]):  # Show first 3
                print(f"  {i+1}. Time: {changepoint['timestamp']}, Index: {changepoint['index']}")
        
        # Show pattern analysis
        pattern_analysis = results['pattern_analysis']
        if pattern_analysis:
            print(f"\n📊 PATTERN ANALYSIS:")
            print(f"  📈 Trend Direction: {pattern_analysis.get('trend_direction', 'N/A')}")
            print(f"  📊 Volatility Level: {pattern_analysis.get('volatility_level', 'N/A')}")
            print(f"  🔢 Total Engagement: {pattern_analysis.get('total_engagement', 0):,}")
            print(f"  📊 Mean Engagement: {pattern_analysis.get('mean_engagement', 0):.1f}")
        
        return spike_score
        
    except Exception as e:
        print(f"❌ Error analyzing tweet engagement spikes: {str(e)}")
        return 0.0

def analyze_conversation_spikes(conversation_id: str) -> float:
    """
    Analyze engagement patterns for an entire conversation thread.
    
    Args:
        conversation_id: Conversation ID to analyze
        
    Returns:
        float: Spike detection score for the conversation
    """
    print(f"🔍 Conversation ID: {conversation_id}")
    
    try:
        # Initialize database connector
        connector = EngagementDataConnector()
        
        # Test connection first
        if not connector.test_connection():
            print("❌ Database connection failed!")
            return 0.0
        
        print("✅ Database connection successful!")
        
        # Retrieve conversation engagement data
        print("📥 Retrieving conversation engagement data...")
        conversation_data = connector.get_conversation_engagement_timeline(conversation_id, hours_back=72)
        
        if conversation_data.empty:
            print(f"❌ No conversation data found for {conversation_id}")
            return 0.0
        
        print("✅ Conversation data retrieved successfully!")
        print(f"📊 Tweets in conversation: {len(conversation_data)}")
        print(f"📅 Time range: {conversation_data['timestamp'].min()} to {conversation_data['timestamp'].max()}")
        
        # Initialize spike detector
        detector = RapidEngagementSpikeDetector()
        
        # Analyze the conversation
        print(f"🔍 Analyzing conversation engagement patterns...")
        results = detector.analyze_tweet_engagement(conversation_id, conversation_data)
        
        if 'error' in results:
            print(f"❌ Analysis error: {results['error']}")
            return 0.0
        
        # Display results
        spike_score = results['spike_detection_score']
        spike_level = results['spike_level']
        
        print(f"\n📊 CONVERSATION SPIKE DETECTION:")
        print(f"🎯 Spike Detection Score: {spike_score:.3f}")
        print(f"🔍 Spike Level: {spike_level}")
        print(f"📈 Anomalies: {results['anomaly_detection']['anomaly_count']}")
        print(f"⚡ Spikes: {results['spike_detection']['spike_count']}")
        print(f"🔄 Changepoints: {results['changepoint_detection']['changepoint_count']}")
        
        return spike_score
        
    except Exception as e:
        print(f"❌ Error analyzing conversation spikes: {str(e)}")
        return 0.0

def main():
    """Main function for engagement spike analysis."""
    print("RAPID ENGAGEMENT SPIKE DETECTOR - TWEET ANALYSIS")
    print("=" * 60)
    print("Input: Tweet ID or Conversation ID from database")
    print("Output: Engagement spike detection score from 0-1")
    print("=" * 60)
    
    print("\nChoose analysis type:")
    print("1. Analyze single tweet")
    print("2. Analyze conversation thread")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        tweet_id = input("\nEnter tweet ID to analyze: ").strip()
        if tweet_id:
            score = analyze_tweet_engagement_spikes(tweet_id)
            print(f"\n🎯 Final Spike Detection Score: {score:.3f}")
        else:
            print("No tweet ID provided.")
    
    elif choice == "2":
        conversation_id = input("\nEnter conversation ID to analyze: ").strip()
        if conversation_id:
            score = analyze_conversation_spikes(conversation_id)
            print(f"\n🎯 Final Conversation Spike Score: {score:.3f}")
        else:
            print("No conversation ID provided.")
    
    else:
        print("Invalid choice. Please run again and select 1 or 2.")

if __name__ == "__main__":
    main()
