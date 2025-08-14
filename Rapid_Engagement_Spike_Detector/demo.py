#!/usr/bin/env python3
"""
Demo script for Rapid Engagement Spike Detector

This script demonstrates the full functionality of the model using real database data.
"""

from rapid_engagement_spike_detector import RapidEngagementSpikeDetector
from data_connector import EngagementDataConnector
import pandas as pd

def demo_spike_detection():
    """Demonstrate spike detection with real database data."""
    print("🚀 RAPID ENGAGEMENT SPIKE DETECTOR DEMO")
    print("=" * 60)
    
    try:
        # Initialize components
        print("🔧 Initializing components...")
        detector = RapidEngagementSpikeDetector()
        connector = EngagementDataConnector()
        
        # Test database connection
        print("🔌 Testing database connection...")
        if not connector.test_connection():
            print("❌ Database connection failed!")
            return
        
        print("✅ Database connection successful!")
        
        # Get sample data
        print("\n📥 Retrieving sample engagement data...")
        sample_data = connector.get_sample_engagement_data(limit=200)
        
        if sample_data.empty:
            print("❌ No sample data retrieved")
            return
        
        print(f"✅ Retrieved {len(sample_data)} sample records")
        print(f"📊 Data range: {sample_data['timestamp'].min()} to {sample_data['timestamp'].max()}")
        print(f"📈 Engagement range: {sample_data['total_engagements'].min()} to {sample_data['total_engagements'].max()}")
        
        # Find a tweet with good engagement for analysis
        print("\n🔍 Finding tweet for detailed analysis...")
        high_engagement_tweets = sample_data[sample_data['total_engagements'] > 50]
        
        if high_engagement_tweets.empty:
            print("⚠️  No high engagement tweets found, using random sample")
            analysis_tweet = sample_data.iloc[0]
        else:
            analysis_tweet = high_engagement_tweets.iloc[0]
        
        tweet_id = analysis_tweet['tweet_id']
        print(f"📝 Selected tweet ID: {tweet_id}")
        print(f"📊 Engagement: {analysis_tweet['total_engagements']}")
        print(f"📅 Created: {analysis_tweet['timestamp']}")
        print(f"📝 Text: {analysis_tweet['text'][:100]}...")
        
        # Get engagement timeline for this tweet
        print(f"\n📈 Retrieving engagement timeline for tweet {tweet_id}...")
        timeline_data = connector.get_tweet_engagement_timeline(tweet_id, hours_back=24)
        
        if timeline_data.empty:
            print("⚠️  No timeline data found, using sample data for demonstration")
            # Use the sample data as a proxy for timeline
            timeline_data = sample_data.copy()
            timeline_data['tweet_id'] = tweet_id
        
        print(f"✅ Timeline data: {len(timeline_data)} data points")
        
        # Analyze the engagement patterns
        print(f"\n🔍 Analyzing engagement patterns for spike detection...")
        results = detector.analyze_tweet_engagement(tweet_id, timeline_data)
        
        if 'error' in results:
            print(f"❌ Analysis error: {results['error']}")
            return
        
        # Display comprehensive results
        print(f"\n📊 COMPREHENSIVE SPIKE DETECTION RESULTS")
        print("=" * 60)
        
        # Main results
        print(f"🎯 Tweet ID: {results['tweet_id']}")
        print(f"📊 Spike Detection Score: {results['spike_detection_score']:.3f}")
        print(f"🔍 Spike Level: {results['spike_level']}")
        print(f"📈 Data Points Analyzed: {results['data_points']}")
        
        # Time range
        time_range = results['time_range']
        print(f"⏰ Analysis Time Range: {time_range['start']} to {time_range['end']}")
        print(f"⏱️  Duration: {time_range['duration_hours']:.1f} hours")
        
        # Anomaly detection
        anomaly_results = results['anomaly_detection']
        print(f"\n🚨 ANOMALY DETECTION:")
        print(f"  📊 Anomaly Score: {anomaly_results['anomaly_score']:.3f}")
        print(f"  🔍 Anomalies Found: {anomaly_results['anomaly_count']}")
        
        if anomaly_results['anomalies']:
            print("  📋 Top Anomalies:")
            for i, anomaly in enumerate(anomaly_results['anomalies'][:3]):
                print(f"    {i+1}. Time: {anomaly['timestamp']}, Score: {anomaly['anomaly_score']:.3f}")
        
        # Spike detection
        spike_results = results['spike_detection']
        print(f"\n⚡ SPIKE DETECTION:")
        print(f"  📊 Spike Score: {spike_results['spike_score']:.3f}")
        print(f"  🔍 Spikes Found: {spike_results['spike_count']}")
        
        if spike_results['spikes']:
            print("  📋 Top Spikes:")
            for i, spike in enumerate(spike_results['spikes'][:3]):
                print(f"    {i+1}. Time: {spike['timestamp']}, Method: {spike['method']}, Score: {spike['spike_score']:.3f}")
        
        # Changepoint detection
        changepoint_results = results['changepoint_detection']
        print(f"\n🔄 CHANGEPOINT DETECTION:")
        print(f"  📊 Changepoint Score: {changepoint_results['changepoint_score']:.3f}")
        print(f"  🔍 Changepoints Found: {changepoint_results['changepoint_count']}")
        
        if changepoint_results['changepoints']:
            print("  📋 Changepoints:")
            for i, changepoint in enumerate(changepoint_results['changepoints'][:3]):
                print(f"    {i+1}. Time: {changepoint['timestamp']}, Index: {changepoint['index']}")
        
        # Pattern analysis
        pattern_results = results['pattern_analysis']
        print(f"\n📊 PATTERN ANALYSIS:")
        print(f"  📈 Trend Direction: {pattern_results.get('trend_direction', 'N/A')}")
        print(f"  📊 Volatility Level: {pattern_results.get('volatility_level', 'N/A')}")
        print(f"  🔢 Total Engagement: {pattern_results.get('total_engagement', 0):,}")
        print(f"  📊 Mean Engagement: {pattern_results.get('mean_engagement', 0):.1f}")
        print(f"  📊 Std Engagement: {pattern_results.get('std_engagement', 0):.1f}")
        print(f"  📊 Pattern Score: {pattern_results.get('pattern_score', 0):.3f}")
        
        # Score interpretation
        print(f"\n🎯 SCORE INTERPRETATION:")
        score = results['spike_detection_score']
        if score >= 0.7:
            interpretation = "HIGH_SPIKE - Very likely to have rapid engagement spikes"
            recommendation = "Monitor closely for viral content or coordinated activity"
        elif score >= 0.4:
            interpretation = "MODERATE_SPIKE - Likely to have some engagement spikes"
            recommendation = "Watch for unusual engagement patterns"
        elif score >= 0.2:
            interpretation = "LOW_SPIKE - Minimal engagement spikes detected"
            recommendation = "Normal engagement patterns"
        else:
            interpretation = "NO_SPIKE - No significant engagement spikes detected"
            recommendation = "Stable, organic engagement"
        
        print(f"  🔍 {interpretation}")
        print(f"  💡 Recommendation: {recommendation}")
        
        # Show sample of raw data
        if results.get('raw_data'):
            print(f"\n📋 SAMPLE RAW DATA (First 5 records):")
            raw_df = pd.DataFrame(results['raw_data'])
            print(raw_df[['timestamp', 'total_engagements', 'likes', 'retweets', 'replies']].head())
        
        print(f"\n✅ Demo completed successfully!")
        print(f"🎯 Final Spike Detection Score: {score:.3f}")
        
    except Exception as e:
        print(f"❌ Demo error: {str(e)}")
        import traceback
        traceback.print_exc()

def demo_conversation_analysis():
    """Demonstrate conversation-level spike detection."""
    print("\n🔗 CONVERSATION SPIKE DETECTION DEMO")
    print("=" * 60)
    
    try:
        # Initialize components
        detector = RapidEngagementSpikeDetector()
        connector = EngagementDataConnector()
        
        # Get sample data to find a conversation
        sample_data = connector.get_sample_engagement_data(limit=500)
        
        if sample_data.empty:
            print("❌ No sample data for conversation analysis")
            return
        
        # Find tweets with conversation IDs
        conversations = sample_data[sample_data['conversation_id'].notna()]
        
        if conversations.empty:
            print("⚠️  No conversation data found")
            return
        
        # Get a conversation ID
        conversation_id = conversations.iloc[0]['conversation_id']
        print(f"🔗 Analyzing conversation: {conversation_id}")
        
        # Get conversation timeline
        conv_data = connector.get_conversation_engagement_timeline(conversation_id, hours_back=48)
        
        if conv_data.empty:
            print("⚠️  No conversation timeline data")
            return
        
        print(f"✅ Conversation data: {len(conv_data)} tweets")
        print(f"📅 Time range: {conv_data['timestamp'].min()} to {conv_data['timestamp'].max()}")
        
        # Analyze conversation
        results = detector.analyze_tweet_engagement(conversation_id, conv_data)
        
        if 'error' in results:
            print(f"❌ Conversation analysis error: {results['error']}")
            return
        
        print(f"\n📊 CONVERSATION ANALYSIS RESULTS:")
        print(f"🎯 Conversation ID: {conversation_id}")
        print(f"📊 Spike Detection Score: {results['spike_detection_score']:.3f}")
        print(f"🔍 Spike Level: {results['spike_level']}")
        print(f"📈 Tweets in Conversation: {results['data_points']}")
        print(f"🚨 Anomalies: {results['anomaly_detection']['anomaly_count']}")
        print(f"⚡ Spikes: {results['spike_detection']['spike_count']}")
        print(f"🔄 Changepoints: {results['changepoint_detection']['changepoint_count']}")
        
    except Exception as e:
        print(f"❌ Conversation demo error: {str(e)}")

if __name__ == "__main__":
    # Run main demo
    demo_spike_detection()
    
    # Run conversation demo
    demo_conversation_analysis()
    
    print(f"\n🎉 All demos completed!")
    print(f"💡 To use the model interactively, run: python simple_usage.py")
