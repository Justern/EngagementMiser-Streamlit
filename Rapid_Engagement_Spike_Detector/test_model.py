#!/usr/bin/env python3
"""
Test script for Rapid Engagement Spike Detector

This script tests the model functionality without requiring interactive input.
"""

from rapid_engagement_spike_detector import RapidEngagementSpikeDetector
from data_connector import EngagementDataConnector

def test_spike_detection():
    """Test the spike detection functionality."""
    print("🧪 TESTING RAPID ENGAGEMENT SPIKE DETECTOR")
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
            return False
        
        print("✅ Database connection successful!")
        
        # Get sample data
        print("\n📥 Retrieving sample engagement data...")
        sample_data = connector.get_sample_engagement_data(limit=100)
        
        if sample_data.empty:
            print("❌ No sample data retrieved")
            return False
        
        print(f"✅ Retrieved {len(sample_data)} sample records")
        
        # Find a tweet for testing
        test_tweet = sample_data.iloc[0]
        tweet_id = test_tweet['tweet_id']
        
        print(f"\n🔍 Testing with tweet ID: {tweet_id}")
        print(f"📊 Engagement: {test_tweet['total_engagements']}")
        print(f"📅 Created: {test_tweet['timestamp']}")
        
        # Test the spike detection
        print(f"\n🔍 Running spike detection analysis...")
        results = detector.analyze_tweet_engagement(tweet_id, sample_data)
        
        if 'error' in results:
            print(f"❌ Analysis error: {results['error']}")
            return False
        
        # Display results
        print(f"\n📊 TEST RESULTS:")
        print(f"🎯 Tweet ID: {results['tweet_id']}")
        print(f"📊 Spike Detection Score: {results['spike_detection_score']:.3f}")
        print(f"🔍 Spike Level: {results['spike_level']}")
        print(f"📈 Data Points: {results['data_points']}")
        
        # Component scores
        print(f"\n🔍 COMPONENT SCORES:")
        print(f"🚨 Anomaly Score: {results['anomaly_detection']['anomaly_score']:.3f}")
        print(f"⚡ Spike Score: {results['spike_detection']['spike_score']:.3f}")
        print(f"🔄 Changepoint Score: {results['changepoint_detection']['changepoint_score']:.3f}")
        print(f"📊 Pattern Score: {results['pattern_analysis'].get('pattern_score', 0):.3f}")
        
        # Counts
        print(f"\n📊 DETECTION COUNTS:")
        print(f"🚨 Anomalies: {results['anomaly_detection']['anomaly_count']}")
        print(f"⚡ Spikes: {results['spike_detection']['spike_count']}")
        print(f"🔄 Changepoints: {results['changepoint_detection']['changepoint_count']}")
        
        # Pattern analysis
        pattern = results['pattern_analysis']
        print(f"\n📊 PATTERN ANALYSIS:")
        print(f"📈 Trend: {pattern.get('trend_direction', 'N/A')}")
        print(f"📊 Volatility: {pattern.get('volatility_level', 'N/A')}")
        print(f"🔢 Total Engagement: {pattern.get('total_engagement', 0):,}")
        print(f"📊 Mean: {pattern.get('mean_engagement', 0):.1f}")
        print(f"📊 Std: {pattern.get('std_engagement', 0):.1f}")
        
        print(f"\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_synthetic_data():
    """Test with synthetic data to ensure core functionality works."""
    print(f"\n🧪 TESTING WITH SYNTHETIC DATA")
    print("=" * 60)
    
    try:
        # Create synthetic engagement data with known spikes
        import pandas as pd
        import numpy as np
        
        print("🔧 Creating synthetic engagement data...")
        
        # Generate time series with spikes
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        base_engagement = 50
        
        # Normal pattern
        normal_pattern = base_engagement + np.random.normal(0, 10, 100)
        
        # Add spikes
        normal_pattern[30:35] += np.random.normal(200, 50, 5)  # Spike 1
        normal_pattern[60:65] += np.random.normal(300, 80, 5)  # Spike 2
        
        synthetic_data = pd.DataFrame({
            'timestamp': dates,
            'tweet_id': 'synthetic_test',
            'total_engagements': normal_pattern,
            'likes': normal_pattern * 0.7,
            'retweets': normal_pattern * 0.2,
            'replies': normal_pattern * 0.1,
            'quotes': normal_pattern * 0.05,
            'author_id': 'synthetic_user',
            'text': 'Synthetic test data'
        })
        
        print(f"✅ Created synthetic data with {len(synthetic_data)} points")
        print(f"📊 Engagement range: {synthetic_data['total_engagements'].min():.1f} to {synthetic_data['total_engagements'].max():.1f}")
        
        # Test spike detection
        detector = RapidEngagementSpikeDetector()
        results = detector.analyze_tweet_engagement("synthetic_test", synthetic_data)
        
        print(f"\n📊 SYNTHETIC DATA RESULTS:")
        print(f"🎯 Spike Detection Score: {results['spike_detection_score']:.3f}")
        print(f"🔍 Spike Level: {results['spike_level']}")
        print(f"🚨 Anomalies: {results['anomaly_detection']['anomaly_count']}")
        print(f"⚡ Spikes: {results['spike_detection']['spike_count']}")
        print(f"🔄 Changepoints: {results['changepoint_detection']['changepoint_count']}")
        
        print(f"\n✅ Synthetic data test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Synthetic data test error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 COMPREHENSIVE TESTING OF RAPID ENGAGEMENT SPIKE DETECTOR")
    print("=" * 70)
    
    # Test 1: Database integration
    db_test_passed = test_spike_detection()
    
    # Test 2: Synthetic data
    synthetic_test_passed = test_synthetic_data()
    
    # Summary
    print(f"\n📊 TEST SUMMARY:")
    print(f"🔌 Database Integration: {'✅ PASSED' if db_test_passed else '❌ FAILED'}")
    print(f"🧪 Synthetic Data: {'✅ PASSED' if synthetic_test_passed else '❌ FAILED'}")
    
    if db_test_passed and synthetic_test_passed:
        print(f"\n🎉 ALL TESTS PASSED! The model is working correctly.")
        print(f"💡 You can now use: python simple_usage.py")
    else:
        print(f"\n⚠️  Some tests failed. Check the error messages above.")
    
    print(f"\n🚀 Testing completed!")
