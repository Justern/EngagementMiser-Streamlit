#!/usr/bin/env python3
"""
Coordinated Account Network Model - Test Script
==============================================

This script provides comprehensive testing of the Coordinated Account Network Model
including database integration and synthetic data validation.
"""

import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from coordinated_account_network_model import CoordinatedAccountNetworkModel
from data_connector import NetworkDataConnector

def test_database_integration():
    """Test database connectivity and data retrieval."""
    print("🔌 TESTING DATABASE INTEGRATION")
    print("=" * 50)
    
    try:
        # Initialize connector
        connector = NetworkDataConnector(
            server='localhost',
            database='EngagementMiser'
        )
        
        # Test connection
        if connector.test_connection():
            print("✅ Database connection successful!")
        else:
            print("❌ Database connection failed!")
            return False
        
        # Test sample data retrieval
        print("\n📥 Testing optimized sample data retrieval...")
        user_data, interactions_data = connector.get_optimized_sample_data(limit=20)
        
        if not user_data.empty:
            print(f"✅ Retrieved {len(user_data)} users and {len(interactions_data)} interactions")
            print(f"📊 Sample user: {user_data.iloc[0]['username']}")
        else:
            print("❌ No sample data retrieved")
            return False
        
        # Test specific tweet retrieval
        if not interactions_data.empty:
            sample_tweet_id = interactions_data.iloc[0]['tweet_id']
            print(f"\n🔍 Testing tweet data retrieval for ID: {sample_tweet_id}")
            
            tweet_data, tweet_users, tweet_interactions = connector.get_tweet_network_data(sample_tweet_id)
            
            if not tweet_data.empty:
                print(f"✅ Tweet data retrieved successfully")
                print(f"   • Participants: {len(tweet_users)}")
                print(f"   • Interactions: {len(tweet_interactions)}")
            else:
                print("❌ Tweet data retrieval failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Database integration test failed: {e}")
        return False

def test_synthetic_data():
    """Test model functionality with synthetic data."""
    print("\n🧪 TESTING WITH SYNTHETIC DATA")
    print("=" * 50)
    
    try:
        # Create synthetic user data
        print("🔧 Creating synthetic user data...")
        synthetic_users = pd.DataFrame({
            'user_id': [f'user_{i}' for i in range(10)],
            'username': [f'user{i}' for i in range(10)],
            'followers_count': np.random.randint(10, 1000, 10),
            'following_count': np.random.randint(5, 500, 10),
            'tweet_count': np.random.randint(50, 2000, 10),
            'created_at': [datetime.now() - timedelta(days=np.random.randint(1, 365)) for _ in range(10)]
        })
        
        # Create synthetic interaction data
        print("🔧 Creating synthetic interaction data...")
        interactions = []
        for i in range(20):
            user_id = f'user_{np.random.randint(0, 10)}'
            interacted_with = f'user_{np.random.randint(0, 10)}'
            if user_id != interacted_with:
                interactions.append({
                    'user_id': user_id,
                    'interacted_with_id': interacted_with,
                    'created_at': datetime.now() - timedelta(minutes=np.random.randint(1, 60)),
                    'interaction_count': np.random.randint(1, 5)
                })
        
        synthetic_interactions = pd.DataFrame(interactions)
        
        print(f"✅ Created {len(synthetic_users)} synthetic users and {len(synthetic_interactions)} interactions")
        
        # Test model with synthetic data
        model = CoordinatedAccountNetworkModel()
        
        # Test preprocessing
        print("\n🔧 Testing data preprocessing...")
        processed_users = model.preprocess_network_data(synthetic_users)
        print(f"✅ Preprocessed {len(processed_users)} user records")
        
        # Test feature extraction
        print("🔧 Testing feature extraction...")
        features = model.extract_network_features(processed_users)
        print(f"✅ Extracted {len(features.columns)} features")
        
        # Test graph building
        print("🔧 Testing graph construction...")
        G = model.build_interaction_graph(synthetic_interactions)
        print(f"✅ Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Test network metrics
        print("🔧 Testing network metrics calculation...")
        metrics = model.calculate_network_metrics(G)
        print(f"✅ Calculated metrics: density={metrics['density']:.3f}, clustering={metrics['clustering_coefficient']:.3f}")
        
        # Test clustering
        print("🔧 Testing clustering detection...")
        cluster_labels, _ = model.detect_coordinated_clusters(features)
        unique_clusters = set(cluster_labels)
        print(f"✅ Detected {len(unique_clusters)} clusters")
        
        # Test coordination analysis
        print("🔧 Testing coordination analysis...")
        if not synthetic_users.empty:
            sample_user_id = synthetic_users.iloc[0]['user_id']
            results = model.analyze_user_coordination(sample_user_id, synthetic_users, synthetic_interactions)
            
            if results:
                print(f"✅ Coordination analysis successful: score={results['coordination_score']:.3f}")
            else:
                print("❌ Coordination analysis failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Synthetic data test failed: {e}")
        return False

def test_real_data_analysis():
    """Test model with real database data."""
    print("\n📊 TESTING WITH REAL DATA")
    print("=" * 50)
    
    try:
        # Initialize components
        model = CoordinatedAccountNetworkModel()
        connector = NetworkDataConnector('localhost', 'EngagementMiser')
        
        # Get real data
        print("📥 Retrieving optimized real network data...")
        user_data, interactions_data = connector.get_optimized_sample_data(limit=50)
        
        if user_data.empty or interactions_data.empty:
            print("❌ Insufficient real data for testing")
            return False
        
        print(f"✅ Retrieved {len(user_data)} users and {len(interactions_data)} interactions")
        
        # Test real data analysis
        print("\n🔍 Testing real data analysis...")
        
        # Preprocess data
        processed_data = model.preprocess_network_data(user_data)
        print(f"✅ Preprocessed {len(processed_data)} user records")
        
        # Extract features
        features = model.extract_network_features(processed_data)
        print(f"✅ Extracted {len(features.columns)} features")
        
        # Build graph
        G = model.build_interaction_graph(interactions_data)
        print(f"✅ Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Calculate metrics
        metrics = model.calculate_network_metrics(G)
        print(f"✅ Network metrics: density={metrics['density']:.3f}, clustering={metrics['clustering_coefficient']:.3f}")
        
        # Detect clusters
        cluster_labels, _ = model.detect_coordinated_clusters(features)
        unique_clusters = set(cluster_labels)
        print(f"✅ Detected {len(unique_clusters)} clusters")
        
        # Test coordination analysis on real user
        if not user_data.empty:
            sample_user = user_data.iloc[0]
            user_id = sample_user['user_id']
            
            print(f"\n🎯 Testing coordination analysis on user: {sample_user['username']}")
            results = model.analyze_user_coordination(user_id, user_data, interactions_data)
            
            if results:
                print(f"✅ Real user analysis successful:")
                print(f"   • Score: {results['coordination_score']:.3f}")
                print(f"   • Level: {results['coordination_level']}")
                print(f"   • Cluster: {results['cluster_id']}")
            else:
                print("❌ Real user analysis failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
        return False

def main():
    """Main testing function."""
    print("🧪 COMPREHENSIVE TESTING OF COORDINATED ACCOUNT NETWORK MODEL")
    print("=" * 70)
    
    test_results = []
    
    try:
        # Test database integration
        db_test = test_database_integration()
        test_results.append(("Database Integration", db_test))
        
        # Test synthetic data
        synthetic_test = test_synthetic_data()
        test_results.append(("Synthetic Data", synthetic_test))
        
        # Test real data analysis
        real_data_test = test_real_data_analysis()
        test_results.append(("Real Data Analysis", real_data_test))
        
        # Summary
        print("\n📊 TEST SUMMARY")
        print("=" * 50)
        
        passed = 0
        total = len(test_results)
        
        for test_name, result in test_results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! The model is working correctly.")
            print("💡 You can now use: python simple_usage.py")
        else:
            print("⚠️ Some tests failed. Please check the errors above.")
        
        print("\n🚀 Testing completed!")
        
    except Exception as e:
        print(f"\n❌ Testing failed with unexpected error: {e}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    main()
