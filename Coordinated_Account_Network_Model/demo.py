#!/usr/bin/env python3
"""
Coordinated Account Network Model - Demo Script
==============================================

This script demonstrates the capabilities of the Coordinated Account Network Model
by analyzing real data from the database.
"""

import pandas as pd
import numpy as np
from coordinated_account_network_model import CoordinatedAccountNetworkModel
from data_connector import NetworkDataConnector

def demo_database_integration():
    """Demonstrate database connectivity and data retrieval."""
    print("🔌 TESTING DATABASE INTEGRATION")
    print("=" * 50)
    
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
    
    # Get sample data
    print("\n📥 Retrieving sample network data...")
    user_data, interactions_data = connector.get_sample_network_data(limit=50)
    
    if not user_data.empty:
        print(f"✅ Retrieved {len(user_data)} users and {len(interactions_data)} interactions")
        print(f"📊 Sample user: {user_data.iloc[0]['username']} ({user_data.iloc[0]['followers_count']} followers)")
    else:
        print("❌ No sample data retrieved")
        return False
    
    return True

def demo_network_analysis():
    """Demonstrate network analysis capabilities."""
    print("\n🌐 DEMONSTRATING NETWORK ANALYSIS")
    print("=" * 50)
    
    # Initialize model
    model = CoordinatedAccountNetworkModel()
    
    # Get sample data
    connector = NetworkDataConnector('localhost', 'EngagementMiser')
    user_data, interactions_data = connector.get_sample_network_data(limit=100)
    
    if user_data.empty or interactions_data.empty:
        print("❌ Insufficient data for analysis")
        return
    
    print(f"📊 Analyzing network with {len(user_data)} users and {len(interactions_data)} interactions")
    
    # Preprocess data
    processed_data = model.preprocess_network_data(user_data)
    print(f"✅ Preprocessed {len(processed_data)} user records")
    
    # Extract features
    features = model.extract_network_features(processed_data)
    print(f"✅ Extracted {len(features.columns)} features")
    
    # Build interaction graph
    G = model.build_interaction_graph(interactions_data)
    print(f"✅ Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Calculate network metrics
    metrics = model.calculate_network_metrics(G)
    print(f"📈 Network density: {metrics['density']:.3f}")
    print(f"📈 Clustering coefficient: {metrics['clustering_coefficient']:.3f}")
    print(f"📈 Modularity: {metrics['modularity']:.3f}")
    
    # Detect clusters
    cluster_labels, _ = model.detect_coordinated_clusters(features)
    unique_clusters = set(cluster_labels)
    print(f"🔗 Detected {len(unique_clusters)} clusters")
    
    return model, user_data, interactions_data

def demo_coordination_detection(model, user_data, interactions_data):
    """Demonstrate coordination detection on sample data."""
    print("\n🔍 DEMONSTRATING COORDINATION DETECTION")
    print("=" * 50)
    
    if user_data.empty:
        print("❌ No user data available")
        return
    
    # Select a sample user for analysis
    sample_user = user_data.iloc[0]
    user_id = sample_user['user_id']
    
    print(f"🎯 Analyzing user: {sample_user['username']} (ID: {user_id})")
    print(f"📊 Profile: {sample_user['followers_count']} followers, {sample_user['following_count']} following")
    
    # Analyze user coordination
    results = model.analyze_user_coordination(user_id, user_data, interactions_data)
    
    if results:
        print(f"\n📊 COORDINATION ANALYSIS RESULTS:")
        print(f"   • Score: {results['coordination_score']:.3f}")
        print(f"   • Level: {results['coordination_level']}")
        print(f"   • Cluster: {results['cluster_id']}")
        
        if results['suspicious_patterns']:
            print(f"\n🚨 Suspicious patterns detected:")
            for pattern in results['suspicious_patterns']:
                print(f"   • {pattern}")
        else:
            print(f"\n✅ No suspicious patterns detected")
    else:
        print("❌ Analysis failed")

def demo_tweet_analysis():
    """Demonstrate tweet-level coordination analysis."""
    print("\n🐦 DEMONSTRATING TWEET ANALYSIS")
    print("=" * 50)
    
    # Initialize components
    model = CoordinatedAccountNetworkModel()
    connector = NetworkDataConnector('localhost', 'EngagementMiser')
    
    # Get a sample tweet (you can replace this with a real tweet ID)
    print("🔍 Looking for sample tweets in database...")
    
    # Try to get sample data to find a tweet
    user_data, interactions_data = connector.get_sample_network_data(limit=10)
    
    if not interactions_data.empty:
        # Get the first tweet from interactions
        sample_tweet_id = interactions_data.iloc[0]['tweet_id']
        print(f"📝 Analyzing tweet: {sample_tweet_id}")
        
        # Get tweet network data
        tweet_data, tweet_users, tweet_interactions = connector.get_tweet_network_data(sample_tweet_id)
        
        if not tweet_data.empty:
            print(f"✅ Retrieved tweet data: {len(tweet_users)} participants, {len(tweet_interactions)} interactions")
            
            # Analyze coordination
            results = model.analyze_tweet_coordination(sample_tweet_id, tweet_users, tweet_interactions)
            
            if results:
                print(f"\n📊 TWEET COORDINATION RESULTS:")
                print(f"   • Score: {results['coordination_score']:.3f}")
                print(f"   • Level: {results['coordination_level']}")
                print(f"   • Network: {results['network_metrics']['node_count']} nodes, {results['network_metrics']['edge_count']} edges")
            else:
                print("❌ Tweet analysis failed")
        else:
            print("❌ No tweet data found")
    else:
        print("❌ No interaction data available for tweet analysis")

def main():
    """Main demonstration function."""
    print("🌐 COORDINATED ACCOUNT NETWORK MODEL DEMO")
    print("=" * 60)
    print("This demo showcases the model's capabilities for detecting")
    print("coordinated behavior in social media networks.")
    print()
    
    try:
        # Test database integration
        if not demo_database_integration():
            print("\n❌ Demo cannot continue due to database issues")
            return
        
        # Demonstrate network analysis
        model, user_data, interactions_data = demo_network_analysis()
        
        # Demonstrate coordination detection
        demo_coordination_detection(model, user_data, interactions_data)
        
        # Demonstrate tweet analysis
        demo_tweet_analysis()
        
        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("The Coordinated Account Network Model is ready for use!")
        print("Run 'python simple_usage.py' to start analyzing tweets.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check your database connection and dependencies.")

if __name__ == "__main__":
    main()
