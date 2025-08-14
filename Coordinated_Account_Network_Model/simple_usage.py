#!/usr/bin/env python3
"""
Coordinated Account Network Model - Simple Usage Script
======================================================

This script provides a simple interface to analyze coordination patterns
for tweets using the Coordinated Account Network Model.

Input: Tweet ID
Output: Coordination score from 0-1 (higher = more coordinated behavior)
"""

import sys
import os
from coordinated_account_network_model import CoordinatedAccountNetworkModel
from data_connector import NetworkDataConnector

def analyze_tweet_by_id(tweet_id):
    """
    Analyze coordination patterns for a specific tweet ID.
    
    Args:
        tweet_id (str): Tweet ID to analyze
        
    Returns:
        dict: Analysis results
    """
    try:
        # Initialize the model
        model = CoordinatedAccountNetworkModel()
        
        # Initialize database connector
        connector = NetworkDataConnector(
            server='localhost',
            database='EngagementMiser'
        )
        
        # Test database connection
        if not connector.test_connection():
            print("❌ Database connection failed!")
            return None
        
        print(f"🔍 Analyzing coordination patterns for tweet: {tweet_id}")
        
        # Get network data for the tweet
        tweet_data, user_data, interactions_data = connector.get_tweet_network_data(tweet_id)
        
        if tweet_data.empty:
            print("❌ Tweet not found in database")
            return None
        
        if user_data.empty:
            print("❌ No user data found for this tweet")
            return None
        
        print(f"✅ Retrieved data: {len(user_data)} users, {len(interactions_data)} interactions")
        
        # Analyze coordination patterns
        results = model.analyze_tweet_coordination(tweet_id, user_data, interactions_data)
        
        return results
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None

def main():
    """Main function for interactive usage."""
    print("🌐 COORDINATED ACCOUNT NETWORK MODEL")
    print("=" * 50)
    print("Detects coordinated behavior using network analysis")
    print("Output: Coordination score from 0-1 (higher = more coordinated)")
    print()
    
    while True:
        try:
            # Get tweet ID from user
            tweet_id = input("Enter tweet ID (or 'quit' to exit): ").strip()
            
            if tweet_id.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not tweet_id:
                print("❌ Please enter a valid tweet ID")
                continue
            
            # Analyze the tweet
            results = analyze_tweet_by_id(tweet_id)
            
            if results:
                print("\n📊 COORDINATION ANALYSIS RESULTS")
                print("=" * 40)
                print(f"🎯 Tweet ID: {tweet_id}")
                print(f"📊 Coordination Score: {results['coordination_score']:.3f}")
                print(f"🔍 Coordination Level: {results['coordination_level']}")
                
                if results['cluster_id'] is not None:
                    print(f"🔗 Cluster ID: {results['cluster_id']}")
                
                print(f"\n📈 NETWORK METRICS:")
                metrics = results['network_metrics']
                print(f"   • Nodes: {metrics.get('node_count', 0)}")
                print(f"   • Edges: {metrics.get('edge_count', 0)}")
                print(f"   • Density: {metrics.get('density', 0):.3f}")
                print(f"   • Clustering Coefficient: {metrics.get('clustering_coefficient', 0):.3f}")
                print(f"   • Modularity: {metrics.get('modularity', 0):.3f}")
                
                if results['suspicious_patterns']:
                    print(f"\n🚨 SUSPICIOUS PATTERNS DETECTED:")
                    for pattern in results['suspicious_patterns']:
                        print(f"   • {pattern}")
                else:
                    print(f"\n✅ No suspicious patterns detected")
                
                # Provide interpretation
                score = results['coordination_score']
                if score >= 0.8:
                    interpretation = "VERY HIGH - Strong evidence of coordinated behavior"
                elif score >= 0.6:
                    interpretation = "HIGH - Likely coordinated behavior detected"
                elif score >= 0.4:
                    interpretation = "MODERATE - Some coordination patterns present"
                elif score >= 0.2:
                    interpretation = "LOW - Minimal coordination indicators"
                else:
                    interpretation = "NONE - No coordination patterns detected"
                
                print(f"\n💡 INTERPRETATION: {interpretation}")
            
            print("\n" + "=" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("Please try again or contact support.")

if __name__ == "__main__":
    main()
