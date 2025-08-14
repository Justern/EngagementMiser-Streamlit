#!/usr/bin/env python3
"""
Coordinated Account Network Model - Training Script
==================================================

This script trains the Coordinated Account Network Model using comprehensive
data from the full Tweets table for better model performance.
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from coordinated_account_network_model import CoordinatedAccountNetworkModel
from data_connector import NetworkDataConnector

def train_model_with_full_data(training_limit=50000):
    """
    Train the model using comprehensive data from the full Tweets table.
    
    Args:
        training_limit (int): Maximum number of records to use for training
        
    Returns:
        bool: True if training successful, False otherwise
    """
    print("🚀 TRAINING COORDINATED ACCOUNT NETWORK MODEL")
    print("=" * 60)
    print("Using full Tweets table for comprehensive training...")
    print(f"Training limit: {training_limit:,} records")
    print()
    
    try:
        # Initialize components
        print("🔧 Initializing model and database connector...")
        model = CoordinatedAccountNetworkModel()
        connector = NetworkDataConnector(
            server='localhost',
            database='EngagementMiser'
        )
        
        # Test database connection
        if not connector.test_connection():
            print("❌ Database connection failed!")
            return False
        
        print("✅ Database connection successful!")
        
        # Get comprehensive training data
        print("\n📥 Retrieving comprehensive training data...")
        user_data, interactions_data = connector.get_training_data_from_full_tweets(training_limit)
        
        if user_data.empty or interactions_data.empty:
            print("❌ Insufficient training data retrieved")
            return False
        
        print(f"✅ Retrieved {len(user_data):,} users and {len(interactions_data):,} interactions")
        
        # Preprocess training data
        print("\n🔧 Preprocessing training data...")
        processed_users = model.preprocess_network_data(user_data)
        print(f"✅ Preprocessed {len(processed_users):,} user records")
        
        # Extract comprehensive features
        print("🔧 Extracting network features...")
        features = model.extract_network_features(processed_users)
        print(f"✅ Extracted {len(features.columns)} features")
        
        # Build comprehensive interaction graph
        print("🔧 Building comprehensive interaction graph...")
        G = model.build_interaction_graph(interactions_data)
        print(f"✅ Built graph with {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges")
        
        # Calculate comprehensive network metrics
        print("🔧 Calculating network metrics...")
        metrics = model.calculate_network_metrics(G)
        print(f"✅ Network density: {metrics['density']:.3f}")
        print(f"✅ Clustering coefficient: {metrics['clustering_coefficient']:.3f}")
        print(f"✅ Modularity: {metrics['modularity']:.3f}")
        
        # Train clustering model on comprehensive data
        print("\n🔧 Training clustering model on comprehensive data...")
        cluster_labels, clustering_model = model.detect_coordinated_clusters(features, method='dbscan')
        unique_clusters = set(cluster_labels)
        print(f"✅ Detected {len(unique_clusters)} clusters")
        
        # Mark model as trained
        model.is_trained = True
        print("✅ Model marked as trained")
        
        # Save the trained model
        model_filename = f"coordinated_network_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        print(f"\n💾 Saving trained model to: {model_filename}")
        model.save_model(model_filename)
        
        # Test the trained model
        print("\n🧪 Testing trained model...")
        if not user_data.empty:
            sample_user = user_data.iloc[0]
            user_id = sample_user['user_id']
            
            print(f"🎯 Testing coordination analysis on user: {sample_user['username']}")
            results = model.analyze_user_coordination(user_id, user_data, interactions_data)
            
            if results:
                print(f"✅ Test analysis successful:")
                print(f"   • Score: {results['coordination_score']:.3f}")
                print(f"   • Level: {results['coordination_level']}")
                print(f"   • Cluster: {results['cluster_id']}")
            else:
                print("❌ Test analysis failed")
        
        print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Model saved as: {model_filename}")
        print("You can now use: python simple_usage.py")
        print("\n💡 For quick analysis, the model will use Tweets_Sample_4M")
        print("💡 For comprehensive analysis, the model will use the full Tweets table")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        return False

def train_model_with_sample_data(training_limit=10000):
    """
    Train the model using the smaller Tweets_Sample_4M table for faster training.
    
    Args:
        training_limit (int): Maximum number of records to use for training
        
    Returns:
        bool: True if training successful, False otherwise
    """
    print("🚀 TRAINING COORDINATED ACCOUNT NETWORK MODEL (SAMPLE DATA)")
    print("=" * 60)
    print("Using Tweets_Sample_4M table for faster training...")
    print(f"Training limit: {training_limit:,} records")
    print()
    
    try:
        # Initialize components
        print("🔧 Initializing model and database connector...")
        model = CoordinatedAccountNetworkModel()
        connector = NetworkDataConnector(
            server='localhost',
            database='EngagementMiser'
        )
        
        # Test database connection
        if not connector.test_connection():
            print("❌ Database connection failed!")
            return False
        
        print("✅ Database connection successful!")
        
        # Get sample training data
        print("\n📥 Retrieving sample training data...")
        user_data, interactions_data = connector.get_optimized_sample_data(training_limit)
        
        if user_data.empty or interactions_data.empty:
            print("❌ Insufficient training data retrieved")
            return False
        
        print(f"✅ Retrieved {len(user_data):,} users and {len(interactions_data):,} interactions")
        
        # Continue with training process...
        # (Same as full data training but with sample data)
        processed_users = model.preprocess_network_data(user_data)
        features = model.extract_network_features(processed_users)
        G = model.build_interaction_graph(interactions_data)
        metrics = model.calculate_network_metrics(G)
        cluster_labels, clustering_model = model.detect_coordinated_clusters(features, method='dbscan')
        
        model.is_trained = True
        
        # Save model
        model_filename = f"coordinated_network_model_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        model.save_model(model_filename)
        
        print(f"\n🎉 SAMPLE DATA TRAINING COMPLETED!")
        print(f"Model saved as: {model_filename}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Sample data training failed with error: {e}")
        return False

def main():
    """Main training function with options."""
    print("🌐 COORDINATED ACCOUNT NETWORK MODEL TRAINING")
    print("=" * 70)
    print("Choose training approach:")
    print("1. Comprehensive training (full Tweets table) - Slower but better")
    print("2. Quick training (Tweets_Sample_4M) - Faster but limited data")
    print("3. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == '1':
                print("\n🚀 Starting comprehensive training...")
                success = train_model_with_full_data(50000)
                if success:
                    print("\n✅ Comprehensive training completed successfully!")
                else:
                    print("\n❌ Comprehensive training failed!")
                break
                
            elif choice == '2':
                print("\n🚀 Starting quick training...")
                success = train_model_with_sample_data(10000)
                if success:
                    print("\n✅ Quick training completed successfully!")
                else:
                    print("\n❌ Quick training failed!")
                break
                
            elif choice == '3':
                print("👋 Exiting training...")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Training interrupted by user.")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()
