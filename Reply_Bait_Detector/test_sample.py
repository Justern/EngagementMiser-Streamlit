#!/usr/bin/env python3
"""
Test script for sample conversations functionality.
"""

from data_connector import TwitterDataConnector

def test_sample_conversations():
    """Test the sample conversations functionality."""
    try:
        # Initialize connector
        connector = TwitterDataConnector('localhost', 'EngagementMiser')
        
        # Test connection
        if not connector.test_connection():
            print("❌ Database connection failed")
            return
        
        print("✅ Database connection successful")
        
        # Test sample conversations
        print("Testing sample conversations...")
        data = connector.get_sample_conversations(5)
        
        if data.empty:
            print("❌ No sample data retrieved")
        else:
            print(f"✅ Retrieved {len(data)} tweets from {data['conversation_id'].nunique()} conversations")
            print(f"Sample conversation IDs: {data['conversation_id'].unique()[:3].tolist()}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_sample_conversations()
