"""
Example Usage of Reply-Bait Detector

This script demonstrates how to use the Reply-Bait Detector model
to analyze Twitter conversations for reply-baiting behavior.

Author: DS Capstone Project
Date: 2025
"""

import pandas as pd
import numpy as np
from reply_bait_detector import ReplyBaitDetector
from data_connector import TwitterDataConnector
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_single_conversation(conversation_id: str, connector: TwitterDataConnector, 
                               detector: ReplyBaitDetector) -> dict:
    """
    Analyze a single conversation for reply-baiting patterns.
    
    Args:
        conversation_id: The conversation ID to analyze
        connector: Database connector instance
        detector: Reply-Bait Detector instance
        
    Returns:
        Analysis results dictionary
    """
    logger.info(f"Analyzing conversation: {conversation_id}")
    
    # Get conversation data
    conversation_data = connector.get_conversation_thread(conversation_id)
    
    if conversation_data.empty:
        logger.warning(f"No data found for conversation {conversation_id}")
        return {'error': 'No conversation data found'}
    
    # Analyze the conversation
    analysis_results = detector.analyze_conversation_thread(conversation_data)
    
    # Generate report
    report = detector.generate_report(analysis_results)
    print(report)
    
    return analysis_results

def analyze_user_patterns(user_id: str, connector: TwitterDataConnector, 
                         detector: ReplyBaitDetector, max_conversations: int = 20) -> dict:
    """
    Analyze reply-baiting patterns for a specific user.
    
    Args:
        user_id: The user ID to analyze
        connector: Database connector instance
        detector: Reply-Bait Detector instance
        max_conversations: Maximum number of conversations to analyze
        
    Returns:
        User analysis results dictionary
    """
    logger.info(f"Analyzing user patterns for user: {user_id}")
    
    # Get user's conversations
    user_data = connector.get_user_conversations(user_id, limit=max_conversations)
    
    if user_data.empty:
        logger.warning(f"No conversation data found for user {user_id}")
        return {'error': 'No user conversation data found'}
    
    # Analyze user patterns
    user_analysis = detector.analyze_user_patterns(user_data, min_conversations=3)
    
    # Generate report
    report = detector.generate_report(user_analysis)
    print(report)
    
    return user_analysis

def analyze_conversations_with_own_replies(connector: TwitterDataConnector, 
                                          detector: ReplyBaitDetector, 
                                          min_own_replies: int = 2, 
                                          max_conversations: int = 10) -> list:
    """
    Analyze conversations that have own replies for reply-baiting patterns.
    
    Args:
        connector: Database connector instance
        detector: Reply-Bait Detector instance
        min_own_replies: Minimum number of own replies required
        max_conversations: Maximum number of conversations to analyze
        
    Returns:
        List of analysis results
    """
    logger.info(f"Analyzing conversations with at least {min_own_replies} own replies")
    
    # Get conversations with own replies
    conversations_data = connector.get_conversations_with_own_replies(
        min_own_replies=min_own_replies, 
        limit=max_conversations
    )
    
    if conversations_data.empty:
        logger.warning("No conversations with own replies found")
        return []
    
    # Group by conversation and analyze each
    conversation_groups = conversations_data.groupby('conversation_id')
    analysis_results = []
    
    for conv_id, conv_data in conversation_groups:
        logger.info(f"Analyzing conversation {conv_id} with {len(conv_data)} tweets")
        
        # Analyze the conversation
        conv_analysis = detector.analyze_conversation_thread(conv_data)
        
        if 'error' not in conv_analysis:
            analysis_results.append(conv_analysis)
            
            # Print summary
            print(f"\nConversation {conv_id}:")
            print(f"  Reply-Bait Score: {conv_analysis['reply_bait_score']:.3f}")
            print(f"  Is Reply-Bait: {conv_analysis['is_reply_bait']}")
            print(f"  Own Replies: {conv_analysis['own_replies_count']}")
            print(f"  Total Replies: {conv_analysis['total_replies']}")
    
    return analysis_results

def run_sample_analysis(connector: TwitterDataConnector, detector: ReplyBaitDetector):
    """
    Run a sample analysis on randomly selected conversations.
    
    Args:
        connector: Database connector instance
        detector: Reply-Bait Detector instance
    """
    logger.info("Running sample analysis on random conversations")
    
    # Get sample conversations
    sample_data = connector.get_sample_conversations(sample_size=20)
    
    if sample_data.empty:
        logger.warning("No sample data retrieved")
        return
    
    # Group by conversation
    conversation_groups = sample_data.groupby('conversation_id')
    
    print(f"\nAnalyzing {len(conversation_groups)} sample conversations...")
    print("=" * 60)
    
    reply_bait_scores = []
    conversations_with_own_replies = 0
    
    for conv_id, conv_data in conversation_groups:
        # Only analyze conversations with multiple tweets
        if len(conv_data) > 1:
            conv_analysis = detector.analyze_conversation_thread(conv_data)
            
            if 'error' not in conv_analysis:
                reply_bait_scores.append(conv_analysis['reply_bait_score'])
                
                if conv_analysis['own_replies_count'] > 0:
                    conversations_with_own_replies += 1
                
                # Print brief summary for each conversation
                print(f"Conversation {conv_id[:8]}... | "
                      f"Score: {conv_analysis['reply_bait_score']:.3f} | "
                      f"Own Replies: {conv_analysis['own_replies_count']} | "
                      f"Total: {conv_analysis['total_replies']}")
    
    if reply_bait_scores:
        print("\n" + "=" * 60)
        print("SAMPLE ANALYSIS SUMMARY:")
        print(f"Conversations Analyzed: {len(reply_bait_scores)}")
        print(f"Average Reply-Bait Score: {np.mean(reply_bait_scores):.3f}")
        print(f"Conversations with Own Replies: {conversations_with_own_replies}")
        print(f"High Risk Conversations (Score > 0.6): {sum(1 for s in reply_bait_scores if s > 0.6)}")

def main():
    """
    Main function demonstrating the Reply-Bait Detector usage.
    """
    print("Reply-Bait Detector - Example Usage")
    print("=" * 60)
    
    # Initialize the detector
    detector = ReplyBaitDetector(
        similarity_threshold=0.7,
        sentiment_threshold=0.3
    )
    
    print("✓ Reply-Bait Detector initialized")
    print(f"  Similarity threshold: {detector.similarity_threshold}")
    print(f"  Sentiment threshold: {detector.sentiment_threshold}")
    
    # Initialize the data connector
    # Modify these parameters based on your SQL Server setup
    connector = TwitterDataConnector(
        server="localhost",  # or your SQL Server instance name
        database="EngagementMiser"
    )
    
    print("\nTesting database connection...")
    
    # Test connection
    if not connector.test_connection():
        print("✗ Database connection failed!")
        print("Please check your connection parameters and ensure SQL Server is running.")
        print("\nExample connection parameters:")
        print("  - Server: 'localhost' or 'DESKTOP-XXXXX\\SQLEXPRESS'")
        print("  - Database: 'EngagementMiser'")
        print("  - Use Windows Authentication (trusted_connection=True)")
        return
    
    print("✓ Database connection successful!")
    
    # Menu for different analysis options
    while True:
        print("\n" + "=" * 60)
        print("REPLY-BAIT DETECTOR ANALYSIS MENU")
        print("=" * 60)
        print("1. Analyze a specific conversation")
        print("2. Analyze user patterns")
        print("3. Analyze conversations with own replies")
        print("4. Run sample analysis")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        try:
            if choice == '1':
                conv_id = input("Enter conversation ID: ").strip()
                if conv_id:
                    analyze_single_conversation(conv_id, connector, detector)
                else:
                    print("Invalid conversation ID")
                    
            elif choice == '2':
                user_id = input("Enter user ID: ").strip()
                if user_id:
                    max_conv = input("Maximum conversations to analyze (default 20): ").strip()
                    max_conv = int(max_conv) if max_conv.isdigit() else 20
                    analyze_user_patterns(user_id, connector, detector, max_conv)
                else:
                    print("Invalid user ID")
                    
            elif choice == '3':
                min_replies = input("Minimum own replies required (default 2): ").strip()
                min_replies = int(min_replies) if min_replies.isdigit() else 2
                max_conv = input("Maximum conversations to analyze (default 10): ").strip()
                max_conv = int(max_conv) if max_conv.isdigit() else 10
                analyze_conversations_with_own_replies(connector, detector, min_replies, max_conv)
                
            elif choice == '4':
                run_sample_analysis(connector, detector)
                
            elif choice == '5':
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please enter a number between 1-5.")
                
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            break
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            print(f"An error occurred: {str(e)}")
    
    print("\nThank you for using the Reply-Bait Detector!")


if __name__ == "__main__":
    main()
