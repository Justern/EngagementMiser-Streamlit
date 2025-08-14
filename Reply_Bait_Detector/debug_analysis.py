#!/usr/bin/env python3
"""
Debug script to investigate Reply-Bait Detector scoring issues
"""

from reply_bait_detector import ReplyBaitDetector
from data_connector import TwitterDataConnector
import pandas as pd
import pyodbc

def debug_conversation_analysis():
    """Debug the conversation analysis to see why scores are zero."""
    
    print("DEBUGGING REPLY-BAIT DETECTOR SCORING")
    print("=" * 50)
    
    # Initialize
    detector = ReplyBaitDetector(similarity_threshold=0.5, sentiment_threshold=0.2)  # Lower thresholds for testing
    connector = TwitterDataConnector('localhost', 'EngagementMiser')
    
    if not connector.test_connection():
        print("❌ Database connection failed")
        return
    
    print("✅ Connected to database")
    
    # Get sample conversations
    print("\n🔍 Getting sample conversations...")
    sample_data = connector.get_sample_conversations(20)
    
    if sample_data.empty:
        print("❌ No sample data retrieved")
        return
    
    print(f"📊 Retrieved {len(sample_data)} tweets from {sample_data['conversation_id'].nunique()} conversations")
    
    # Find conversations with multiple tweets
    conv_sizes = sample_data.groupby('conversation_id').size()
    multi_tweet_convs = conv_sizes[conv_sizes > 1].index.tolist()
    
    print(f"🎯 Found {len(multi_tweet_convs)} conversations with multiple tweets")
    
    if not multi_tweet_convs:
        print("❌ No conversations with multiple tweets found")
        return
    
    # Test the first multi-tweet conversation
    test_conv_id = multi_tweet_convs[0]
    print(f"\n🧪 Testing conversation: {test_conv_id}")
    
    conv_data = connector.get_conversation_thread(test_conv_id)
    print(f"📝 Conversation has {len(conv_data)} tweets")
    
    # Show conversation structure
    print("\n📋 Conversation structure:")
    for i, row in conv_data.iterrows():
        author = row['author_id']
        is_main = row['in_reply_to_user_id'] is None
        text_preview = row['text'][:50] + "..." if len(row['text']) > 50 else row['text']
        print(f"  {i}: Author {author} {'(MAIN)' if is_main else '(REPLY)'}: {text_preview}")
    
    # Analyze the conversation
    print(f"\n🔬 Running full analysis...")
    analysis = detector.analyze_conversation_thread(conv_data)
    
    if 'error' in analysis:
        print(f"❌ Analysis error: {analysis['error']}")
        return
    
    print(f"📊 Analysis results:")
    print(f"  Reply-bait score: {analysis['reply_bait_score']:.3f}")
    print(f"  Is reply-bait: {analysis['is_reply_bait']}")
    print(f"  Total replies: {analysis['total_replies']}")
    print(f"  Own replies: {analysis['own_replies_count']}")
    
    # Check repetitive analysis
    rep_analysis = analysis['repetitive_analysis']
    print(f"\n🔄 Repetitive analysis:")
    print(f"  Is repetitive: {rep_analysis['is_repetitive']}")
    print(f"  Overall similarity: {rep_analysis['overall_similarity']:.3f}")
    print(f"  Repetitive pairs: {len(rep_analysis.get('repetitive_pairs', []))}")
    
    # Check sentiment analysis
    sent_analysis = analysis['sentiment_analysis']
    print(f"\n😊 Sentiment analysis:")
    print(f"  Main sentiment: {sent_analysis['main_sentiment']:.3f}")
    print(f"  Avg reply sentiment: {sent_analysis['avg_reply_sentiment']:.3f}")
    print(f"  Has inversion: {sent_analysis['has_inversion']}")
    print(f"  Inversion score: {sent_analysis['inversion_score']:.3f}")
    
    # Test the simple analyze_tweet method
    print(f"\n🎯 Testing analyze_tweet method...")
    score = detector.analyze_tweet(test_conv_id, connector)
    print(f"  Simple score: {score:.3f}")
    
    return analysis

def find_conversations_with_own_replies():
    """Find conversations where authors reply to their own posts."""
    
    print("\n" + "=" * 50)
    print("FINDING CONVERSATIONS WITH OWN REPLIES")
    print("=" * 50)
    
    connector = TwitterDataConnector('localhost', 'EngagementMiser')
    
    # Get conversations with own replies
    data = connector.get_conversations_with_own_replies(min_own_replies=2, limit=5)
    
    if data.empty:
        print("❌ No conversations with own replies found")
        return
    
    print(f"✅ Found {len(data)} tweets from {data['conversation_id'].nunique()} conversations")
    
    # Show conversation details
    for conv_id in data['conversation_id'].unique()[:3]:
        conv_data = data[data['conversation_id'] == conv_id]
        print(f"\n📝 Conversation {conv_id}:")
        print(f"  Total tweets: {len(conv_data)}")
        print(f"  Authors: {conv_data['author_id'].unique()}")
        
        # Show first few tweets
        for i, row in conv_data.head(3).iterrows():
            author = row['author_id']
            is_main = row['in_reply_to_user_id'] is None
            text_preview = row['text'][:60] + "..." if len(row['text']) > 60 else row['text']
            print(f"    {i}: Author {author} {'(MAIN)' if is_main else '(REPLY)'}: {text_preview}")

def investigate_specific_conversation():
    """Investigate a specific conversation to understand the data structure."""
    
    print("\n" + "=" * 50)
    print("INVESTIGATING SPECIFIC CONVERSATION")
    print("=" * 50)
    
    connector = TwitterDataConnector('localhost', 'EngagementMiser')
    
    # Test with a conversation that has own replies
    conv_id = "663578384291745797"
    print(f"🔍 Investigating conversation: {conv_id}")
    
    # Get the full conversation thread
    conv_data = connector.get_conversation_thread(conv_id)
    print(f"📝 Retrieved {len(conv_data)} tweets")
    
    # Show all tweets in the conversation
    print(f"\n📋 All tweets in conversation:")
    for i, row in conv_data.iterrows():
        tweet_id = row['tweet_id']
        author = row['author_id']
        in_reply_to = row['in_reply_to_user_id']
        text_preview = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
        print(f"  {i}: Tweet {tweet_id} | Author {author} | Reply to: {in_reply_to}")
        print(f"     Text: {text_preview}")
        print()
    
    # Check if we can find the main post by looking for tweets with no in_reply_to_user_id
    main_posts = conv_data[conv_data['in_reply_to_user_id'].isna()]
    print(f"🎯 Main posts found: {len(main_posts)}")
    
    if not main_posts.empty:
        print("Main post details:")
        for i, row in main_posts.iterrows():
            print(f"  Tweet ID: {row['tweet_id']}")
            print(f"  Author: {row['author_id']}")
            print(f"  Text: {row['text'][:100]}...")
    else:
        print("❌ No main post found - all tweets are replies")
        
        # Try to find the original conversation starter
        print("\n🔍 Looking for conversation starter...")
        # Check if there are tweets that this conversation is replying to
        for i, row in conv_data.iterrows():
            if row['in_reply_to_user_id']:
                print(f"  Tweet {row['tweet_id']} replies to user {row['in_reply_to_user_id']}")
                # Try to get the tweet this is replying to
                try:
                    reply_to_tweet = connector.get_conversation_thread(str(row['in_reply_to_user_id']))
                    if not reply_to_tweet.empty:
                        print(f"    Found reply-to tweet: {reply_to_tweet.iloc[0]['text'][:50]}...")
                except:
                    print(f"    Could not retrieve reply-to tweet")

def search_for_self_reply_conversations():
    """Search for conversations where authors reply to their own main posts."""
    
    print("\n" + "=" * 50)
    print("SEARCHING FOR SELF-REPLY CONVERSATIONS")
    print("=" * 50)
    
    connector = TwitterDataConnector('localhost', 'EngagementMiser')
    
    # Query to find conversations where authors reply to their own posts
    query = """
    WITH MainPosts AS (
        SELECT 
            conversation_id,
            author_id,
            tweet_id as main_tweet_id,
            text as main_text
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M]
        WHERE in_reply_to_user_id IS NULL
    ),
    SelfReplies AS (
        SELECT 
            t.conversation_id,
            t.author_id,
            t.tweet_id,
            t.text,
            t.in_reply_to_user_id,
            mp.main_tweet_id,
            mp.main_text
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
        INNER JOIN MainPosts mp ON t.conversation_id = mp.conversation_id
        WHERE t.in_reply_to_user_id = t.author_id  -- Author replying to themselves
        AND t.in_reply_to_user_id IS NOT NULL
    )
    SELECT TOP 10
        conversation_id,
        author_id,
        COUNT(*) as self_reply_count,
        MIN(text) as sample_reply
    FROM SelfReplies
    GROUP BY conversation_id, author_id
    HAVING COUNT(*) >= 2
    ORDER BY self_reply_count DESC
    """
    
    try:
        with pyodbc.connect(connector.connection_string) as conn:
            df = pd.read_sql(query, conn)
            
        if df.empty:
            print("❌ No self-reply conversations found")
            return
        
        print(f"✅ Found {len(df)} conversations with self-replies")
        
        for i, row in df.iterrows():
            conv_id = row['conversation_id']
            author = row['author_id']
            count = row['self_reply_count']
            sample = row['sample_reply']
            
            print(f"\n📝 Conversation {conv_id}:")
            print(f"  Author: {author}")
            print(f"  Self-replies: {count}")
            print(f"  Sample reply: {sample[:80]}...")
            
            # Test this conversation
            print(f"  🧪 Testing analysis...")
            conv_data = connector.get_conversation_thread(conv_id)
            
            if len(conv_data) > 1:
                detector = ReplyBaitDetector(similarity_threshold=0.5, sentiment_threshold=0.2)
                analysis = detector.analyze_conversation_thread(conv_data)
                
                if 'error' not in analysis:
                    score = analysis['reply_bait_score']
                    own_replies = analysis['own_replies_count']
                    print(f"    Reply-bait score: {score:.3f}")
                    print(f"    Own replies detected: {own_replies}")
                else:
                    print(f"    Analysis error: {analysis['error']}")
            else:
                print(f"    Conversation has only {len(conv_data)} tweet(s)")
                
    except Exception as e:
        print(f"❌ Error searching for self-reply conversations: {str(e)}")

if __name__ == "__main__":
    debug_conversation_analysis()
    find_conversations_with_own_replies()
    investigate_specific_conversation()
    search_for_self_reply_conversations()
