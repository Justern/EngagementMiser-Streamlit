#!/usr/bin/env python3
"""
Test the scoring function with actual data
"""

def calculate_simple_reply_bait_score(tweet_data: dict) -> float:
    """Calculate a simple reply-bait score based on tweet characteristics."""
    if not tweet_data:
        return 0.0
    
    try:
        text = tweet_data['text'].lower()
        is_reply = tweet_data['in_reply_to_user_id'] is not None
        reply_count = tweet_data['reply_count']
        
        print(f"Analyzing text: {text[:100]}...")
        print(f"Is reply: {is_reply}")
        print(f"Reply count: {reply_count}")
        
        # Reply-bait indicators
        reply_bait_phrases = [
            'what do you think?', 'thoughts?', 'agree?', 'disagree?',
            'your opinion?', 'what\'s your take?', 'how about you?',
            'anyone else?', 'am i right?', 'am i wrong?', 'thoughts?',
            'agree or disagree?', 'what say you?', 'your thoughts?',
            'anyone?', 'thoughts on this?', 'what do you think?',
            'agree?', 'disagree?', 'thoughts?', 'anyone else?'
        ]
        
        # Count reply-bait phrases
        reply_bait_count = sum(1 for phrase in reply_bait_phrases if phrase in text)
        print(f"Reply-bait phrases found: {reply_bait_count}")
        
        # Question marks (indicate seeking engagement)
        question_marks = text.count('?')
        print(f"Question marks: {question_marks}")
        
        # Calculate score
        phrase_score = min(reply_bait_count / 3, 1.0)  # Normalize to 0-1
        question_score = min(question_marks / 5, 1.0)  # Normalize to 0-1
        
        print(f"Phrase score: {phrase_score}")
        print(f"Question score: {question_score}")
        
        # Combine scores
        final_score = (phrase_score * 0.6) + (question_score * 0.4)
        print(f"Final score: {final_score}")
        
        return max(0.0, min(1.0, final_score))
        
    except Exception as e:
        print(f"Error in scoring: {e}")
        return 0.0

# Test with the data we retrieved
test_data = {
    'text': 'LOVED answering your questions at the pop-up wig shop. I have the most incredible fans and it was amazing to meet you all in such a special setting. @youtubemusic https://t.co/7RGXBbo6FL',
    'author_id': '457554412',
    'in_reply_to_user_id': None,
    'conversation_id': '1233064764357726209',
    'like_count': 1227,
    'retweet_count': 95,
    'reply_count': 45,
    'quote_count': 4
}

print("Testing scoring function:")
print("=" * 50)
score = calculate_simple_reply_bait_score(test_data)
print(f"\nFinal result: {score}")
