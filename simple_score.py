#!/usr/bin/env python3
"""
Simple Score Script for ECS Models - Cloud Compatible
===================================================

Takes tweet text and model name as input and returns a single 0-1 score.
Usage: python simple_score.py --text <text_file> --model <model_name>
Output: Single line with score (0.0 to 1.0)
"""

import sys
import argparse
import os

def calculate_authority_signal_score(text: str) -> float:
    """Calculate authority signal manipulation score."""
    try:
        text_lower = text.lower()
        
        # Authority manipulation indicators
        authority_phrases = [
            'expert', 'professional', 'doctor', 'scientist', 'researcher',
            'study shows', 'research proves', 'experts agree', 'authority',
            'scientifically proven', 'clinically tested', 'doctor recommended',
            'according to science', 'research indicates', 'studies confirm',
            'medical evidence', 'scientific evidence', 'clinical evidence',
            'expert opinion', 'professional opinion', 'authority figure'
        ]
        
        # Count authority phrases
        authority_count = sum(1 for phrase in authority_phrases if phrase in text_lower)
        
        # Calculate score based on authority language density
        score = min(authority_count / 3, 1.0)  # Normalize to 0-1
        
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        return 0.0

def calculate_clickbait_score(text: str) -> float:
    """Calculate clickbait headline score."""
    try:
        text_lower = text.lower()
        
        # Clickbait indicators
        clickbait_phrases = [
            'you won\'t believe', 'shocking', 'amazing', 'incredible',
            'this will blow your mind', 'what happened next', 'the truth about',
            'they don\'t want you to know', 'secret', 'exposed', 'revealed',
            'breaking', 'urgent', 'warning', 'alert', 'critical',
            'number one reason', 'top secret', 'hidden', 'forbidden'
        ]
        
        # Count clickbait phrases
        clickbait_count = sum(1 for phrase in clickbait_phrases if phrase in text_lower)
        
        # Calculate score
        score = min(clickbait_count / 2, 1.0)
        
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        return 0.0

def calculate_content_recycling_score(text: str) -> float:
    """Calculate content recycling score."""
    try:
        text_lower = text.lower()
        
        # Content recycling indicators
        recycling_phrases = [
            'repost', 'reposting', 'repost this', 'share this',
            'viral', 'going viral', 'trending', 'trending now',
            'everyone is talking about', 'everyone needs to see',
            'spread the word', 'pass it on', 'forward this',
            'retweet this', 'like and share', 'comment and share'
        ]
        
        # Count recycling phrases
        recycling_count = sum(1 for phrase in recycling_phrases if phrase in text_lower)
        
        # Calculate score
        score = min(recycling_count / 2, 1.0)
        
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        return 0.0

def calculate_coordinated_network_score(text: str) -> float:
    """Calculate coordinated account network score."""
    try:
        text_lower = text.lower()
        
        # Coordinated network indicators
        network_phrases = [
            'bot', 'bots', 'automated', 'script', 'scripted',
            'coordinated', 'network', 'campaign', 'operation',
            'mass', 'bulk', 'flood', 'spam', 'spamming',
            'trending', 'trend', 'hashtag', 'hashtags'
        ]
        
        # Count network phrases
        network_count = sum(1 for phrase in network_phrases if phrase in text_lower)
        
        # Calculate score
        score = min(network_count / 2, 1.0)
        
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        return 0.0

def calculate_emotive_manipulation_score(text: str) -> float:
    """Calculate emotive manipulation score."""
    try:
        text_lower = text.lower()
        
        # Emotive manipulation indicators
        emotive_phrases = [
            'outrage', 'outraged', 'angry', 'furious', 'livid',
            'shocked', 'shocking', 'disgusting', 'horrible', 'terrible',
            'amazing', 'incredible', 'wonderful', 'fantastic', 'amazing',
            'heartbreaking', 'devastating', 'tragic', 'sad', 'depressing',
            'excited', 'thrilled', 'ecstatic', 'overjoyed', 'elated'
        ]
        
        # Count emotive phrases
        emotive_count = sum(1 for phrase in emotive_phrases if phrase in text_lower)
        
        # Calculate score
        score = min(emotive_count / 3, 1.0)
        
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        return 0.0

def calculate_engagement_mismatch_score(text: str) -> float:
    """Calculate engagement mismatch score."""
    try:
        text_lower = text.lower()
        
        # Engagement mismatch indicators
        mismatch_phrases = [
            'like this', 'like if you agree', 'like for more',
            'retweet this', 'retweet if you agree', 'retweet for more',
            'comment below', 'comment your thoughts', 'comment your opinion',
            'share this', 'share if you agree', 'share for awareness',
            'follow me', 'follow for more', 'follow for updates'
        ]
        
        # Count mismatch phrases
        mismatch_count = sum(1 for phrase in mismatch_phrases if phrase in text_lower)
        
        # Calculate score
        score = min(mismatch_count / 2, 1.0)
        
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        return 0.0

def calculate_generic_comment_score(text: str) -> float:
    """Calculate generic comment score."""
    try:
        text_lower = text.lower()
        
        # Generic comment indicators
        generic_phrases = [
            'nice', 'good', 'great', 'awesome', 'cool',
            'interesting', 'wow', 'omg', 'lol', 'haha',
            'thanks', 'thank you', 'appreciate it', 'love this',
            'agree', 'disagree', 'true', 'false', 'yes', 'no'
        ]
        
        # Count generic phrases
        generic_count = sum(1 for phrase in generic_phrases if phrase in text_lower)
        
        # Calculate score
        score = min(generic_count / 3, 1.0)
        
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        return 0.0

def calculate_hyperbole_falsehood_score(text: str) -> float:
    """Calculate hyperbole and falsehood score."""
    try:
        text_lower = text.lower()
        
        # Hyperbole and falsehood indicators
        hyperbole_phrases = [
            'always', 'never', 'everyone', 'nobody', 'every single',
            '100%', 'guaranteed', 'proven', 'definitely', 'absolutely',
            'worst ever', 'best ever', 'most', 'least', 'completely',
            'totally', 'literally', 'actually', 'really', 'very'
        ]
        
        # Count hyperbole phrases
        hyperbole_count = sum(1 for phrase in hyperbole_phrases if phrase in text_lower)
        
        # Calculate score
        score = min(hyperbole_count / 3, 1.0)
        
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        return 0.0

def calculate_rapid_engagement_score(text: str) -> float:
    """Calculate rapid engagement spike score."""
    try:
        text_lower = text.lower()
        
        # Rapid engagement indicators
        rapid_phrases = [
            'trending', 'trending now', 'going viral', 'viral',
            'exploding', 'blowing up', 'skyrocketing', 'soaring',
            'breaking', 'breaking news', 'just in', 'update',
            'developing', 'developing story', 'latest', 'newest'
        ]
        
        # Count rapid phrases
        rapid_count = sum(1 for phrase in rapid_phrases if phrase in text_lower)
        
        # Calculate score
        score = min(rapid_count / 2, 1.0)
        
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        return 0.0

def calculate_reply_bait_score(text: str) -> float:
    """Calculate reply bait score."""
    try:
        text_lower = text.lower()
        
        # Reply bait indicators
        reply_bait_phrases = [
            'what do you think', 'your thoughts', 'your opinion',
            'agree or disagree', 'comment below', 'comment your thoughts',
            'what\'s your take', 'how do you feel', 'do you agree',
            'share your experience', 'tell me', 'let me know',
            'what about you', 'your turn', 'your say'
        ]
        
        # Count reply bait phrases
        reply_bait_count = sum(1 for phrase in reply_bait_phrases if phrase in text_lower)
        
        # Calculate score
        score = min(reply_bait_count / 2, 1.0)
        
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        return 0.0

def main():
    """Main function to get score for a specific model and text."""
    parser = argparse.ArgumentParser(description='Calculate ECS model score for text')
    parser.add_argument('--text', required=True, help='Path to text file')
    parser.add_argument('--model', required=True, help='Model name to use')
    
    args = parser.parse_args()
    
    try:
        # Read text from file
        with open(args.text, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        if not text:
            print("0.0")
            return
        
        # Select model and calculate score
        model_name = args.model.lower()
        
        if 'authority' in model_name:
            score = calculate_authority_signal_score(text)
        elif 'clickbait' in model_name:
            score = calculate_clickbait_score(text)
        elif 'content_recycling' in model_name or 'content_recycling' in model_name:
            score = calculate_content_recycling_score(text)
        elif 'coordinated' in model_name:
            score = calculate_coordinated_network_score(text)
        elif 'emotive' in model_name:
            score = calculate_emotive_manipulation_score(text)
        elif 'engagement_mismatch' in model_name:
            score = calculate_engagement_mismatch_score(text)
        elif 'generic_comment' in model_name:
            score = calculate_generic_comment_score(text)
        elif 'hyperbole' in model_name or 'falsehood' in model_name:
            score = calculate_hyperbole_falsehood_score(text)
        elif 'rapid_engagement' in model_name:
            score = calculate_rapid_engagement_score(text)
        elif 'reply_bait' in model_name:
            score = calculate_reply_bait_score(text)
        else:
            # Default to generic score if model not recognized
            score = calculate_generic_comment_score(text)
        
        # Output the score
        print(f"{score:.3f}")
        
    except Exception as e:
        # On any error, return 0.0
        print("0.0")

if __name__ == "__main__":
    main()
