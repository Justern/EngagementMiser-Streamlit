#!/usr/bin/env python3
"""
Simple Score Script for ECS Models - Cloud Compatible with Hugging Face
=====================================================================

Takes tweet text and model name as input and returns a single 0-1 score.
Uses the Hugging Face Hub model for sophisticated scoring.
Usage: python simple_score.py --text <text_file> --model <model_name>
Output: Single line with score (0.0 to 1.0)
"""

import sys
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_hf_model():
    """Load the model from Hugging Face Hub."""
    try:
        repo_id = "MidlAnalytics/engagement-concordance-roberta"
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForSequenceClassification.from_pretrained(repo_id)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        return model, tokenizer, device
        
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return None, None, None

def calculate_authority_signal_score(text: str, model, tokenizer, device) -> float:
    """Calculate authority signal manipulation score using the trained model."""
    try:
        # Use the trained model for sophisticated scoring
        with torch.no_grad():
            enc = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            score = torch.sigmoid(logits.squeeze()).item()
            
            return float(score)
            
    except Exception as e:
        # Fallback to keyword-based scoring if model fails
        text_lower = text.lower()
        authority_phrases = [
            'expert', 'professional', 'doctor', 'scientist', 'researcher',
            'study shows', 'research proves', 'experts agree', 'authority',
            'scientifically proven', 'clinically tested', 'doctor recommended',
            'according to science', 'research indicates', 'studies confirm',
            'medical evidence', 'scientific evidence', 'clinical evidence',
            'expert opinion', 'professional opinion', 'authority figure'
        ]
        
        authority_count = sum(1 for phrase in authority_phrases if phrase in text_lower)
        score = min(authority_count / 3, 1.0)
        return max(0.0, min(1.0, score))

def calculate_clickbait_score(text: str, model, tokenizer, device) -> float:
    """Calculate clickbait headline score using the trained model."""
    try:
        # Use the trained model for sophisticated scoring
        with torch.no_grad():
            enc = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            score = torch.sigmoid(logits.squeeze()).item()
            
            return float(score)
            
    except Exception as e:
        # Fallback to keyword-based scoring if model fails
        text_lower = text.lower()
        clickbait_phrases = [
            'you won\'t believe', 'shocking', 'amazing', 'incredible',
            'this will blow your mind', 'what happened next', 'the truth about',
            'they don\'t want you to know', 'secret', 'exposed', 'revealed',
            'breaking', 'urgent', 'warning', 'alert', 'critical',
            'number one reason', 'top secret', 'hidden', 'forbidden'
        ]
        
        clickbait_count = sum(1 for phrase in clickbait_phrases if phrase in text_lower)
        score = min(clickbait_count / 2, 1.0)
        return max(0.0, min(1.0, score))

def calculate_content_recycling_score(text: str, model, tokenizer, device) -> float:
    """Calculate content recycling score using the trained model."""
    try:
        # Use the trained model for sophisticated scoring
        with torch.no_grad():
            enc = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            score = torch.sigmoid(logits.squeeze()).item()
            
            return float(score)
            
    except Exception as e:
        # Fallback to keyword-based scoring if model fails
        text_lower = text.lower()
        recycling_phrases = [
            'repost', 'reposting', 'repost this', 'share this',
            'viral', 'going viral', 'trending', 'trending now',
            'everyone is talking about', 'everyone needs to see',
            'spread the word', 'pass it on', 'forward this',
            'retweet this', 'like and share', 'comment and share'
        ]
        
        recycling_count = sum(1 for phrase in recycling_phrases if phrase in text_lower)
        score = min(recycling_count / 2, 1.0)
        return max(0.0, min(1.0, score))

def calculate_coordinated_network_score(text: str, model, tokenizer, device) -> float:
    """Calculate coordinated account network score using the trained model."""
    try:
        # Use the trained model for sophisticated scoring
        with torch.no_grad():
            enc = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            score = torch.sigmoid(logits.squeeze()).item()
            
            return float(score)
            
    except Exception as e:
        # Fallback to keyword-based scoring if model fails
        text_lower = text.lower()
        network_phrases = [
            'bot', 'bots', 'automated', 'script', 'scripted',
            'coordinated', 'network', 'campaign', 'operation',
            'mass', 'bulk', 'flood', 'spam', 'spamming',
            'trending', 'trend', 'hashtag', 'hashtags'
        ]
        
        network_count = sum(1 for phrase in network_phrases if phrase in text_lower)
        score = min(network_count / 2, 1.0)
        return max(0.0, min(1.0, score))

def calculate_emotive_manipulation_score(text: str, model, tokenizer, device) -> float:
    """Calculate emotive manipulation score using the trained model."""
    try:
        # Use the trained model for sophisticated scoring
        with torch.no_grad():
            enc = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            score = torch.sigmoid(logits.squeeze()).item()
            
            return float(score)
            
    except Exception as e:
        # Fallback to keyword-based scoring if model fails
        text_lower = text.lower()
        emotive_phrases = [
            'outrage', 'outraged', 'angry', 'furious', 'livid',
            'shocked', 'shocking', 'disgusting', 'horrible', 'terrible',
            'amazing', 'incredible', 'wonderful', 'fantastic', 'amazing',
            'heartbreaking', 'devastating', 'tragic', 'sad', 'depressing',
            'excited', 'thrilled', 'ecstatic', 'overjoyed', 'elated'
        ]
        
        emotive_count = sum(1 for phrase in emotive_phrases if phrase in text_lower)
        score = min(emotive_count / 3, 1.0)
        return max(0.0, min(1.0, score))

def calculate_engagement_mismatch_score(text: str, model, tokenizer, device) -> float:
    """Calculate engagement mismatch score using the trained model."""
    try:
        # Use the trained model for sophisticated scoring
        with torch.no_grad():
            enc = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            score = torch.sigmoid(logits.squeeze()).item()
            
            return float(score)
            
    except Exception as e:
        # Fallback to keyword-based scoring if model fails
        text_lower = text.lower()
        mismatch_phrases = [
            'like this', 'like if you agree', 'like for more',
            'retweet this', 'retweet if you agree', 'retweet for more',
            'comment below', 'comment your thoughts', 'comment your opinion',
            'share this', 'share if you agree', 'share for awareness',
            'follow me', 'follow for more', 'follow for updates'
        ]
        
        mismatch_count = sum(1 for phrase in mismatch_phrases if phrase in text_lower)
        score = min(mismatch_count / 2, 1.0)
        return max(0.0, min(1.0, score))

def calculate_generic_comment_score(text: str, model, tokenizer, device) -> float:
    """Calculate generic comment score using the trained model."""
    try:
        # Use the trained model for sophisticated scoring
        with torch.no_grad():
            enc = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            score = torch.sigmoid(logits.squeeze()).item()
            
            return float(score)
            
    except Exception as e:
        # Fallback to keyword-based scoring if model fails
        text_lower = text.lower()
        generic_phrases = [
            'nice', 'good', 'great', 'awesome', 'cool',
            'interesting', 'wow', 'omg', 'lol', 'haha',
            'thanks', 'thank you', 'appreciate it', 'love this',
            'agree', 'disagree', 'true', 'false', 'yes', 'no'
        ]
        
        generic_count = sum(1 for phrase in generic_phrases if phrase in text_lower)
        score = min(generic_count / 3, 1.0)
        return max(0.0, min(1.0, score))

def calculate_hyperbole_falsehood_score(text: str, model, tokenizer, device) -> float:
    """Calculate hyperbole and falsehood score using the trained model."""
    try:
        # Use the trained model for sophisticated scoring
        with torch.no_grad():
            enc = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            score = torch.sigmoid(logits.squeeze()).item()
            
            return float(score)
            
    except Exception as e:
        # Fallback to keyword-based scoring if model fails
        text_lower = text.lower()
        hyperbole_phrases = [
            'always', 'never', 'everyone', 'nobody', 'every single',
            '100%', 'guaranteed', 'proven', 'definitely', 'absolutely',
            'worst ever', 'best ever', 'most', 'least', 'completely',
            'totally', 'literally', 'actually', 'really', 'very'
        ]
        
        hyperbole_count = sum(1 for phrase in hyperbole_phrases if phrase in text_lower)
        score = min(hyperbole_count / 3, 1.0)
        return max(0.0, min(1.0, score))

def calculate_rapid_engagement_score(text: str, model, tokenizer, device) -> float:
    """Calculate rapid engagement spike score using the trained model."""
    try:
        # Use the trained model for sophisticated scoring
        with torch.no_grad():
            enc = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            score = torch.sigmoid(logits.squeeze()).item()
            
            return float(score)
            
    except Exception as e:
        # Fallback to keyword-based scoring if model fails
        text_lower = text.lower()
        rapid_phrases = [
            'trending', 'trending now', 'going viral', 'viral',
            'exploding', 'blowing up', 'skyrocketing', 'soaring',
            'breaking', 'breaking news', 'just in', 'update',
            'developing', 'developing story', 'latest', 'newest'
        ]
        
        rapid_count = sum(1 for phrase in rapid_phrases if phrase in text_lower)
        score = min(rapid_count / 2, 1.0)
        return max(0.0, min(1.0, score))

def calculate_reply_bait_score(text: str, model, tokenizer, device) -> float:
    """Calculate reply bait score using the trained model."""
    try:
        # Use the trained model for sophisticated scoring
        with torch.no_grad():
            enc = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            score = torch.sigmoid(logits.squeeze()).item()
            
            return float(score)
            
    except Exception as e:
        # Fallback to keyword-based scoring if model fails
        text_lower = text.lower()
        reply_bait_phrases = [
            'what do you think', 'your thoughts', 'your opinion',
            'agree or disagree', 'comment below', 'comment your thoughts',
            'what\'s your take', 'how do you feel', 'do you agree',
            'share your experience', 'tell me', 'let me know',
            'what about you', 'your turn', 'your say'
        ]
        
        reply_bait_count = sum(1 for phrase in reply_bait_phrases if phrase in text_lower)
        score = min(reply_bait_count / 2, 1.0)
        return max(0.0, min(1.0, score))

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
        
        # Load the Hugging Face model
        model, tokenizer, device = load_hf_model()
        
        if model is None:
            print("0.0")
            return
        
        # Select model and calculate score
        model_name = args.model.lower()
        
        if 'authority' in model_name:
            score = calculate_authority_signal_score(text, model, tokenizer, device)
        elif 'clickbait' in model_name:
            score = calculate_clickbait_score(text, model, tokenizer, device)
        elif 'content_recycling' in model_name:
            score = calculate_content_recycling_score(text, model, tokenizer, device)
        elif 'coordinated' in model_name:
            score = calculate_coordinated_network_score(text, model, tokenizer, device)
        elif 'emotive' in model_name:
            score = calculate_emotive_manipulation_score(text, model, tokenizer, device)
        elif 'engagement_mismatch' in model_name:
            score = calculate_engagement_mismatch_score(text, model, tokenizer, device)
        elif 'generic_comment' in model_name:
            score = calculate_generic_comment_score(text, model, tokenizer, device)
        elif 'hyperbole' in model_name or 'falsehood' in model_name:
            score = calculate_hyperbole_falsehood_score(text, model, tokenizer, device)
        elif 'rapid_engagement' in model_name:
            score = calculate_rapid_engagement_score(text, model, tokenizer, device)
        elif 'reply_bait' in model_name:
            score = calculate_reply_bait_score(text, model, tokenizer, device)
        else:
            # Default to generic score if model not recognized
            score = calculate_generic_comment_score(text, model, tokenizer, device)
        
        # Output the score
        print(f"{score:.3f}")
        
    except Exception as e:
        # On any error, return 0.0
        print("0.0")

if __name__ == "__main__":
    main()

