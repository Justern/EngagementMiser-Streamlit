#!/usr/bin/env python3
"""
Standalone Content Recycling Detector Scorer
===========================================

This script loads a pre-trained content recycling detection model and scores
individual tweets by their tweet_id to identify recycled or reused content.

Usage:
    python score_content_recycling_standalone.py <tweet_id>
    python score_content_recycling_standalone.py --interactive
"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text as sql_text
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from typing import Union, Tuple
import warnings
warnings.filterwarnings('ignore')


# Automatically navigate to the script's directory
SCRIPT_DIR = r"C:\Users\justi\OneDrive\Desktop\MSc. Data Science\DS Capstone\Supporting_Files\Models\Content_Recycling_Detector"
os.chdir(SCRIPT_DIR)
print(f"📁 Working directory set to: {SCRIPT_DIR}")

# Configuration
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths
MODEL_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "content_recycling_detector")
SENTENCE_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# SQL connection details
SQL_SERVER = os.getenv("SQL_SERVER", "localhost")
SQL_DB = os.getenv("SQL_DB", "EngagementMiser")
SQL_DRIVER = os.getenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server")

# Connection string for SQL Server over ODBC with Windows Authentication
CONN_STR = (
    f"mssql+pyodbc://@{SQL_SERVER}/{SQL_DB}"
    f"?driver={SQL_DRIVER.replace(' ', '+')}"
    "&Trusted_Connection=yes"
    "&TrustServerCertificate=yes"
)

def load_model_and_tokenizer():
    """Load the pre-trained model and tokenizer."""
    try:
        # Load tokenizer and model from the checkpoints directory
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        model.eval()
        print(f"✅ Model loaded successfully: {os.path.basename(MODEL_PATH)}")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Make sure you have trained the model first using content_recycling_detector.py")
        sys.exit(1)

def create_database_connection():
    """Create and test database connection."""
    try:
        engine = create_engine(CONN_STR, fast_executemany=True)
        # Test connection
        with engine.connect() as conn:
            conn.execute(sql_text("SELECT 1"))
        return engine
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)

def find_similar_content(text: str, engine, top_k=5):
    """Find similar content in the database using semantic similarity."""
    try:
        # Encode the input text
        text_embedding = SENTENCE_MODEL.encode(text, convert_to_tensor=True)
        
        # Query for similar content
        query = sql_text("""
            SELECT TOP 10
                tweet_id,
                clean_text,
                created_at,
                user_id,
                like_count,
                retweet_count,
                reply_count,
                quote_count
            FROM [EngagementMiser].[dbo].[Tweets_Sample_4M]
            WHERE clean_text IS NOT NULL 
                AND LEN(clean_text) > 10
                AND clean_text != :input_text
            ORDER BY CHECKSUM(NEWID())
        """)
        
        with engine.connect() as conn:
            results = conn.execute(query, {"input_text": text}).fetchall()
        
        if not results:
            return []
        
        # Calculate similarities
        similarities = []
        for row in results:
            try:
                # Try to access by column name first
                similar_text = str(row.clean_text)
                tweet_id = str(row.tweet_id)
                created_at = row.created_at
                user_id = str(row.user_id)
                engagement = (row.like_count or 0) + (row.retweet_count or 0) + (row.reply_count or 0) + (row.quote_count or 0)
            except AttributeError:
                # Fallback to index access
                similar_text = str(row[1])
                tweet_id = str(row[0])
                created_at = row[2]
                user_id = str(row[3])
                engagement = (row[4] or 0) + (row[5] or 0) + (row[6] or 0) + (row[7] or 0)
            
            # Calculate semantic similarity
            similar_embedding = SENTENCE_MODEL.encode(similar_text, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(text_embedding, similar_embedding).item()
            
            similarities.append({
                'tweet_id': tweet_id,
                'text': similar_text,
                'similarity': similarity,
                'created_at': created_at,
                'user_id': user_id,
                'engagement': engagement
            })
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
        
    except Exception as e:
        print(f"⚠️ Warning: Could not find similar content: {e}")
        return []

def predict_content_recycling(text: str, model, tokenizer) -> float:
    """Predict content recycling score for a single text input."""
    model.eval()
    
    with torch.no_grad():
        enc = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        score = torch.sigmoid(logits.squeeze()).item()
        
        return float(score)

def get_recycling_interpretation(score: float) -> str:
    """Get interpretation of the content recycling score."""
    if score < 0.3:
        return "Original content - low recycling likelihood"
    elif score < 0.5:
        return "Slightly recycled content - moderate similarity"
    elif score < 0.7:
        return "Recycled content - high similarity to existing posts"
    else:
        return "Highly recycled content - likely duplicate or repost"

def calculate_content_freshness(created_at) -> float:
    """Calculate content freshness score based on tweet age."""
    if pd.isna(created_at):
        return 0.5  # Neutral score for unknown dates
    
    try:
        tweet_date = pd.to_datetime(created_at)
        days_old = (datetime.now() - tweet_date).days
        
        if days_old < 1:
            return 1.0  # Very fresh
        elif days_old < 7:
            return 0.8  # Fresh
        elif days_old < 30:
            return 0.6  # Moderately fresh
        elif days_old < 90:
            return 0.4  # Older
        else:
            return 0.2  # Very old
    except:
        return 0.5

def score_tweet_id(tweet_id: Union[str, int], model, tokenizer, engine) -> Tuple[float, str, dict]:
    """Look up tweet by tweet_id and return recycling score, text, and metadata."""
    query = sql_text("""
        SELECT TOP 1 
            t.text, 
            t.clean_text, 
            t.created_at,
            t.like_count, 
            t.retweet_count, 
            t.reply_count, 
            t.quote_count,
            t.hashtag1, 
            t.hashtag2, 
            t.hashtag3,
            t.has_popular_entity,
            u.followers_count,
            u.tweet_count,
            u.verified,
            u.protected
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
        INNER JOIN [EngagementMiser].[dbo].[TwitterUsers] u 
            ON t.author_id = u.id
        WHERE CAST(t.tweet_id AS VARCHAR(32)) = :tid
    """)
    
    try:
        with engine.connect() as conn:
            row = conn.execute(query, {"tid": str(tweet_id)}).fetchone()
        
        if not row:
            raise ValueError(f"tweet_id {tweet_id} not found.")
        
        # Try to access text fields using different methods
        try:
            # Try attribute access first
            tweet_text = str(row.clean_text if row.clean_text else row.text)
            created_at = row.created_at
            hashtags = [h for h in [row.hashtag1, row.hashtag2, row.hashtag3] if pd.notna(h)]
            followers = row.followers_count or 0
            tweet_count = row.tweet_count or 0
            verified = row.verified
            protected = row.protected
        except AttributeError:
            try:
                # Try index access
                tweet_text = str(row[1] if row[1] else row[0])
                created_at = row[2]
                hashtags = [h for h in [row[8], row[9], row[10]] if pd.notna(h)]
                followers = row[12] or 0
                tweet_count = row[13] or 0
                verified = row[14]
                protected = row[15]
            except (IndexError, TypeError):
                # Last resort: convert to dict
                row_dict = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
                tweet_text = str(row_dict.get('clean_text', row_dict.get('text', '')))
                created_at = row_dict.get('created_at')
                hashtags = [h for h in [row_dict.get('hashtag1'), row_dict.get('hashtag2'), row_dict.get('hashtag3')] if pd.notna(h)]
                followers = row_dict.get('followers_count', 0) or 0
                tweet_count = row_dict.get('tweet_count', 0) or 0
                verified = row_dict.get('verified')
                protected = row_dict.get('protected')
        
        if not tweet_text:
            raise ValueError(f"tweet_id {tweet_id} has no text content.")
        
        # Get recycling score
        score = predict_content_recycling(tweet_text, model, tokenizer)
        
        # Find similar content
        similar_content = find_similar_content(tweet_text, engine)
        
        # Calculate content freshness
        freshness = calculate_content_freshness(created_at)
        
        # Create metadata
        metadata = {
            'created_at': created_at,
            'hashtags': hashtags,
            'followers_count': followers,
            'tweet_count': tweet_count,
            'verified': verified,
            'protected': protected,
            'content_freshness': freshness,
            'similar_content_count': len(similar_content),
            'similar_content': similar_content[:3]  # Top 3 similar
        }
        
        return score, tweet_text, metadata
        
    except Exception as e:
        raise ValueError(f"Database error for tweet_id {tweet_id}: {e}")

def main():
    """Main function to score tweets for content recycling."""
    parser = argparse.ArgumentParser(description='Score tweet for content recycling likelihood')
    parser.add_argument('tweet_id', nargs='?', help='Tweet ID to score (optional, will prompt if not provided)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    try:
        # Load model and tokenizer
        print("🔄 Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer()
        print("✅ Model loaded successfully!")
        
        # Create database connection
        print("🔄 Connecting to database...")
        engine = create_database_connection()
        print("✅ Database connected!")
        
        if args.tweet_id:
            # Use command-line argument
            tweet_id = args.tweet_id
            print(f"📱 Scoring tweet ID: {tweet_id}")
            
            try:
                score, text, metadata = score_tweet_id(tweet_id, model, tokenizer, engine)
                interpretation = get_recycling_interpretation(score)
                
                print(f"\n🎯 Content Recycling Score: {score:.3f}")
                print(f"📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
                print(f"📊 Interpretation: {interpretation}")
                print(f"🔢 Content Analysis:")
                print(f"   Content Freshness: {metadata['content_freshness']:.2f}")
                print(f"   Hashtags: {', '.join(metadata['hashtags']) if metadata['hashtags'] else 'None'}")
                print(f"   Similar Content Found: {metadata['similar_content_count']}")
                print(f"   User Status: {'Verified' if metadata['verified'] else 'Unverified'}")
                print(f"   Profile Type: {'Protected' if metadata['protected'] else 'Public'}")
                
                if metadata['similar_content']:
                    print(f"\n🔍 Top Similar Content:")
                    for i, similar in enumerate(metadata['similar_content'], 1):
                        print(f"   {i}. Tweet {similar['tweet_id']} (Similarity: {similar['similarity']:.3f})")
                        print(f"      Text: {similar['text'][:80]}{'...' if len(similar['text']) > 80 else ''}")
                        
            except ValueError as e:
                print(f"❌ Error: {e}")
                
        elif args.interactive:
            # Interactive mode
            while True:
                try:
                    tweet_id = input("\n📱 Enter tweet ID (or 'quit' to exit): ").strip()
                    
                    if tweet_id.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break
                    
                    if not tweet_id:
                        print("❌ Please enter a valid tweet ID.")
                        continue
                    
                    print(f"🔄 Scoring tweet ID: {tweet_id}...")
                    
                    score, text, metadata = score_tweet_id(tweet_id, model, tokenizer, engine)
                    interpretation = get_recycling_interpretation(score)
                    
                    print(f"\n🎯 Content Recycling Score: {score:.3f}")
                    print(f"📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
                    print(f"📊 Interpretation: {interpretation}")
                    print(f"🔢 Content Analysis:")
                    print(f"   Content Freshness: {metadata['content_freshness']:.2f}")
                    print(f"   Hashtags: {', '.join(metadata['hashtags']) if metadata['hashtags'] else 'None'}")
                    print(f"   Similar Content Found: {metadata['similar_content_count']}")
                    
                except ValueError as e:
                    print(f"❌ Error: {e}")
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except EOFError:
                    print("\n👋 Goodbye!")
                    break
        else:
            # No arguments provided, show usage
            print("📖 Usage:")
            print("  python score_content_recycling_standalone.py <tweet_id>")
            print("  python score_content_recycling_standalone.py --interactive")
            print("\nExample:")
            print("  python score_content_recycling_standalone.py 1503813284")
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
