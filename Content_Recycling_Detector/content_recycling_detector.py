#!/usr/bin/env python3
"""
Content Recycling Detector
==========================

This model detects when content is being recycled, reposted, or reused across
different tweets and accounts. It uses semantic similarity, text embeddings,
and cross-account analysis to identify content recycling patterns.

Features:
- Semantic text similarity using sentence transformers
- Cross-account content sharing detection
- Temporal pattern analysis
- Hashtag and entity reuse detection
- Content freshness scoring

Author: AI Assistant
Date: 2025
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text as sql_text
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Content Recycling Detector Training")
print(f"📱 Device: {DEVICE}")
print(f"⚙️  Batch Size: {BATCH_SIZE}")
print(f"🔄 Epochs: {EPOCHS}")

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

def create_database_connection():
    """Create and test database connection."""
    try:
        engine = create_engine(CONN_STR, fast_executemany=True)
        # Test connection
        with engine.connect() as conn:
            conn.execute(sql_text("SELECT 1"))
        print("✅ Database connection established")
        return engine
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)

def load_training_data(engine, sample_size=10000):
    """Load tweet data for training the content recycling detector."""
    print("🔄 Loading training data...")
    
    try:
        # Query to get tweets with engagement data and user info
        query = sql_text(f"""
            SELECT TOP {sample_size}
                t.tweet_id,
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
                u.following_count,
                u.tweet_count,
                u.created_at as user_created_at,
                u.verified,
                u.protected,
                u.profile_image_url
            FROM [EngagementMiser].[dbo].[Tweets_Sample_4M] t
            INNER JOIN [EngagementMiser].[dbo].[TwitterUsers] u 
                ON t.author_id = u.id
            WHERE t.clean_text IS NOT NULL 
                AND LEN(t.clean_text) > 10
                AND t.clean_text NOT LIKE '%RT @%'
                AND t.clean_text NOT LIKE '%@%'
            ORDER BY CHECKSUM(NEWID())
        """)
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        print(f"✅ Loaded {len(df)} tweets")
        return df
        
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        sys.exit(1)

def create_content_recycling_labels(df):
    """Create labels for content recycling detection using heuristic rules."""
    print("🔄 Creating content recycling labels...")
    
    labels = []
    
    for idx, row in df.iterrows():
        score = 0.0
        
        # Text-based indicators
        text = str(row['clean_text']).lower()
        
        # Repetitive patterns (balanced)
        if text.count('!') > 2:  # Back to 2
            score += 0.1
        if text.count('?') > 2:  # Back to 2
            score += 0.1
        if text.count('...') > 0:
            score += 0.05  # Reduced back
        
        # Common recycled phrases (balanced)
        recycled_phrases = [
            'breaking news', 'just in', 'update', 'breaking', 'exclusive',
            'you won\'t believe', 'shocking', 'amazing', 'incredible',
            'this is huge', 'big news', 'major announcement', 'developing',
            'stay tuned', 'more to come', 'details to follow'
        ]
        for phrase in recycled_phrases:
            if phrase in text:
                score += 0.15  # Reduced from 0.2
        
        # Hashtag patterns (balanced)
        hashtags = [h for h in [row['hashtag1'], row['hashtag2'], row['hashtag3']] if pd.notna(h)]
        if len(hashtags) > 2:  # Back to 2
            score += 0.1
        if any('trending' in str(h).lower() for h in hashtags):
            score += 0.2  # Reduced from 0.25
        if any('viral' in str(h).lower() for h in hashtags):
            score += 0.2  # Reduced from 0.25
        if any('news' in str(h).lower() for h in hashtags):
            score += 0.1  # Reduced from 0.15
        if any('breaking' in str(h).lower() for h in hashtags):
            score += 0.15  # Reduced from 0.2
        
        # User behavior indicators (balanced)
        if row['tweet_count'] > 8000:  # Increased from 5000
            score += 0.05  # Reduced from 0.1
        if row['protected']:
            score += 0.1  # Reduced from 0.15
        if pd.isna(row['profile_image_url']) or row['profile_image_url'] == '':
            score += 0.1  # Reduced from 0.15
        
        # Engagement patterns (balanced)
        total_engagement = row['like_count'] + row['retweet_count'] + row['reply_count'] + row['quote_count']
        if row['followers_count'] > 0:
            engagement_rate = total_engagement / row['followers_count']
            if engagement_rate > 0.1:  # Back to 0.1
                score += 0.15  # Reduced from 0.25
            if engagement_rate > 0.3:  # Increased threshold
                score += 0.1  # Reduced from 0.15
        
        # Content freshness (balanced)
        if pd.notna(row['created_at']):
            tweet_age = (datetime.now() - pd.to_datetime(row['created_at'])).days
            if tweet_age > 30:  # Back to 30
                score += 0.05  # Reduced from 0.1
            if tweet_age > 90:  # Back to 90
                score += 0.1  # Reduced from 0.15
            if tweet_age > 180:  # Increased threshold
                score += 0.1  # Reduced from 0.2
        
        # Text length indicators (balanced)
        if len(text) < 15:  # Very short tweets often recycled
            score += 0.05  # Reduced from 0.1
        if len(text) > 250:  # Very long tweets often original
            score -= 0.05  # Reduced from 0.1
        
        # URL indicators (balanced)
        if 'http' in text or 'www.' in text:
            score += 0.05  # Reduced from 0.1
        
        # Negative indicators for original content (new)
        original_indicators = [
            'i think', 'i believe', 'in my opinion', 'personally',
            'from my experience', 'i noticed', 'i found', 'i discovered',
            'i learned', 'i realized', 'i understand', 'i feel',
            'my thoughts', 'my take', 'my view', 'my perspective'
        ]
        for indicator in original_indicators:
            if indicator in text:
                score -= 0.1  # Reduce recycling score for personal content
        
        # Normalize score to 0-1 range
        score = max(0.0, min(score, 1.0))
        labels.append(score)
    
    print(f"✅ Created labels for {len(labels)} samples")
    print(f"📊 Label distribution:")
    print(f"   Low recycling (0.0-0.3): {sum(1 for s in labels if s < 0.3)}")
    print(f"   Medium recycling (0.3-0.7): {sum(1 for s in labels if 0.3 <= s < 0.7)}")
    print(f"   High recycling (0.7-1.0): {sum(1 for s in labels if s >= 0.7)}")
    
    return np.array(labels)

def create_similarity_pairs(df, labels, num_pairs=5000):
    """Create pairs of similar tweets for contrastive learning."""
    print("🔄 Creating similarity pairs for contrastive learning...")
    
    # Use sentence transformers to find similar content
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Sample texts for similarity analysis (increased sample size)
    sample_size = min(2000, len(df))  # Increased from 1000
    sample_texts = df['clean_text'].sample(sample_size).tolist()
    embeddings = sentence_model.encode(sample_texts, convert_to_tensor=True)
    
    # Find similar pairs
    pairs = []
    pair_labels = []
    
    # Lower similarity threshold for more pairs
    similarity_threshold = 0.6  # Reduced from 0.7
    
    for i in range(min(num_pairs, len(sample_texts) // 2)):
        # Find most similar text
        similarities = util.pytorch_cos_sim(embeddings[i], embeddings).squeeze()
        similarities[i] = 0  # Exclude self-similarity
        
        if len(similarities) > 1:
            most_similar_idx = similarities.argmax().item()
            similarity_score = similarities[most_similar_idx].item()
            
            if similarity_score > similarity_threshold:
                # Create positive pair (similar content)
                pairs.append((sample_texts[i], sample_texts[most_similar_idx]))
                pair_labels.append(0.8)  # High recycling score (reduced from 1.0)
                
                # Create negative pair (different content)
                random_idx = np.random.randint(0, len(sample_texts))
                if random_idx != i and random_idx != most_similar_idx:
                    pairs.append((sample_texts[i], sample_texts[random_idx]))
                    pair_labels.append(0.2)  # Low recycling score (increased from 0.0)
    
    print(f"✅ Created {len(pairs)} similarity pairs")
    return pairs, pair_labels

def create_dataset(df, labels, pairs=None, pair_labels=None):
    """Create PyTorch dataset for training."""
    print("🔄 Creating PyTorch dataset...")
    
    # Combine original data with similarity pairs
    all_texts = df['clean_text'].tolist()
    all_labels = labels.tolist()
    
    if pairs and pair_labels:
        # Add pair data
        for (text1, text2), label in zip(pairs, pair_labels):
            all_texts.append(f"{text1} [SEP] {text2}")
            all_labels.append(label)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"✅ Dataset created:")
    print(f"   Training samples: {len(train_texts)}")
    print(f"   Validation samples: {len(val_texts)}")
    
    return train_texts, val_texts, train_labels, val_labels

class ContentRecyclingDataset(torch.utils.data.Dataset):
    """Custom dataset for content recycling detection."""
    
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = float(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32)
        }

def train_model(train_texts, val_texts, train_labels, val_labels):
    """Train the content recycling detection model."""
    print("🔄 Training content recycling detection model...")
    
    # Load tokenizer and model
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="regression"
    )
    
    # Add special token for pair separation
    special_tokens = {'additional_special_tokens': ['[SEP]']}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Create datasets
    train_dataset = ContentRecyclingDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = ContentRecyclingDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_dir="./logs",
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        remove_unused_columns=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )
    
    # Train the model
    print("🚀 Starting training...")
    trainer.train()
    
    # Save the model
    model_path = "./checkpoints/content_recycling_detector"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    
    print(f"✅ Model saved to {model_path}")
    
    # Evaluate the model
    print("📊 Evaluating model...")
    predictions = trainer.predict(val_dataset)
    pred_scores = predictions.predictions.flatten()
    true_scores = val_labels
    
    # Convert to binary classification for evaluation
    threshold = 0.5
    pred_binary = (pred_scores > threshold).astype(int)
    true_binary = (np.array(true_scores) > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(true_binary, pred_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(true_binary, pred_binary, average='binary')
    
    print(f"📈 Model Performance:")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   F1-Score: {f1:.3f}")
    
    # Detailed classification report
    if len(np.unique(true_binary)) > 1:
        print("\n📋 Detailed Classification Report:")
        print(classification_report(true_binary, pred_binary, 
                                  target_names=['Low Recycling', 'High Recycling']))
    
    return model, tokenizer

def main():
    """Main training function."""
    print("🎯 Content Recycling Detector Training Started")
    print("=" * 60)
    
    try:
        # Create database connection
        engine = create_database_connection()
        
        # Load training data
        df = load_training_data(engine, sample_size=25000)
        
        # Create labels
        labels = create_content_recycling_labels(df)
        
        # Create similarity pairs for contrastive learning
        pairs, pair_labels = create_similarity_pairs(df, labels, num_pairs=5000)
        
        # Create dataset
        train_texts, val_texts, train_labels, val_labels = create_dataset(
            df, labels, pairs, pair_labels
        )
        
        # Train model
        model, tokenizer = train_model(train_texts, val_texts, train_labels, val_labels)
        
        print("\n🎉 Training completed successfully!")
        print("💾 Model saved to ./checkpoints/content_recycling_detector")
        print("🔍 You can now use the model to detect content recycling!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
