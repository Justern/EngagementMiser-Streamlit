#!/usr/bin/env python3
"""
Clickbait Headline Classifier
=============================

This script trains a RoBERTa-based model to classify headlines as clickbait or not,
incorporating popular entities and colloquial phrasing patterns from the corpus.

Features:
- Popular entity frequency analysis
- Colloquial phrasing detection
- Engagement pattern analysis
- Text-based clickbait patterns

Usage:
    python clickbait_classifier.py
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
from torch import nn
import re
from typing import List, Tuple, Dict, Union
from sqlalchemy import create_engine, text as sql_text
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_NAME = "roberta-base"
MAX_LEN = 128  # Reduced to save memory
BATCH_SIZE = 32  # Increased batch size for better learning
EPOCHS = 100  # Reduced epochs but with better learning rate
LEARNING_RATE = 2e-5  # Increased learning rate for better convergence
WARMUP_STEPS = 1000  # Reduced warmup steps for faster learning
WEIGHT_DECAY = 0.01  # Reduced weight decay for less aggressive regularization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class ClickbaitDataset(Dataset):
    """Custom dataset for clickbait classification."""
    
    def __init__(self, texts: List[str], labels: List[float], tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def create_database_connection():
    """Create and test database connection."""
    try:
        engine = create_engine(CONN_STR, fast_executemany=True)
        with engine.connect() as conn:
            conn.execute(sql_text("SELECT 1"))
        print("✅ Database connection successful")
        return engine
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)

def load_popular_entities(engine) -> Dict[str, float]:
    """Load popular entities and their frequency scores."""
    try:
        query = sql_text(
            """
            SELECT entity_name, frequency_count, confidence_score
            FROM [EngagementMiser].[dbo].[Popular_Entities_Corpus]
            WHERE confidence_score > 0.5
            ORDER BY frequency_count DESC
            """
        )
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        # Create entity score mapping
        entity_scores = {}
        for _, row in df.iterrows():
            # Normalize score based on frequency and confidence
            score = (row['frequency_count'] * row['confidence_score']) / 1000
            entity_scores[row['entity_name'].lower()] = min(score, 1.0)
        
        print(f"✅ Loaded {len(entity_scores)} popular entities")
        return entity_scores
        
    except Exception as e:
        print(f"⚠️ Warning: Could not load popular entities: {e}")
        return {}

def load_colloquial_phrases(engine) -> List[str]:
    """Load colloquial phrases that might indicate clickbait."""
    try:
        query = sql_text(
            """
            SELECT phrase, meaning
            FROM [EngagementMiser].[dbo].[Colloquial_Phrasing_Corpus]
            """
        )
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        phrases = df['phrase'].tolist()
        print(f"✅ Loaded {len(phrases)} colloquial phrases")
        return phrases
        
    except Exception as e:
        print(f"⚠️ Warning: Could not load colloquial phrases: {e}")
        return []

def extract_clickbait_features(text: str, popular_entities: Dict[str, float], 
                             colloquial_phrases: List[str]) -> Dict[str, float]:
    """Extract features that indicate clickbait content."""
    
    text_lower = text.lower()
    
    # 1. Popular entity density
    entity_score = 0.0
    entity_count = 0
    for entity, score in popular_entities.items():
        if entity in text_lower:
            entity_score += score
            entity_count += 1
    
    entity_density = entity_score / max(len(text.split()), 1)
    
    # 2. Colloquial phrase density
    colloquial_score = 0.0
    for phrase in colloquial_phrases:
        if phrase.lower() in text_lower:
            colloquial_score += 1.0
    
    colloquial_density = colloquial_score / max(len(text.split()), 1)
    
    # 3. Clickbait pattern indicators
    patterns = {
        'question_mark': text_lower.count('?') / max(len(text), 1),
        'exclamation': text_lower.count('!') / max(len(text), 1),
        'numbers': len(re.findall(r'\d+', text)) / max(len(text.split()), 1),
        'capital_words': len(re.findall(r'\b[A-Z]{2,}\b', text)) / max(len(text.split()), 1),
        'emotional_words': len(re.findall(r'\b(wow|amazing|incredible|shocking|unbelievable|crazy|insane|mind-blowing)\b', text_lower)) / max(len(text.split()), 1),
        'urgency_words': len(re.findall(r'\b(breaking|urgent|now|immediately|last|final|deadline|rush)\b', text_lower)) / max(len(text.split()), 1),
        'mystery_words': len(re.findall(r'\b(secret|hidden|revealed|exposed|truth|mystery|unknown|discovered)\b', text_lower)) / max(len(text.split()), 1)
    }
    
    # 4. Length and structure features
    structure_features = {
        'text_length': min(len(text) / 100, 1.0),
        'word_count': min(len(text.split()) / 20, 1.0),
        'hashtag_count': text_lower.count('#') / max(len(text.split()), 1),
        'mention_count': text_lower.count('@') / max(len(text.split()), 1)
    }
    
    return {
        'entity_density': entity_density,
        'colloquial_density': colloquial_density,
        **patterns,
        **structure_features
    }

def create_clickbait_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create clickbait labels using heuristic scoring."""
    
    print("🏷️ Creating clickbait labels...")
    
    scores = []
    for idx, row in df.iterrows():
        score = 0.0
        
        # 1. Engagement rate (higher = more likely clickbait) - BALANCED WEIGHTS
        if row['engagement_rate'] > 0.05:  # 5% engagement rate
            score += 0.4  # Reduced from 0.7
        elif row['engagement_rate'] > 0.02:  # 2% engagement rate
            score += 0.3  # Reduced from 0.6
        elif row['engagement_rate'] > 0.005:  # 0.5% engagement rate
            score += 0.2  # Reduced from 0.5
        elif row['engagement_rate'] > 0.001:  # 0.1% engagement rate
            score += 0.1  # Reduced from 0.3
        
        # 2. Text length (very short or very long = more likely clickbait) - BALANCED WEIGHTS
        text_len = len(str(row['text']))
        if text_len < 30 or text_len > 300:
            score += 0.3  # Reduced from 0.6
        elif text_len < 50 or text_len > 200:
            score += 0.2  # Reduced from 0.5
        elif text_len < 80 or text_len > 150:
            score += 0.1  # Reduced from 0.3
        
        # 3. Hashtag density (more hashtags = more likely clickbait) - BALANCED WEIGHTS
        hashtag_count = sum([1 for col in ['hashtag1', 'hashtag2', 'hashtag3'] if pd.notna(row[col])])
        if hashtag_count >= 3:
            score += 0.3  # Reduced from 0.6
        elif hashtag_count >= 2:
            score += 0.2  # Reduced from 0.5
        elif hashtag_count >= 1:
            score += 0.1  # Reduced from 0.2
        
        # 4. Popular entity presence - BALANCED WEIGHT
        if row['has_popular_entity']:
            score += 0.2  # Reduced from 0.5
        
        # 5. Engagement bucket (higher bucket = more likely clickbait) - BALANCED WEIGHTS
        if pd.notna(row['eng_bucket_3']):
            try:
                bucket = int(row['eng_bucket_3'])
                if bucket >= 8:  # Very high engagement bucket
                    score += 0.3  # Reduced from 0.6
                elif bucket >= 6:  # High engagement bucket
                    score += 0.2  # Reduced from 0.5
                elif bucket >= 4:  # Medium engagement bucket
                    score += 0.1  # Reduced from 0.3
            except:
                pass
        
        # 6. Add text pattern indicators - BALANCED WEIGHTS
        text_lower = str(row['text']).lower()
        if any(word in text_lower for word in ['breaking', 'urgent', 'shocking', 'amazing', 'incredible', 'unbelievable', 'crazy', 'insane']):
            score += 0.2  # Reduced from 0.5
        if text_lower.count('!') > 0:
            score += 0.1  # Reduced from 0.3
        if text_lower.count('?') > 0:
            score += 0.1  # Reduced from 0.3
        
        # 7. Add more clickbait indicators - BALANCED WEIGHTS
        if any(word in text_lower for word in ['you won\'t believe', 'this will shock you', 'what happened next', 'the truth about', 'secret', 'hidden', 'revealed']):
            score += 0.2  # Reduced from 0.4
        if any(word in text_lower for word in ['viral', 'trending', 'everyone is talking about', 'internet is losing it']):
            score += 0.2  # Reduced from 0.4
        if re.search(r'\d+', text_lower):  # Contains numbers
            score += 0.1  # Reduced from 0.2
        if re.search(r'\b[A-Z]{2,}\b', str(row['text'])):  # ALL CAPS words
            score += 0.1  # Reduced from 0.3
        
        # 8. Add negative indicators (reduce score for likely non-clickbait)
        if any(word in text_lower for word in ['fact', 'research', 'study', 'data', 'analysis', 'report']):
            score -= 0.1
        if text_len > 100 and text_len < 200:  # Optimal length range
            score -= 0.1
        if hashtag_count == 0:  # No hashtags
            score -= 0.1
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        scores.append(score)
    
    df['clickbait_score'] = scores
    
    # Print score distribution
    score_dist = df['clickbait_score'].value_counts(bins=10).sort_index()
    print("📊 Score distribution:")
    for bin_range, count in score_dist.items():
        print(f"   {bin_range}: {count} samples")
    
    return df

def load_training_data(engine, popular_entities: Dict[str, float], 
                      colloquial_phrases: List[str], sample_size: int = 20000) -> Tuple[List[str], List[float]]:
    """Load and prepare training data from Tweets_Sample_4M."""
    
    print("📊 Loading training data...")
    
    # Load sample of tweets with better balance - INCREASED SAMPLE SIZE
    query = sql_text(
        f"""
        SELECT TOP ({sample_size}) 
            text, clean_text, like_count, retweet_count, reply_count, quote_count,
            followers_count, hashtag1, hashtag2, hashtag3, has_popular_entity, eng_bucket_3
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M]
        WHERE text IS NOT NULL 
            AND LEN(text) > 20 
            AND LEN(text) < 500
            AND followers_count > 0
            AND (
                -- Include high engagement tweets (likely clickbait) - BALANCED THRESHOLD
                (like_count + retweet_count + reply_count + quote_count) / NULLIF(followers_count, 0) > 0.005
                OR
                -- Include tweets with popular entities
                has_popular_entity = 1
                OR
                -- Include tweets with multiple hashtags
                (CASE WHEN hashtag1 IS NOT NULL THEN 1 ELSE 0 END + 
                 CASE WHEN hashtag2 IS NOT NULL THEN 1 ELSE 0 END + 
                 CASE WHEN hashtag3 IS NOT NULL THEN 1 ELSE 0 END) >= 1
                OR
                -- Include tweets with emotional/urgent words
                (text LIKE '%!%' OR text LIKE '%?%' OR text LIKE '%breaking%' OR text LIKE '%urgent%' 
                 OR text LIKE '%amazing%' OR text LIKE '%incredible%' OR text LIKE '%shocking%')
                OR
                -- Random sample for balance - BALANCED FREQUENCY
                CHECKSUM(NEWID()) % 10 = 0
            )
        ORDER BY CHECKSUM(NEWID())
        """
    )
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    print(f"📈 Loaded {len(df)} tweets")
    
    # Calculate engagement rate before creating labels
    df['engagement_rate'] = (df['like_count'] + df['retweet_count'] + df['reply_count'] + df['quote_count']) / df['followers_count'].replace(0, 1)
    
    # Create clickbait labels
    df = create_clickbait_labels(df)
    
    # Use clean_text if available, otherwise use text
    texts = df['clean_text'].fillna(df['text']).tolist()
    labels = df['clickbait_score'].tolist()
    
    # Filter out very low-quality samples and create better balance - IMPROVED LOGIC
    filtered_data = []
    high_score_count = 0
    low_score_count = 0
    medium_score_count = 0
    
    for text, label in zip(texts, labels):
        if len(str(text).strip()) > 20 and label >= 0.0:
            if label > 0.6 and high_score_count < 1500:  # Balanced threshold and count
                filtered_data.append((text, label))
                high_score_count += 1
            elif label < 0.4 and low_score_count < 1500:  # Balanced threshold and count
                filtered_data.append((text, label))
                low_score_count += 1
            elif 0.4 <= label <= 0.6 and medium_score_count < 1000:  # Balanced count
                filtered_data.append((text, label))
                medium_score_count += 1
    
    # Add more data augmentation for high-score samples to improve balance
    if high_score_count < 1000:  # If we don't have enough high-score samples
        high_score_texts = [text for text, label in filtered_data if label > 0.6]
        high_score_labels = [label for text, label in filtered_data if label > 0.6]
        
        # Balanced augmentation: add variations
        for i in range(min(len(high_score_texts), 500)):  # Balanced augmented samples
            text = high_score_texts[i]
            label = high_score_labels[i]
            
            # Add some variations
            if '!' in text:
                augmented_text = text.replace('!', '!!!')
                filtered_data.append((augmented_text, min(label + 0.05, 1.0)))
            
            if '?' in text:
                augmented_text = text + " Find out more!"
                filtered_data.append((augmented_text, min(label + 0.05, 1.0)))
            
            # Add more variations
            if 'amazing' in text.lower():
                augmented_text = text.replace('amazing', 'INCREDIBLY AMAZING')
                filtered_data.append((augmented_text, min(label + 0.05, 1.0)))
            
            if 'breaking' in text.lower():
                augmented_text = text.replace('breaking', 'BREAKING NEWS')
                filtered_data.append((augmented_text, min(label + 0.05, 1.0)))
    
    texts, labels = zip(*filtered_data)
    
    print(f"✅ Prepared {len(texts)} training samples")
    print(f"📊 Label distribution: {np.mean(labels):.3f} mean, {np.std(labels):.3f} std")
    print(f"📊 High score samples (>0.6): {sum(1 for l in labels if l > 0.6)}")
    print(f"📊 Low score samples (<0.4): {sum(1 for l in labels if l < 0.4)}")
    print(f"📊 Medium score samples (0.4-0.6): {sum(1 for l in labels if 0.4 <= l <= 0.6)}")
    print(f"📊 Score range: {min(labels):.3f} to {max(labels):.3f}")
    
    return list(texts), list(labels)

def train_model(texts: List[str], labels: List[float], 
                popular_entities: Dict[str, float], colloquial_phrases: List[str]):
    """Train the clickbait classification model."""
    
    print("🚀 Starting model training...")
    
    # Split data - remove stratification for regression scores
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=1,
        problem_type="regression",
        hidden_dropout_prob=0.1,  # Reduced dropout for better learning
        attention_probs_dropout_prob=0.1  # Reduced dropout for better learning
    )
    
    # Add special tokens for clickbait patterns
    special_tokens = ["[CLICKBAIT]", "[SENSATIONAL]", "[URGENT]", "[MYSTERY]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create datasets
    train_dataset = ClickbaitDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = ClickbaitDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    
    # Create dataloaders for evaluation
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Training arguments - IMPROVED FOR BETTER LEARNING
    training_args = TrainingArguments(
        output_dir="./checkpoints/clickbait_classifier",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_dir="./logs",
        logging_steps=50,  # More frequent logging
        eval_steps=200,  # More frequent evaluation
        save_steps=500,  # More frequent saving
        save_total_limit=5,  # Keep more checkpoints
        report_to=None,
        learning_rate=LEARNING_RATE,
        gradient_accumulation_steps=2,  # Reduced for better gradient flow
        dataloader_pin_memory=False,  # Avoid memory issues on Windows
        fp16=True,  # Enable mixed precision to save memory
        dataloader_num_workers=0,  # Avoid multiprocessing issues on Windows
        # Removed problematic parameters for compatibility
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,
        # evaluation_strategy="steps",
        # save_strategy="steps"
    )
    
    # Initialize trainer with custom optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=len(train_dataset) // BATCH_SIZE * EPOCHS
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        optimizers=(optimizer, scheduler)
    )
    
    # Train model
    print("🔥 Training in progress...")
    
    # Add training progress monitoring
    print(f"📊 Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    print(f"📊 Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, Learning rate: {LEARNING_RATE}")
    
    trainer.train()
    
    # Clear GPU memory after training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Save model and tokenizer
    model_path = "./checkpoints/clickbait_classifier"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    print(f"✅ Model saved to {model_path}")
    
    # Evaluate on validation set
    print("📊 Evaluating model...")
    
    # Evaluate baseline model
    print("\n🔍 Evaluating baseline model...")
    baseline_scores = [0.5] * len(val_texts)  # Simple baseline
    
    # Convert to binary for classification metrics - use balanced threshold
    baseline_binary = (np.array(baseline_scores) > 0.5).astype(int)
    true_binary_baseline = (np.array(val_labels) > 0.5).astype(int)
    
    # Check if we have both classes for classification report
    if len(np.unique(true_binary_baseline)) > 1 and len(np.unique(baseline_binary)) > 1:
        baseline_report = classification_report(true_binary_baseline, baseline_binary, 
                                             target_names=["Not Clickbait", "Clickbait"])
        print("📊 Baseline Classification Report:")
        print(baseline_report)
    else:
        print("⚠️ Baseline evaluation skipped: insufficient class diversity")
        print(f"   True labels unique values: {np.unique(true_binary_baseline)}")
        print(f"   Baseline predictions unique values: {np.unique(baseline_binary)}")
    
    # Evaluate trained model
    print("\n🔍 Evaluating trained model...")
    model.eval()
    pred_scores = []
    
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            scores = outputs.logits.squeeze().cpu().numpy()
            pred_scores.extend(scores)
    
    # Convert to binary for classification metrics - use balanced threshold
    pred_binary = (np.array(pred_scores) > 0.5).astype(int)
    true_binary = (np.array(val_labels) > 0.5).astype(int)
    
    # Check if we have both classes for classification report
    if len(np.unique(true_binary)) > 1 and len(np.unique(pred_binary)) > 1:
        model_report = classification_report(true_binary, pred_binary, 
                                          target_names=["Not Clickbait", "Clickbait"])
        print("📊 Model Classification Report:")
        print(model_report)
    else:
        print("⚠️ Model evaluation skipped: insufficient class diversity")
        print(f"   True labels unique values: {np.unique(true_binary)}")
        print(f"   Model predictions unique values: {np.unique(pred_binary)}")
    
    # Always show basic statistics
    print(f"\n📊 Model Performance Summary:")
    print(f"   True labels distribution: {np.bincount(true_binary)}")
    print(f"   Predictions distribution: {np.bincount(pred_binary)}")
    print(f"   Mean predicted score: {np.mean(pred_scores):.3f}")
    print(f"   Mean true score: {np.mean(val_labels):.3f}")
    print(f"   Score correlation: {np.corrcoef(pred_scores, val_labels)[0,1]:.3f}")
    
    # Show some example predictions
    print(f"\n📝 Example Predictions:")
    for i in range(min(5, len(val_texts))):
        text_preview = val_texts[i][:100] + "..." if len(val_texts[i]) > 100 else val_texts[i]
        print(f"   Text: {text_preview}")
        print(f"   True: {val_labels[i]:.3f}, Predicted: {pred_scores[i]:.3f}")
        print()
    
    return model, tokenizer

def main():
    """Main training function."""
    print("🎯 Clickbait Headline Classifier Training")
    print("=" * 50)
    
    # Create database connection
    engine = create_database_connection()
    
    # Load corpus data
    popular_entities = load_popular_entities(engine)
    colloquial_phrases = load_colloquial_phrases(engine)
    
    # Load training data
    texts, labels = load_training_data(engine, popular_entities, colloquial_phrases)
    
    # Train model
    model, tokenizer = train_model(texts, labels, popular_entities, colloquial_phrases)
    
    print("\n🎉 Training completed successfully!")
    print("📁 Model saved to: ./checkpoints/clickbait_classifier")
    print("🔧 Use the standalone scorer script to test the model")

if __name__ == "__main__":
    main()
