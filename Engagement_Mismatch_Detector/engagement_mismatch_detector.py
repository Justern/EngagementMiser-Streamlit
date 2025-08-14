#!/usr/bin/env python3
"""
Engagement Mismatch Detector
============================

This model detects tweets with unexpectedly high engagement relative to the account's
follower count, which could indicate viral content, manipulation, or interesting anomalies.

Features:
- Z-score analysis for engagement metrics
- Distribution modeling for follower counts
- Multiple statistical detection methods
- Engagement rate normalization
- Outlier detection using IQR and percentile methods

Usage:
    python engagement_mismatch_detector.py
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text as sql_text
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Union, Tuple, List, Dict, Optional
import re
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
MODEL_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "engagement_mismatch_detector")
MAX_LEN = 128
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

# Model hyperparameters
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 4

class EngagementDataset(torch.utils.data.Dataset):
    """Dataset for engagement mismatch detection."""
    
    def __init__(self, texts: List[str], labels: List[float], tokenizer, max_len: int):
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
        # Test connection
        with engine.connect() as conn:
            conn.execute(sql_text("SELECT 1"))
        print("✅ Database connection established")
        return engine
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)

def calculate_engagement_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive engagement metrics and rates."""
    print("📊 Calculating engagement metrics...")
    
    # Basic engagement metrics
    df['total_engagement'] = df['like_count'] + df['retweet_count'] + df['reply_count'] + df['quote_count']
    df['engagement_rate'] = df['total_engagement'] / df['followers_count'].replace(0, 1)
    
    # Normalized engagement rates (per 1000 followers)
    df['engagement_per_1k'] = (df['total_engagement'] / df['followers_count'].replace(0, 1)) * 1000
    
    # Individual engagement type rates
    df['like_rate'] = df['like_count'] / df['followers_count'].replace(0, 1)
    df['retweet_rate'] = df['retweet_count'] / df['followers_count'].replace(0, 1)
    df['reply_rate'] = df['reply_count'] / df['followers_count'].replace(0, 1)
    df['quote_rate'] = df['quote_count'] / df['followers_count'].replace(0, 1)
    
    # Engagement diversity (how spread out engagement is across types)
    df['engagement_diversity'] = df[['like_rate', 'retweet_rate', 'reply_rate', 'quote_rate']].std(axis=1)
    
    # Viral coefficient (retweets + quotes) / total engagement
    df['viral_coefficient'] = (df['retweet_count'] + df['quote_count']) / df['total_engagement'].replace(0, 1)
    
    return df

def detect_engagement_mismatches(df: pd.DataFrame) -> pd.DataFrame:
    """Detect engagement mismatches using multiple statistical methods."""
    print("🔍 Detecting engagement mismatches...")
    
    # Method 1: Z-score analysis for engagement rates
    print("   📈 Method 1: Z-score analysis...")
    engagement_metrics = ['engagement_rate', 'engagement_per_1k', 'like_rate', 'retweet_rate', 'reply_rate', 'quote_rate']
    
    for metric in engagement_metrics:
        if metric in df.columns:
            # Remove infinite values and calculate Z-scores
            clean_metric = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean_metric) > 0:
                z_scores = np.abs(stats.zscore(clean_metric))
                df[f'{metric}_zscore'] = pd.Series(z_scores, index=clean_metric.index)
    
    # Method 2: Percentile-based outlier detection
    print("   📊 Method 2: Percentile analysis...")
    for metric in engagement_metrics:
        if metric in df.columns:
            clean_metric = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean_metric) > 0:
                p95 = np.percentile(clean_metric, 95)
                p99 = np.percentile(clean_metric, 99)
                df[f'{metric}_p95'] = clean_metric > p95
                df[f'{metric}_p99'] = clean_metric > p99
    
    # Method 3: IQR-based outlier detection
    print("   📏 Method 3: IQR analysis...")
    for metric in engagement_metrics:
        if metric in df.columns:
            clean_metric = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean_metric) > 0:
                Q1 = np.percentile(clean_metric, 25)
                Q3 = np.percentile(clean_metric, 75)
                IQR = Q3 - Q1
                upper_bound = Q3 + 1.5 * IQR
                df[f'{metric}_iqr_outlier'] = clean_metric > upper_bound
    
    # Method 4: Follower count vs engagement analysis
    print("   👥 Method 4: Follower-engagement analysis...")
    df['follower_engagement_ratio'] = df['total_engagement'] / df['followers_count'].replace(0, 1)
    
    # Calculate expected engagement based on follower count
    follower_bins = pd.cut(df['followers_count'], bins=10, labels=False)
    expected_engagement = df.groupby(follower_bins)['total_engagement'].transform('mean')
    df['engagement_deviation'] = (df['total_engagement'] - expected_engagement) / expected_engagement.replace(0, 1)
    
    # Method 5: Isolation Forest for anomaly detection
    print("   🌲 Method 5: Isolation Forest analysis...")
    try:
        # Prepare features for anomaly detection
        anomaly_features = df[['engagement_rate', 'engagement_per_1k', 'viral_coefficient', 'engagement_diversity']].copy()
        anomaly_features = anomaly_features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Normalize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(anomaly_features)
        
        # Apply Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        df['isolation_forest_score'] = iso_forest.fit_predict(scaled_features)
        
    except Exception as e:
        print(f"   ⚠️ Isolation Forest failed: {e}")
        df['isolation_forest_score'] = 0
    
    return df

def create_mismatch_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create labels for engagement mismatch detection."""
    print("🏷️ Creating mismatch labels...")
    
    scores = []
    for idx, row in df.iterrows():
        score = 0.0
        
        # Base score from engagement rate
        if pd.notna(row.get('engagement_rate', np.nan)):
            if row['engagement_rate'] > 0.1: score += 0.3
            elif row['engagement_rate'] > 0.05: score += 0.2
            elif row['engagement_rate'] > 0.02: score += 0.1
        
        # Z-score contributions
        for metric in ['engagement_rate', 'engagement_per_1k', 'like_rate', 'retweet_rate']:
            zscore_col = f'{metric}_zscore'
            if zscore_col in df.columns and pd.notna(row.get(zscore_col, np.nan)):
                if row[zscore_col] > 3: score += 0.2
                elif row[zscore_col] > 2: score += 0.15
                elif row[zscore_col] > 1.5: score += 0.1
        
        # Percentile contributions
        for metric in ['engagement_rate', 'engagement_per_1k']:
            p99_col = f'{metric}_p99'
            p95_col = f'{metric}_p95'
            if p99_col in df.columns and row.get(p99_col, False): score += 0.2
            elif p95_col in df.columns and row.get(p95_col, False): score += 0.15
        
        # IQR outlier contributions
        for metric in ['engagement_rate', 'engagement_per_1k']:
            iqr_col = f'{metric}_iqr_outlier'
            if iqr_col in df.columns and row.get(iqr_col, False): score += 0.15
        
        # Follower-engagement ratio analysis
        if pd.notna(row.get('engagement_deviation', np.nan)):
            if row['engagement_deviation'] > 2: score += 0.2
            elif row['engagement_deviation'] > 1: score += 0.15
            elif row['engagement_deviation'] > 0.5: score += 0.1
        
        # Viral coefficient contribution
        if pd.notna(row.get('viral_coefficient', np.nan)):
            if row['viral_coefficient'] > 0.5: score += 0.15
            elif row['viral_coefficient'] > 0.3: score += 0.1
        
        # Isolation Forest contribution
        if 'isolation_forest_score' in df.columns:
            if row['isolation_forest_score'] == -1: score += 0.2
        
        # Cap score at 1.0
        score = min(score, 1.0)
        scores.append(score)
    
    df['mismatch_score'] = scores
    
    # Print score distribution
    score_dist = df['mismatch_score'].value_counts(bins=10).sort_index()
    print("📊 Mismatch score distribution:")
    for bin_range, count in score_dist.items():
        print(f"   {bin_range}: {count} samples")
    
    return df

def load_training_data(engine, sample_size: int = 15000) -> Tuple[List[str], List[float]]:
    """Load training data with engagement metrics."""
    print("📊 Loading training data...")
    
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
            AND (like_count + retweet_count + reply_count + quote_count) > 0
        ORDER BY CHECKSUM(NEWID())
        """
    )
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    print(f"📈 Loaded {len(df)} tweets")
    
    # Calculate engagement metrics
    df = calculate_engagement_metrics(df)
    
    # Detect mismatches
    df = detect_engagement_mismatches(df)
    
    # Create labels
    df = create_mismatch_labels(df)
    
    # Filter and prepare data
    texts = df['clean_text'].fillna(df['text']).tolist()
    labels = df['mismatch_score'].tolist()
    
    # Balance the dataset
    filtered_data = []
    high_score_count = 0
    low_score_count = 0
    medium_score_count = 0
    
    for text, label in zip(texts, labels):
        if len(str(text).strip()) > 20 and label >= 0.0:
            if label > 0.6 and high_score_count < 2000:
                filtered_data.append((text, label))
                high_score_count += 1
            elif label < 0.4 and low_score_count < 2000:
                filtered_data.append((text, label))
                low_score_count += 1
            elif 0.4 <= label <= 0.6 and medium_score_count < 1500:
                filtered_data.append((text, label))
                medium_score_count += 1
    
    if high_score_count < 1500:
        # Augment high-score samples
        high_score_texts = [text for text, label in filtered_data if label > 0.6]
        high_score_labels = [label for text, label in filtered_data if label > 0.6]
        
        for i in range(min(len(high_score_texts), 500)):
            text = high_score_texts[i]
            label = high_score_labels[i]
            # Simple augmentation
            if '!' in text: filtered_data.append((text.replace('!', '!!!'), min(label + 0.05, 1.0)))
            if '?' in text: filtered_data.append((text + " This is huge!", min(label + 0.05, 1.0)))
    
    texts, labels = zip(*filtered_data)
    
    print(f"✅ Prepared {len(texts)} training samples")
    print(f"📊 Label distribution: {np.mean(labels):.3f} mean, {np.std(labels):.3f} std")
    print(f"📊 High score samples (>0.6): {sum(1 for l in labels if l > 0.6)}")
    print(f"📊 Low score samples (<0.4): {sum(1 for l in labels if l < 0.4)}")
    print(f"📊 Medium score samples (0.4-0.6): {sum(1 for l in labels if 0.4 <= l <= 0.6)}")
    
    return list(texts), list(labels)

def train_model(train_texts: List[str], train_labels: List[float], 
                val_texts: List[str], val_labels: List[float], tokenizer) -> None:
    """Train the engagement mismatch detection model."""
    print("🚀 Training engagement mismatch detection model...")
    
    # Create datasets
    train_dataset = EngagementDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = EngagementDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=1,
        problem_type="regression"
    )
    model.to(DEVICE)
    
    # Training arguments
    training_args = {
        'output_dir': MODEL_PATH,
        'num_train_epochs': EPOCHS,
        'per_device_train_batch_size': BATCH_SIZE,
        'per_device_eval_batch_size': BATCH_SIZE,
        'warmup_steps': WARMUP_STEPS,
        'weight_decay': WEIGHT_DECAY,
        'logging_dir': os.path.join(MODEL_PATH, 'logs'),
        'logging_steps': 100,
        'save_steps': 500,
        'eval_steps': 500,
        'evaluation_strategy': 'steps',
        'save_strategy': 'steps',
        'load_best_model_at_end': True,
        'metric_for_best_model': 'eval_loss',
        'greater_is_better': False,
        'fp16': True,
        'dataloader_num_workers': 0,
        'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
        'hidden_dropout_prob': 0.3,
        'attention_probs_dropout_prob': 0.3
    }
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=EPOCHS)
    
    # Training loop
    model.train()
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\n📚 Epoch {epoch + 1}/{EPOCHS}")
        
        # Training
        model.train()
        total_train_loss = 0
        train_steps = 0
        
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            if GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            loss.backward()
            
            if (train_steps + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_train_loss += loss.item()
            train_steps += 1
            
            if train_steps % 100 == 0:
                print(f"   Step {train_steps}, Loss: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / train_steps
        print(f"   📊 Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                total_val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = total_val_loss / val_steps
        print(f"   📊 Average validation loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"   💾 New best model! Saving...")
            model.save_pretrained(MODEL_PATH)
            tokenizer.save_pretrained(MODEL_PATH)
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n✅ Training completed! Best validation loss: {best_val_loss:.4f}")
    print(f"💾 Model saved to: {MODEL_PATH}")

def main():
    """Main function to run the engagement mismatch detector training."""
    print("🚀 Engagement Mismatch Detector Training")
    print("=" * 60)
    
    # Create database connection
    engine = create_database_connection()
    
    # Load training data
    texts, labels = load_training_data(engine)
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 Training set: {len(train_texts)} samples")
    print(f"📊 Validation set: {len(val_texts)} samples")
    
    # Initialize tokenizer
    print("\n🔧 Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    # Train model
    train_model(train_texts, train_labels, val_texts, val_labels, tokenizer)
    
    print("\n🎉 Engagement Mismatch Detector training completed!")
    print(f"📁 Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
