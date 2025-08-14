#!/usr/bin/env python3
"""
Simple Evaluation Script for Hyperbole & Falsehood Detector
=========================================================

This script provides basic evaluation metrics including:
- Precision, Recall, F1-Score, Accuracy
- Confusion Matrix
- Simple performance summary

Usage: python simple_evaluate.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sqlalchemy import create_engine, text as sql_text

def load_model():
    """Load the trained model and tokenizer."""
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "text_softlabel_roberta")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    print("🔄 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"✅ Model loaded on {device}")
    
    return model, tokenizer, device

def setup_database():
    """Setup database connection."""
    SQL_SERVER = "localhost"
    SQL_DB = "EngagementMiser"
    SQL_DRIVER = "ODBC Driver 18 for SQL Server"
    
    CONN_STR = (
        f"mssql+pyodbc://@{SQL_SERVER}/{SQL_DB}"
        f"?driver={SQL_DRIVER.replace(' ', '+')}"
        "&Trusted_Connection=yes"
        "&TrustServerCertificate=yes"
    )
    
    try:
        engine = create_engine(CONN_STR)
        print("✅ Database connection established")
        return engine
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def load_evaluation_data(engine, test_size=0.2, random_state=42):
    """Load data for evaluation from the annotated dataset."""
    print("🔄 Loading evaluation data...")
    
    query = """
    SELECT tweet_id, text, label_confidence
    FROM [EngagementMiser].[dbo].[Hyperbole_Falsehood_tweets_annot]
    WHERE text IS NOT NULL 
    AND LEN(text) > 10 
    AND label_confidence IS NOT NULL
    """
    
    try:
        df = pd.read_sql_query(sql_text(query), engine)
        print(f"✅ Loaded {len(df)} annotated tweets")
        
        # Convert to binary labels (threshold at 0.5)
        df['binary_label'] = (df['label_confidence'] >= 0.5).astype(int)
        
        # Split into train/test
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, 
            stratify=df['binary_label']
        )
        
        print(f"📊 Train set: {len(train_df)} tweets")
        print(f"📊 Test set: {len(test_df)} tweets")
        print(f"📊 Class distribution in test set:")
        print(test_df['binary_label'].value_counts().sort_index())
        
        return test_df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def predict_batch(texts, model, tokenizer, device, max_len=128):
    """Make predictions on a batch of texts."""
    predictions = []
    
    with torch.no_grad():
        for text in texts:
            # Tokenize
            enc = tokenizer(
                str(text),
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            
            # Predict
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            score = torch.sigmoid(logits.squeeze()).item()
            predictions.append(score)
    
    return np.array(predictions)

def evaluate_model(test_df, model, tokenizer, device):
    """Evaluate the model and calculate all metrics."""
    print("🔄 Running model evaluation...")
    
    # Get predictions
    test_texts = test_df['text'].tolist()
    y_pred_proba = predict_batch(test_texts, model, tokenizer, device)
    y_true = test_df['binary_label'].values
    
    # Convert probabilities to binary predictions (threshold 0.5)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def print_metrics(results):
    """Print all evaluation metrics."""
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION RESULTS")
    print("="*60)
    
    # Basic metrics
    print(f"\n🎯 Basic Metrics:")
    print(f"   Accuracy:  {results['accuracy']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print(f"   F1-Score:  {results['f1']:.4f}")
    
    # Confusion matrix
    print(f"\n🔍 Confusion Matrix:")
    cm = results['confusion_matrix']
    print(f"   Predicted")
    print(f"Actual    0    1")
    print(f"    0   {cm[0,0]:4d} {cm[0,1]:4d}")
    print(f"    1   {cm[1,0]:4d} {cm[1,1]:4d}")
    
    # Interpretation
    print(f"\n💡 Interpretation:")
    print(f"   True Negatives (TN): {cm[0,0]} - Correctly identified truthful tweets")
    print(f"   False Positives (FP): {cm[0,1]} - Truthful tweets misclassified as hyperbole/false")
    print(f"   False Negatives (FN): {cm[1,0]} - Hyperbole/false tweets misclassified as truthful")
    print(f"   True Positives (TP): {cm[1,1]} - Correctly identified hyperbole/false tweets")
    
    # Detailed classification report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(
        results['y_true'], 
        results['y_pred'],
        target_names=['Truthful (0)', 'Hyperbole/False (1)']
    ))
    
    # Performance summary
    print(f"\n📈 Performance Summary:")
    if results['f1'] >= 0.8:
        performance = "Excellent"
    elif results['f1'] >= 0.7:
        performance = "Good"
    elif results['f1'] >= 0.6:
        performance = "Fair"
    else:
        performance = "Needs Improvement"
    
    print(f"   Overall Performance: {performance}")
    print(f"   F1-Score: {results['f1']:.4f}")
    print(f"   Balanced Accuracy: {(results['precision'] + results['recall']) / 2:.4f}")

def main():
    """Main evaluation function."""
    print("🔍 Hyperbole & Falsehood Detector - Simple Evaluation")
    print("=" * 60)
    
    try:
        # Setup database
        engine = setup_database()
        if engine is None:
            return
        
        # Load model
        model, tokenizer, device = load_model()
        
        # Load evaluation data
        test_df = load_evaluation_data(engine)
        if test_df is None:
            print("❌ Failed to load evaluation data")
            return
        
        # Run evaluation
        results = evaluate_model(test_df, model, tokenizer, device)
        
        # Print metrics
        print_metrics(results)
        
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


