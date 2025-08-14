#!/usr/bin/env python3
"""
Calibrated Evaluation Script for Hyperbole & Falsehood Detector
==============================================================

This script addresses the model bias issue by:
1. Finding optimal classification thresholds
2. Providing realistic performance metrics
3. Showing performance at different threshold levels

Usage: python calibrated_evaluate.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import train_test_split
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

def find_optimal_threshold(y_true, y_pred_proba):
    """Find optimal threshold for classification."""
    print("🔄 Finding optimal threshold...")
    
    thresholds = np.arange(0.1, 1.0, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            }
    
    print(f"✅ Optimal threshold: {best_threshold:.3f}")
    print(f"   F1-Score: {best_f1:.4f}")
    print(f"   Precision: {best_metrics['precision']:.4f}")
    print(f"   Recall: {best_metrics['recall']:.4f}")
    print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
    
    return best_threshold, best_metrics

def evaluate_at_thresholds(y_true, y_pred_proba):
    """Evaluate model performance at different threshold levels."""
    print("\n🔄 Evaluating at different thresholds...")
    
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        })
        
        print(f"   Threshold {threshold:.2f}: F1={f1:.3f}, Prec={precision:.3f}, Rec={recall:.3f}")
    
    return pd.DataFrame(results)

def print_calibrated_metrics(results, optimal_threshold, optimal_metrics):
    """Print calibrated evaluation metrics."""
    print("\n" + "="*60)
    print("📊 CALIBRATED MODEL EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n🎯 Performance at Optimal Threshold ({optimal_threshold:.3f}):")
    print(f"   Accuracy:  {optimal_metrics['accuracy']:.4f}")
    print(f"   Precision: {optimal_metrics['precision']:.4f}")
    print(f"   Recall:    {optimal_metrics['recall']:.4f}")
    print(f"   F1-Score:  {optimal_metrics['f1']:.4f}")
    
    # Performance at default threshold (0.5)
    y_pred_default = (results['y_pred_proba'] >= 0.5).astype(int)
    default_metrics = {
        'accuracy': accuracy_score(results['y_true'], y_pred_default),
        'precision': precision_score(results['y_true'], y_pred_default, zero_division=0),
        'recall': recall_score(results['y_true'], y_pred_default, zero_division=0),
        'f1': f1_score(results['y_true'], y_pred_default, zero_division=0)
    }
    
    print(f"\n⚠️  Performance at Default Threshold (0.5):")
    print(f"   Accuracy:  {default_metrics['accuracy']:.4f}")
    print(f"   Precision: {default_metrics['precision']:.4f}")
    print(f"   Recall:    {default_metrics['recall']:.4f}")
    print(f"   F1-Score:  {default_metrics['f1']:.4f}")
    
    # Improvement
    f1_improvement = optimal_metrics['f1'] - default_metrics['f1']
    print(f"\n🚀 Improvement with Optimal Threshold:")
    print(f"   F1-Score improvement: +{f1_improvement:.4f}")
    print(f"   Relative improvement: {f1_improvement/default_metrics['f1']*100:.1f}%" if default_metrics['f1'] > 0 else "N/A")
    
    # Confusion matrix at optimal threshold
    y_pred_optimal = (results['y_pred_proba'] >= optimal_threshold).astype(int)
    cm = confusion_matrix(results['y_true'], y_pred_optimal)
    
    print(f"\n🔍 Confusion Matrix (Threshold {optimal_threshold:.3f}):")
    print(f"   Predicted")
    print(f"Actual    0    1")
    print(f"    0   {cm[0,0]:4d} {cm[0,1]:4d}")
    print(f"    1   {cm[1,0]:4d} {cm[1,1]:4d}")
    
    # Detailed classification report at optimal threshold
    print(f"\n📋 Detailed Classification Report (Threshold {optimal_threshold:.3f}):")
    print(classification_report(
        results['y_true'], 
        y_pred_optimal,
        target_names=['Truthful (0)', 'Hyperbole/False (1)']
    ))
    
    # Performance summary
    print(f"\n📈 Performance Summary:")
    if optimal_metrics['f1'] >= 0.8:
        performance = "Excellent"
    elif optimal_metrics['f1'] >= 0.7:
        performance = "Good"
    elif optimal_metrics['f1'] >= 0.6:
        performance = "Fair"
    else:
        performance = "Needs Improvement"
    
    print(f"   Overall Performance: {performance}")
    print(f"   Optimal F1-Score: {optimal_metrics['f1']:.4f}")
    print(f"   Recommended Threshold: {optimal_threshold:.3f}")

def main():
    """Main calibrated evaluation function."""
    print("🔍 Hyperbole & Falsehood Detector - Calibrated Evaluation")
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
        print("🔄 Running model evaluation...")
        test_texts = test_df['text'].tolist()
        y_pred_proba = predict_batch(test_texts, model, tokenizer, device)
        y_true = test_df['binary_label'].values
        
        # Find optimal threshold
        optimal_threshold, optimal_metrics = find_optimal_threshold(y_true, y_pred_proba)
        
        # Evaluate at different thresholds
        threshold_results = evaluate_at_thresholds(y_true, y_pred_proba)
        
        # Store results
        results = {
            'y_true': y_true,
            'y_pred_proba': y_pred_proba,
            'optimal_threshold': optimal_threshold,
            'optimal_metrics': optimal_metrics
        }
        
        # Print calibrated metrics
        print_calibrated_metrics(results, optimal_threshold, optimal_metrics)
        
        print("\n✅ Calibrated evaluation completed successfully!")
        print(f"💡 Recommendation: Use threshold {optimal_threshold:.3f} instead of 0.5")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


