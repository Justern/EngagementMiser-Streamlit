#!/usr/bin/env python3
"""
Debug Predictions Script for Hyperbole & Falsehood Detector
=========================================================

This script helps debug why the model might be making poor predictions.
"""

import os
import sys
import numpy as np
import pandas as pd
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

def load_sample_data(engine, n_samples=10):
    """Load a small sample of data for debugging."""
    print("🔄 Loading sample data...")
    
    query = """
    SELECT TOP 10 tweet_id, text, label_confidence
    FROM [EngagementMiser].[dbo].[Hyperbole_Falsehood_tweets_annot]
    WHERE text IS NOT NULL 
    AND LEN(text) > 10 
    AND label_confidence IS NOT NULL
    ORDER BY NEWID()
    """
    
    try:
        df = pd.read_sql_query(sql_text(query), engine)
        print(f"✅ Loaded {len(df)} sample tweets")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def predict_single(text, model, tokenizer, device, max_len=128):
    """Make prediction on a single text."""
    with torch.no_grad():
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
        raw_logits = logits.squeeze().item()
        score = torch.sigmoid(logits.squeeze()).item()
        
        return raw_logits, score

def debug_predictions(df, model, tokenizer, device):
    """Debug predictions on sample data."""
    print("\n🔍 DEBUGGING PREDICTIONS")
    print("=" * 50)
    
    results = []
    
    for idx, row in df.iterrows():
        tweet_id = row['tweet_id']
        text = row['text']
        true_label = row['label_confidence']
        binary_label = 1 if true_label >= 0.5 else 0
        
        # Get prediction
        raw_logit, prob_score = predict_single(text, model, tokenizer, device)
        predicted_label = 1 if prob_score >= 0.5 else 0
        
        # Store results
        results.append({
            'tweet_id': tweet_id,
            'text_preview': text[:100] + "..." if len(text) > 100 else text,
            'true_label': true_label,
            'binary_label': binary_label,
            'raw_logit': raw_logit,
            'prob_score': prob_score,
            'predicted_label': predicted_label,
            'correct': binary_label == predicted_label
        })
        
        # Print individual results
        print(f"\n📱 Tweet {idx+1}:")
        print(f"   ID: {tweet_id}")
        print(f"   Text: {text[:80]}...")
        print(f"   True Label: {true_label:.3f} (binary: {binary_label})")
        print(f"   Raw Logit: {raw_logit:.3f}")
        print(f"   Probability: {prob_score:.3f}")
        print(f"   Predicted: {predicted_label}")
        print(f"   Correct: {'✅' if binary_label == predicted_label else '❌'}")
    
    # Summary
    results_df = pd.DataFrame(results)
    print(f"\n📊 SUMMARY:")
    print(f"   Total tweets: {len(results_df)}")
    print(f"   Correct predictions: {results_df['correct'].sum()}")
    print(f"   Accuracy: {results_df['correct'].mean():.3f}")
    
    print(f"\n📈 Score Distribution:")
    print(f"   Min probability: {results_df['prob_score'].min():.3f}")
    print(f"   Max probability: {results_df['prob_score'].max():.3f}")
    print(f"   Mean probability: {results_df['prob_score'].mean():.3f}")
    print(f"   Std probability: {results_df['prob_score'].std():.3f}")
    
    print(f"\n🔍 Raw Logit Distribution:")
    print(f"   Min logit: {results_df['raw_logit'].min():.3f}")
    print(f"   Max logit: {results_df['raw_logit'].max():.3f}")
    print(f"   Mean logit: {results_df['raw_logit'].mean():.3f}")
    print(f"   Std logit: {results_df['raw_logit'].std():.3f}")
    
    return results_df

def main():
    """Main debug function."""
    print("🔍 Hyperbole & Falsehood Detector - Debug Predictions")
    print("=" * 60)
    
    try:
        # Setup database
        engine = setup_database()
        if engine is None:
            return
        
        # Load model
        model, tokenizer, device = load_model()
        
        # Load sample data
        df = load_sample_data(engine)
        if df is None:
            print("❌ Failed to load sample data")
            return
        
        # Debug predictions
        results = debug_predictions(df, model, tokenizer, device)
        
        print("\n✅ Debug completed successfully!")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


