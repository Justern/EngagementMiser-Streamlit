#!/usr/bin/env python3
"""
Analyze the annotated Generic Comment dataset
"""

import pandas as pd
import numpy as np

def analyze_dataset():
    """Analyze the annotated dataset structure and distribution."""
    
    # Load the dataset
    df = pd.read_csv('Annotate export - Generic Comment.csv')
    
    print("=" * 60)
    print("GENERIC COMMENT DATASET ANALYSIS")
    print("=" * 60)
    
    # Basic info
    print(f"Total tweets: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Score analysis
    scores = df['label confidence']
    print(f"Score range: {scores.min():.1f} - {scores.max():.1f}")
    print(f"Mean score: {scores.mean():.3f}")
    print(f"Median score: {scores.median():.3f}")
    print(f"Standard deviation: {scores.std():.3f}")
    print()
    
    # Score distribution
    print("Score distribution:")
    score_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(score_bins) - 1):
        count = len(df[(scores >= score_bins[i]) & (scores < score_bins[i+1])])
        print(f"  {score_bins[i]:.1f}-{score_bins[i+1]:.1f}: {count:3d} tweets")
    
    print()
    
    # Examples by score level
    print("EXAMPLES BY QUALITY LEVEL:")
    print()
    
    # Low quality (0.0-0.2)
    low_quality = df[scores <= 0.2].sample(min(3, len(df[scores <= 0.2])))
    print("LOW QUALITY (0.0-0.2):")
    for _, row in low_quality.iterrows():
        print(f"  Score: {row['label confidence']:.1f}")
        print(f"  Text: {row['text'][:100]}...")
        print()
    
    # Medium quality (0.4-0.6)
    med_quality = df[(scores >= 0.4) & (scores <= 0.6)].sample(min(3, len(df[(scores >= 0.4) & (scores <= 0.6)])))
    print("MEDIUM QUALITY (0.4-0.6):")
    for _, row in med_quality.iterrows():
        print(f"  Score: {row['label confidence']:.1f}")
        print(f"  Text: {row['text'][:100]}...")
        print()
    
    # High quality (0.7-1.0)
    high_quality = df[scores >= 0.7].sample(min(3, len(df[scores >= 0.7])))
    print("HIGH QUALITY (0.7-1.0):")
    for _, row in high_quality.iterrows():
        print(f"  Score: {row['label confidence']:.1f}")
        print(f"  Text: {row['text'][:100]}...")
        print()
    
    # Text length analysis
    df['text_length'] = df['text'].str.len()
    print(f"Text length - Mean: {df['text_length'].mean():.1f} chars, Median: {df['text_length'].median():.1f} chars")
    print()
    
    # Source analysis
    print("Top sources:")
    print(df['source'].value_counts().head(5))
    print()
    
    return df

if __name__ == "__main__":
    df = analyze_dataset()
