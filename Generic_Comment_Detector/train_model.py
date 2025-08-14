#!/usr/bin/env python3
"""
Train the Generic Comment Detector using the annotated CSV data
"""

import pandas as pd
import numpy as np
from generic_comment_detector import GenericCommentDetector
import os

def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load and prepare the annotated dataset for training.
    
    Args:
        csv_path: Path to the annotated CSV file
        
    Returns:
        DataFrame: Prepared training data
    """
    print(f"Loading data from: {csv_path}")
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} tweets")
    print(f"Columns: {list(df.columns)}")
    
    # Check for required columns
    required_columns = ['text', 'label confidence']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Clean the data
    print("Cleaning data...")
    
    # Remove rows with missing text or labels
    df = df.dropna(subset=['text', 'label confidence'])
    
    # Ensure label confidence is numeric
    df['label confidence'] = pd.to_numeric(df['label confidence'], errors='coerce')
    df = df.dropna(subset=['label confidence'])
    
    # Filter out rows with invalid scores
    df = df[(df['label confidence'] >= 0.0) & (df['label confidence'] <= 1.0)]
    
    print(f"After cleaning: {len(df)} tweets")
    
    # Display score distribution
    scores = df['label confidence']
    print(f"\nScore Distribution:")
    print(f"  Range: {scores.min():.1f} - {scores.max():.1f}")
    print(f"  Mean: {scores.mean():.3f}")
    print(f"  Median: {scores.median():.3f}")
    print(f"  Std: {scores.std():.3f}")
    
    # Show examples by score level
    print(f"\nExamples by quality level:")
    
    # Low quality (0.0-0.2)
    low_quality = df[scores <= 0.2].sample(min(2, len(df[scores <= 0.2])))
    print(f"  LOW (0.0-0.2): {len(df[scores <= 0.2])} tweets")
    for _, row in low_quality.iterrows():
        print(f"    Score: {row['label confidence']:.1f} - {row['text'][:80]}...")
    
    # Medium quality (0.3-0.6)
    med_quality = df[(scores >= 0.3) & (scores <= 0.6)].sample(min(2, len(df[(scores >= 0.3) & (scores <= 0.6)])))
    print(f"  MEDIUM (0.3-0.6): {len(df[(scores >= 0.3) & (scores <= 0.6)])} tweets")
    for _, row in med_quality.iterrows():
        print(f"    Score: {row['label confidence']:.1f} - {row['text'][:80]}...")
    
    # High quality (0.7-1.0)
    high_quality = df[scores >= 0.7].sample(min(2, len(df[scores >= 0.7])))
    print(f"  HIGH (0.7-1.0): {len(df[scores >= 0.7])} tweets")
    for _, row in high_quality.iterrows():
        print(f"    Score: {row['label confidence']:.1f} - {row['text'][:80]}...")
    
    return df

def train_and_evaluate_model(training_data: pd.DataFrame, model_save_path: str = None) -> GenericCommentDetector:
    """
    Train and evaluate the Generic Comment Detector.
    
    Args:
        training_data: Prepared training data
        model_save_path: Path to save the trained model
        
    Returns:
        GenericCommentDetector: Trained model
    """
    print("\n" + "="*60)
    print("TRAINING GENERIC COMMENT DETECTOR")
    print("="*60)
    
    # Initialize detector
    detector = GenericCommentDetector()
    
    # Train the model
    training_results = detector.train_model(training_data)
    
    print(f"\nTraining Results:")
    print(f"  R² Score: {training_results['r2']:.3f}")
    print(f"  RMSE: {training_results['rmse']:.3f}")
    print(f"  Training samples: {training_results['training_samples']}")
    print(f"  Test samples: {training_results['test_samples']}")
    
    # Test on some examples
    print(f"\nTesting on training examples:")
    test_samples = training_data.sample(min(5, len(training_data)))
    
    for _, row in test_samples.iterrows():
        actual_score = row['label confidence']
        predicted_score = detector.predict_content_quality_score(row['text'])
        error = abs(actual_score - predicted_score)
        
        print(f"  Actual: {actual_score:.3f}, Predicted: {predicted_score:.3f}, Error: {error:.3f}")
        print(f"    Text: {row['text'][:60]}...")
        print()
    
    # Save model if path provided
    if model_save_path:
        detector.save_model(model_save_path)
        print(f"Model saved to: {model_save_path}")
    
    return detector

def main():
    """Main training function."""
    print("GENERIC COMMENT DETECTOR - TRAINING")
    print("=" * 50)
    
    # File paths
    csv_path = "Annotate export - Generic Comment.csv"
    model_save_path = "generic_comment_model.joblib"
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV file not found at {csv_path}")
        print("Please ensure the annotated CSV file is in the current directory.")
        return
    
    try:
        # Load and prepare data
        training_data = load_and_prepare_data(csv_path)
        
        # Train and evaluate model
        detector = train_and_evaluate_model(training_data, model_save_path)
        
        print("\n✅ Training completed successfully!")
        print(f"\nModel is ready for use!")
        print(f"To analyze new text:")
        print(f"  score = detector.predict_content_quality_score('your text here')")
        print(f"  results = detector.analyze_text('your text here')")
        
        # Test the trained model
        print(f"\n" + "="*50)
        print("TESTING TRAINED MODEL")
        print("="*50)
        
        test_texts = [
            "This is a generic response that adds no value to the conversation.",
            "Great analysis! The data clearly shows that the correlation coefficient of 0.87 indicates a strong positive relationship between these variables.",
            "nice",
            "Thanks for sharing this insightful research. The methodology appears sound, and the conclusions align with recent findings in the field."
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}:")
            print(f"Text: {text}")
            
            try:
                score = detector.predict_content_quality_score(text)
                results = detector.analyze_text(text)
                
                print(f"Quality Score: {score:.3f}")
                print(f"Quality Level: {results['quality_level']}")
                
                # Show detected patterns
                patterns = results['pattern_analysis']
                if patterns:
                    print("Patterns detected:")
                    for pattern_type, pattern_data in patterns.items():
                        print(f"  {pattern_type}: {pattern_data['count']} indicators")
                
            except Exception as e:
                print(f"Error: {str(e)}")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
