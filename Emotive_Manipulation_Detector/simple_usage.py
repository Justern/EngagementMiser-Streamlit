#!/usr/bin/env python3
"""
Simple Usage Example for Emotive Manipulation Detector
Input: Tweet ID or direct text
Output: Single score from 0-1
"""

from emotive_manipulation_detector import EmotiveManipulationDetector
import os

def analyze_text_direct(text: str) -> float:
    """
    Analyze text directly and return manipulation score.
    
    Args:
        text: Text to analyze
        
    Returns:
        float: Manipulation score from 0-1
    """
    try:
        # Try to load pre-trained model first
        model_path = "emotive_manipulation_model.joblib"
        if os.path.exists(model_path):
            detector = EmotiveManipulationDetector(model_path)
            print("✅ Loaded pre-trained model")
        else:
            # Use untrained model with rule-based scoring
            detector = EmotiveManipulationDetector()
            print("⚠️  No pre-trained model found, using rule-based scoring")
        
        print(f"✅ Analyzing text: {text[:50]}...")
        
        # Get the score
        if detector.is_trained:
            results = detector.analyze_text(text)
            score = results['manipulation_score']
            source = results.get('score_source', 'unknown')
        else:
            results = detector.analyze_text(text)
            score = results['manipulation_score']
            source = 'rule_based'
        
        print(f"📊 Manipulation Score: {score:.3f}")
        print(f"🔍 Score Source: {source}")
        
        # Interpret the score
        if score >= 0.7:
            interpretation = "HIGH - Strong evidence of emotional manipulation"
        elif score >= 0.5:
            interpretation = "MEDIUM - Moderate evidence of emotional manipulation"
        elif score >= 0.3:
            interpretation = "LOW - Some evidence of emotional manipulation"
        else:
            interpretation = "MINIMAL - No significant manipulation detected"
        
        print(f"🔍 Interpretation: {interpretation}")
        
        # Show score source explanation
        if source == 'rule_based_override':
            print(f"💡 Note: Rule-based scoring used due to ML model underperformance")
        elif source == 'ml_model':
            print(f"💡 Note: ML model prediction used")
        elif source == 'rule_based':
            print(f"💡 Note: Rule-based scoring used (no trained model)")
        
        return score
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 0.0

def analyze_tweet_by_id(tweet_id: str) -> float:
    """
    Analyze a tweet by ID (placeholder for database integration).
    
    Args:
        tweet_id: Tweet ID to analyze
        
    Returns:
        float: Manipulation score from 0-1
    """
    print(f"🔍 Tweet ID: {tweet_id}")
    print("⚠️  Note: Database integration not implemented yet")
    print("   For now, please input the tweet text directly")
    print("   Or train the model first using: python train_model.py")
    
    return 0.0

def main():
    """Main function for simple analysis."""
    print("EMOTIVE MANIPULATION DETECTOR - SIMPLE ANALYSIS")
    print("=" * 60)
    print("Input: Tweet text or tweet ID")
    print("Output: Single score from 0-1")
    print("=" * 60)
    
    print("\nChoose input method:")
    print("1. Enter tweet text directly")
    print("2. Enter tweet ID (requires database setup)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        text = input("Enter tweet text to analyze: ").strip()
        if text:
            score = analyze_text_direct(text)
            print(f"\n🎯 Final Result: {score:.3f}")
        else:
            print("No text provided.")
    
    elif choice == "2":
        tweet_id = input("Enter tweet ID to analyze: ").strip()
        if tweet_id:
            score = analyze_tweet_by_id(tweet_id)
            print(f"\n🎯 Final Result: {score:.3f}")
        else:
            print("No tweet ID provided.")
    
    else:
        print("Invalid choice. Please run again and select 1 or 2.")

if __name__ == "__main__":
    main()
