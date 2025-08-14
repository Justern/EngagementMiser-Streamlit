#!/usr/bin/env python3
"""
Test the hybrid approach for the Generic Comment Detector
"""

from generic_comment_detector import GenericCommentDetector

def test_hybrid_approach():
    """Test the hybrid approach with trained ML model."""
    
    print("=" * 70)
    print("TESTING HYBRID APPROACH (ML + Rule-based)")
    print("=" * 70)
    
    # Load trained model
    detector = GenericCommentDetector('generic_comment_model.joblib')
    print("✅ Loaded trained ML model")
    
    test_texts = [
        "This is a generic response that adds no value to the conversation.",
        "nice",
        "Great analysis! The data clearly shows that the correlation coefficient of 0.87 indicates a strong positive relationship between these variables.",
        "Thanks for sharing this insightful research. The methodology appears sound, and the conclusions align with recent findings in the field."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print(f"Text: {text}")
        
        try:
            results = detector.analyze_text(text)
            score = results['content_quality_score']
            quality = results['quality_level']
            source = results.get('score_source', 'unknown')
            
            print(f"Quality Score: {score:.3f}")
            print(f"Quality Level: {quality}")
            print(f"Score Source: {source}")
            
            # Show detected patterns
            patterns = results['pattern_analysis']
            if patterns:
                print("Patterns detected:")
                for pattern_type, pattern_data in patterns.items():
                    print(f"  {pattern_type}: {pattern_data['count']} indicators")
            
            # Show key features
            features = results['features']
            print(f"Key Features:")
            print(f"  Lexical Diversity: {features.get('lexical_diversity', 0):.3f}")
            print(f"  Generic Phrase Density: {features.get('generic_phrase_density', 0):.3f}")
            print(f"  High-Quality Indicators: {features.get('high_quality_indicator_count', 0)}")
            print(f"  Overall Quality Score: {features.get('overall_quality_score', 0):.3f}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("-" * 70)

if __name__ == "__main__":
    test_hybrid_approach()
