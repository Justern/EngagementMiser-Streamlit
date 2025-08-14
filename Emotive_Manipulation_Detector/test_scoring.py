#!/usr/bin/env python3
"""
Test the enhanced rule-based scoring and hybrid approach for high-manipulation texts
"""

from emotive_manipulation_detector import EmotiveManipulationDetector
import os

def test_hybrid_approach():
    """Test the hybrid approach with trained ML model."""
    
    print("=" * 70)
    print("TESTING HYBRID APPROACH (ML + Rule-based)")
    print("=" * 70)
    
    # Try to load trained model
    model_path = "emotive_manipulation_model.joblib"
    if os.path.exists(model_path):
        detector = EmotiveManipulationDetector(model_path)
        print("✅ Loaded trained ML model")
    else:
        detector = EmotiveManipulationDetector()
        print("⚠️  No trained model found, using rule-based only")
    
    test_texts = [
        "URGENT! Don't miss this LIMITED TIME offer! Act NOW!",
        "You MUST buy this NOW! Everyone else is doing it!",
        "This is a normal informative tweet about current events.",
        "DON'T WAIT! This is your FINAL CHANCE! You'll REGRET missing this!",
        "SCARED? You SHOULD be! This is CRITICAL! ACT NOW!"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print(f"Text: {text}")
        
        try:
            results = detector.analyze_text(text)
            score = results['manipulation_score']
            risk = results['risk_level']
            source = results.get('score_source', 'unknown')
            
            print(f"Manipulation Score: {score:.3f}")
            print(f"Risk Level: {risk}")
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
            print(f"  Total Patterns: {features.get('total_patterns', 0)}")
            print(f"  Pressure Count: {features.get('pressure_count', 0)}")
            print(f"  Urgency Count: {features.get('urgency_count', 0)}")
            print(f"  Exclamation Count: {features.get('exclamation_density', 0):.3f}")
            print(f"  Caps Intensity: {features.get('caps_intensity', 0):.3f}")
            print(f"  Overall Intensity: {features.get('overall_manipulation_intensity', 0):.3f}")
            
            # Score interpretation
            if score >= 0.7:
                interpretation = "HIGH - Strong evidence of emotional manipulation"
            elif score >= 0.5:
                interpretation = "MEDIUM - Moderate evidence of emotional manipulation"
            elif score >= 0.3:
                interpretation = "LOW - Some evidence of emotional manipulation"
            else:
                interpretation = "MINIMAL - No significant manipulation detected"
            
            print(f"Interpretation: {interpretation}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("-" * 70)

def test_rule_based_only():
    """Test rule-based scoring without ML model."""
    
    print("=" * 70)
    print("TESTING RULE-BASED SCORING ONLY")
    print("=" * 70)
    
    detector = EmotiveManipulationDetector()
    
    test_texts = [
        "URGENT! Don't miss this LIMITED TIME offer! Act NOW!",
        "You MUST buy this NOW! Everyone else is doing it!",
        "This is a normal informative tweet about current events.",
        "DON'T WAIT! This is your FINAL CHANCE! You'll REGRET missing this!",
        "SCARED? You SHOULD be! This is CRITICAL! ACT NOW!"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print(f"Text: {text}")
        
        try:
            results = detector.analyze_text(text)
            score = results['manipulation_score']
            risk = results['risk_level']
            
            print(f"Manipulation Score: {score:.3f}")
            print(f"Risk Level: {risk}")
            
            # Show detected patterns
            patterns = results['pattern_analysis']
            if patterns:
                print("Patterns detected:")
                for pattern_type, pattern_data in patterns.items():
                    print(f"  {pattern_type}: {pattern_data['count']} indicators")
            
            # Show key features
            features = results['features']
            print(f"Key Features:")
            print(f"  Total Patterns: {features.get('total_patterns', 0)}")
            print(f"  Pressure Count: {features.get('pressure_count', 0)}")
            print(f"  Urgency Count: {features.get('urgency_count', 0)}")
            print(f"  Exclamation Count: {features.get('exclamation_density', 0):.3f}")
            print(f"  Caps Intensity: {features.get('caps_intensity', 0):.3f}")
            print(f"  Overall Intensity: {features.get('overall_manipulation_intensity', 0):.3f}")
            
            # Score interpretation
            if score >= 0.7:
                interpretation = "HIGH - Strong evidence of emotional manipulation"
            elif score >= 0.5:
                interpretation = "MEDIUM - Moderate evidence of emotional manipulation"
            elif score >= 0.3:
                interpretation = "LOW - Some evidence of emotional manipulation"
            else:
                interpretation = "MINIMAL - No significant manipulation detected"
            
            print(f"Interpretation: {interpretation}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("-" * 70)

def main():
    """Main test function."""
    print("EMOTIVE MANIPULATION DETECTOR - COMPREHENSIVE TESTING")
    print("=" * 70)
    
    # Test rule-based scoring first
    test_rule_based_only()
    
    print("\n" + "="*70)
    print("NOW TESTING HYBRID APPROACH")
    print("="*70)
    
    # Test hybrid approach
    test_hybrid_approach()

if __name__ == "__main__":
    main()
