"""
Emotive Manipulation Detector

This model detects emotional manipulation in text content by analyzing:
1. Emotional language patterns
2. Manipulative rhetoric techniques
3. Sentiment intensity and polarity shifts
4. Psychological pressure indicators
5. Urgency and scarcity language

Author: DS Capstone Project
Date: 2025
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class EmotiveManipulationDetector:
    """
    Detects emotional manipulation in text content using machine learning
    and linguistic pattern analysis.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Emotive Manipulation Detector.
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        # Enhanced emotional manipulation patterns with more intensity
        self.emotional_patterns = {
            'urgency': [
                'urgent', 'immediate', 'now', 'hurry', 'quick', 'fast', 'limited time',
                'deadline', 'expires', 'last chance', 'don\'t miss out', 'act fast',
                'right now', 'this instant', 'asap', 'rush', 'hurry up', 'don\'t wait'
            ],
            'scarcity': [
                'limited', 'exclusive', 'rare', 'unique', 'one of a kind', 'only',
                'last one', 'while supplies last', 'limited edition', 'rare opportunity',
                'final chance', 'last opportunity', 'never again', 'once in a lifetime'
            ],
            'fear': [
                'scared', 'afraid', 'terrified', 'panic', 'danger', 'threat', 'risk',
                'warning', 'caution', 'beware', 'scary', 'frightening', 'horrifying',
                'terrifying', 'shocking', 'alarming', 'disturbing'
            ],
            'guilt': [
                'should', 'must', 'have to', 'need to', 'responsible', 'duty',
                'obligation', 'owe it to', 'let down', 'disappoint', 'fail',
                'you\'re wrong if', 'you\'ll regret', 'you\'re missing out'
            ],
            'flattery': [
                'amazing', 'incredible', 'brilliant', 'genius', 'expert', 'master',
                'professional', 'special', 'elite', 'premium', 'vip', 'exclusive',
                'outstanding', 'extraordinary', 'phenomenal', 'revolutionary'
            ],
            'social_proof': [
                'everyone', 'everybody', 'people', 'others', 'they', 'them',
                'join the crowd', 'don\'t be left out', 'follow the trend',
                'millions of people', 'thousands agree', 'everyone loves', 'people say'
            ],
            'authority': [
                'expert', 'professional', 'doctor', 'scientist', 'researcher',
                'study shows', 'research proves', 'experts agree', 'authority',
                'scientifically proven', 'clinically tested', 'doctor recommended'
            ]
        }
        
        # Enhanced psychological pressure indicators
        self.pressure_indicators = [
            'you must', 'you have to', 'you need to', 'you should',
            'don\'t wait', 'don\'t delay', 'don\'t miss out',
            'this is your only chance', 'this won\'t last',
            'everyone else is doing it', 'you\'ll regret it',
            'you\'re missing out', 'don\'t be left behind',
            'this is critical', 'this is essential', 'you can\'t afford to miss'
        ]
        
        # Load pre-trained model if provided
        if model_path:
            self.load_model(model_path)
        else:
            self.is_trained = False
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for analysis.
        
        Args:
            text: Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text or text == 'None':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove RT (retweet indicator)
        text = re.sub(r'\brt\b', '', text)
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_emotional_features(self, text: str) -> Dict:
        """
        Extract emotional manipulation features from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict: Emotional manipulation features
        """
        clean_text = self.preprocess_text(text)
        words = clean_text.split()
        
        features = {}
        
        # Enhanced pattern-based features with intensity scoring
        for pattern_type, pattern_words in self.emotional_patterns.items():
            count = sum(1 for word in pattern_words if word in clean_text)
            features[f'{pattern_type}_count'] = count
            features[f'{pattern_type}_density'] = count / max(len(words), 1)
            
            # Add intensity score based on pattern frequency
            if count > 0:
                features[f'{pattern_type}_intensity'] = min(count * 0.3, 1.0)  # Cap at 1.0
            else:
                features[f'{pattern_type}_intensity'] = 0.0
        
        # Enhanced pressure indicator features
        pressure_count = sum(1 for indicator in self.pressure_indicators if indicator in clean_text)
        features['pressure_count'] = pressure_count
        features['pressure_density'] = pressure_count / max(len(words), 1)
        features['pressure_intensity'] = min(pressure_count * 0.4, 1.0)  # Higher weight for pressure
        
        # Sentiment analysis
        try:
            blob = TextBlob(clean_text)
            features['sentiment_polarity'] = blob.sentiment.polarity
            features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        except:
            features['sentiment_polarity'] = 0.0
            features['sentiment_subjectivity'] = 0.0
        
        # Text complexity features
        features['text_length'] = len(clean_text)
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        
        # Enhanced emotional intensity indicators
        exclamation_count = clean_text.count('!')
        question_count = clean_text.count('?')
        features['exclamation_density'] = exclamation_count / max(len(words), 1)
        features['question_density'] = question_count / max(len(words), 1)
        features['exclamation_intensity'] = min(exclamation_count * 0.2, 1.0)  # Cap at 1.0
        
        # Enhanced caps lock detection
        caps_count = sum(1 for char in text if char.isupper())
        features['caps_density'] = caps_count / max(len(text), 1)
        features['caps_intensity'] = min(caps_count / max(len(text), 1) * 2, 1.0)  # Amplify caps effect
        
        # Repetition detection
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        features['repetition_score'] = sum(freq - 1 for freq in word_freq.values()) / max(len(words), 1)
        
        # New: Combined manipulation intensity
        total_patterns = sum(features.get(f'{pt}_count', 0) for pt in self.emotional_patterns.keys())
        features['total_patterns'] = total_patterns
        features['overall_manipulation_intensity'] = min(total_patterns * 0.15, 1.0)
        
        # New: Emotional language ratio
        emotional_words = sum(features.get(f'{pt}_count', 0) for pt in self.emotional_patterns.keys())
        features['emotional_language_ratio'] = emotional_words / max(len(words), 1)
        
        return features
    
    def train_model(self, training_data: pd.DataFrame) -> Dict:
        """
        Train the emotional manipulation detection model.
        
        Args:
            training_data: DataFrame with 'text' and 'label confidence' columns
            
        Returns:
            Dict: Training results and metrics
        """
        print("Training Emotive Manipulation Detector...")
        
        # Extract features
        print("Extracting features...")
        feature_data = []
        for _, row in training_data.iterrows():
            features = self.extract_emotional_features(row['text'])
            feature_data.append(features)
        
        feature_df = pd.DataFrame(feature_data)
        
        # Prepare training data
        X = feature_df
        y = training_data['label confidence']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store feature names for later use
        self.feature_names = list(X.columns)
        
        self.is_trained = True
        
        results = {
            'mse': mse,
            'r2': r2,
            'rmse': np.sqrt(mse),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        print(f"Training completed!")
        print(f"R² Score: {r2:.3f}")
        print(f"RMSE: {np.sqrt(mse):.3f}")
        
        return results
    
    def predict_manipulation_score(self, text: str) -> float:
        """
        Predict emotional manipulation score for given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            float: Manipulation score between 0 and 1
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        features = self.extract_emotional_features(text)
        feature_vector = pd.DataFrame([features])
        
        # Ensure all expected features are present
        for feature in self.feature_names:
            if feature not in feature_vector.columns:
                feature_vector[feature] = 0
        
        # Reorder columns to match training data
        feature_vector = feature_vector[self.feature_names]
        
        # Make prediction
        score = self.model.predict(feature_vector)[0]
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def analyze_text(self, text: str) -> Dict:
        """
        Comprehensive analysis of text for emotional manipulation.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict: Comprehensive analysis results
        """
        # Extract features
        features = self.extract_emotional_features(text)
        
        # Get prediction if model is trained
        if self.is_trained:
            ml_score = self.predict_manipulation_score(text)
            rule_score = self._rule_based_scoring(features)
            
            # Use hybrid approach: if ML score is very low but rule-based score is high,
            # use the higher score (ML model might not have learned these patterns well)
            if ml_score < 0.4 and rule_score > 0.5:
                manipulation_score = rule_score
                score_source = "rule_based_override"
            else:
                manipulation_score = ml_score
                score_source = "ml_model"
        else:
            # Use rule-based scoring as fallback
            manipulation_score = self._rule_based_scoring(features)
            score_source = "rule_based"
        
        # Analyze specific patterns
        pattern_analysis = self._analyze_manipulation_patterns(text)
        
        # Risk assessment
        risk_level = self._assess_risk_level(manipulation_score)
        
        return {
            'manipulation_score': manipulation_score,
            'risk_level': risk_level,
            'score_source': score_source,
            'features': features,
            'pattern_analysis': pattern_analysis,
            'text_length': len(text),
            'analysis_timestamp': pd.Timestamp.now()
        }
    
    def _rule_based_scoring(self, features: Dict) -> float:
        """
        Enhanced rule-based scoring when ML model is not available.
        
        Args:
            features: Extracted features
            
        Returns:
            float: Rule-based manipulation score
        """
        score = 0.0
        
        # Enhanced pattern-based scoring with higher weights
        for pattern_type in ['urgency', 'scarcity', 'fear', 'guilt', 'flattery', 'social_proof', 'authority']:
            count = features.get(f'{pattern_type}_count', 0)
            density = features.get(f'{pattern_type}_density', 0)
            intensity = features.get(f'{pattern_type}_intensity', 0)
            
            # Higher scoring for multiple patterns
            if count >= 3:  # High frequency
                score += 0.25
            elif count >= 2:  # Medium frequency
                score += 0.18
            elif count >= 1:  # Low frequency
                score += 0.12
            
            # Additional scoring for high density
            if density > 0.15:
                score += 0.20
            elif density > 0.08:
                score += 0.15
        
        # Enhanced pressure indicators with higher weight
        pressure_count = features.get('pressure_count', 0)
        pressure_intensity = features.get('pressure_intensity', 0)
        
        if pressure_count >= 2:
            score += 0.35  # High pressure
        elif pressure_count >= 1:
            score += 0.25  # Medium pressure
        
        # Add intensity bonus
        score += pressure_intensity * 0.2
        
        # Enhanced sentiment features
        sentiment_polarity = features.get('sentiment_polarity', 0)
        if sentiment_polarity < -0.4:  # Strong negative sentiment
            score += 0.15
        elif sentiment_polarity < -0.2:  # Moderate negative sentiment
            score += 0.10
        
        # Enhanced emotional intensity
        exclamation_intensity = features.get('exclamation_intensity', 0)
        caps_intensity = features.get('caps_intensity', 0)
        
        score += exclamation_intensity * 0.15
        score += caps_intensity * 0.20
        
        # Overall manipulation intensity bonus
        overall_intensity = features.get('overall_manipulation_intensity', 0)
        score += overall_intensity * 0.25
        
        # Emotional language ratio bonus
        emotional_ratio = features.get('emotional_language_ratio', 0)
        if emotional_ratio > 0.2:  # High emotional language
            score += 0.20
        elif emotional_ratio > 0.1:  # Medium emotional language
            score += 0.15
        
        return min(score, 1.0)
    
    def _analyze_manipulation_patterns(self, text: str) -> Dict:
        """
        Analyze specific manipulation patterns in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict: Pattern analysis results
        """
        clean_text = self.preprocess_text(text)
        
        patterns_found = {}
        
        for pattern_type, pattern_words in self.emotional_patterns.items():
            found_words = [word for word in pattern_words if word in clean_text]
            if found_words:
                patterns_found[pattern_type] = {
                    'words_found': found_words,
                    'count': len(found_words)
                }
        
        # Pressure indicators
        pressure_found = [indicator for indicator in self.pressure_indicators if indicator in clean_text]
        if pressure_found:
            patterns_found['pressure_indicators'] = {
                'indicators_found': pressure_found,
                'count': len(pressure_found)
            }
        
        return patterns_found
    
    def _assess_risk_level(self, manipulation_score: float) -> str:
        """
        Assess the risk level of emotional manipulation.
        
        Args:
            manipulation_score: Manipulation score
            
        Returns:
            str: Risk level assessment
        """
        if manipulation_score > 0.7:
            return 'HIGH'
        elif manipulation_score > 0.5:
            return 'MEDIUM'
        elif manipulation_score > 0.3:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        import joblib
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a pre-trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        import joblib
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {model_path}")
    
    def generate_report(self, analysis_results: Dict) -> str:
        """
        Generate a human-readable report from analysis results.
        
        Args:
            analysis_results: Results from text analysis
            
        Returns:
            str: Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("EMOTIVE MANIPULATION ANALYSIS REPORT")
        report.append("=" * 60)
        
        report.append(f"Manipulation Score: {analysis_results['manipulation_score']:.3f}")
        report.append(f"Risk Level: {analysis_results['risk_level']}")
        report.append(f"Text Length: {analysis_results['text_length']} characters")
        report.append("")
        
        # Pattern analysis
        patterns = analysis_results['pattern_analysis']
        if patterns:
            report.append("MANIPULATION PATTERNS DETECTED:")
            for pattern_type, pattern_data in patterns.items():
                report.append(f"  {pattern_type.replace('_', ' ').title()}:")
                if 'words_found' in pattern_data:
                    report.append(f"    Words: {', '.join(pattern_data['words_found'])}")
                if 'indicators_found' in pattern_data:
                    report.append(f"    Indicators: {', '.join(pattern_data['indicators_found'])}")
                report.append(f"    Count: {pattern_data['count']}")
                report.append("")
        else:
            report.append("No manipulation patterns detected.")
            report.append("")
        
        # Feature highlights
        features = analysis_results['features']
        report.append("KEY FEATURES:")
        report.append(f"  Sentiment Polarity: {features.get('sentiment_polarity', 0):.3f}")
        report.append(f"  Sentiment Subjectivity: {features.get('sentiment_subjectivity', 0):.3f}")
        report.append(f"  Exclamation Density: {features.get('exclamation_density', 0):.3f}")
        report.append(f"  Caps Lock Density: {features.get('caps_density', 0):.3f}")
        report.append(f"  Pressure Indicators: {features.get('pressure_count', 0)}")
        report.append(f"  Total Patterns: {features.get('total_patterns', 0)}")
        report.append(f"  Overall Intensity: {features.get('overall_manipulation_intensity', 0):.3f}")
        
        report.append("")
        report.append(f"Analysis Timestamp: {analysis_results['analysis_timestamp']}")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """
    Example usage of the Emotive Manipulation Detector.
    """
    print("Emotive Manipulation Detector")
    print("=" * 40)
    
    # Initialize detector
    detector = EmotiveManipulationDetector()
    
    # Example text analysis
    example_texts = [
        "URGENT! Don't miss this LIMITED TIME offer! Act NOW before it's gone forever!",
        "This is a normal informative tweet about current events.",
        "You MUST buy this product NOW! Everyone else is doing it! Don't be left behind!"
    ]
    
    print("Example Analysis:")
    print()
    
    for i, text in enumerate(example_texts, 1):
        print(f"Example {i}:")
        print(f"Text: {text}")
        
        try:
            results = detector.analyze_text(text)
            print(f"Manipulation Score: {results['manipulation_score']:.3f}")
            print(f"Risk Level: {results['risk_level']}")
        except Exception as e:
            print(f"Analysis error: {str(e)}")
        
        print("-" * 40)
    
    print("\nTo train the model with your annotated data:")
    print("detector.train_model(your_dataframe)")
    print("\nTo analyze new text:")
    print("score = detector.predict_manipulation_score('your text here')")


if __name__ == "__main__":
    main()
