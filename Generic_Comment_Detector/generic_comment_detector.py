"""
Generic Comment Detector

This model detects low-quality engagement and content by analyzing:
1. Lexical diversity and vocabulary richness
2. Semantic coherence and embedding distance
3. Content structure and engagement patterns
4. Text complexity and readability
5. Generic vs. specific content indicators

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
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class GenericCommentDetector:
    """
    Detects low-quality engagement and generic content using machine learning
    and linguistic analysis techniques.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Generic Comment Detector.
        
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
        
        # Generic content indicators
        self.generic_phrases = [
            'nice', 'good', 'great', 'awesome', 'amazing', 'cool', 'wow', 'omg',
            'lol', 'haha', 'thanks', 'thank you', 'congrats', 'congratulations',
            'well done', 'good job', 'keep it up', 'you got this', 'stay strong',
            'sending love', 'thoughts and prayers', 'stay safe', 'take care',
            'good luck', 'best wishes', 'have a great day', 'happy friday',
            'mood', 'same', 'relatable', 'this', 'that', 'yes', 'no', 'okay',
            'sure', 'maybe', 'idk', 'i don\'t know', 'whatever', 'fine', 'ok'
        ]
        
        # Low-quality engagement patterns
        self.low_quality_patterns = {
            'generic_responses': [
                'nice', 'good', 'great', 'awesome', 'cool', 'wow', 'omg',
                'lol', 'haha', 'thanks', 'congrats', 'well done'
            ],
            'minimal_engagement': [
                'this', 'that', 'yes', 'no', 'okay', 'sure', 'maybe',
                'idk', 'i don\'t know', 'whatever', 'fine', 'ok'
            ],
            'spam_indicators': [
                'buy now', 'click here', 'limited time', 'act fast',
                'don\'t miss out', 'exclusive offer', 'free trial',
                'earn money', 'make money', 'work from home'
            ],
            'bot_like': [
                'follow me', 'follow back', 'retweet', 'like for like',
                'check out my profile', 'new follower', 'trending'
            ]
        }
        
        # High-quality content indicators
        self.high_quality_indicators = [
            'because', 'therefore', 'however', 'although', 'nevertheless',
            'furthermore', 'moreover', 'additionally', 'specifically',
            'in particular', 'for example', 'such as', 'according to',
            'research shows', 'study indicates', 'evidence suggests',
            'analysis reveals', 'data shows', 'statistics indicate'
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
    
    def extract_content_quality_features(self, text: str) -> Dict:
        """
        Extract content quality features from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict: Content quality features
        """
        clean_text = self.preprocess_text(text)
        words = clean_text.split()
        
        features = {}
        
        # Basic text features
        features['text_length'] = len(clean_text)
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        
        # Lexical diversity features
        unique_words = set(words)
        features['unique_word_count'] = len(unique_words)
        features['lexical_diversity'] = len(unique_words) / max(len(words), 1)  # Type-token ratio
        features['vocabulary_richness'] = len(unique_words) / max(len(words), 1) * 100
        
        # Generic content detection
        generic_count = sum(1 for phrase in self.generic_phrases if phrase in clean_text)
        features['generic_phrase_count'] = generic_count
        features['generic_phrase_density'] = generic_count / max(len(words), 1)
        
        # Low-quality pattern detection
        for pattern_type, pattern_words in self.low_quality_patterns.items():
            count = sum(1 for word in pattern_words if word in clean_text)
            features[f'{pattern_type}_count'] = count
            features[f'{pattern_type}_density'] = count / max(len(words), 1)
        
        # High-quality content indicators
        high_quality_count = sum(1 for indicator in self.high_quality_indicators if indicator in clean_text)
        features['high_quality_indicator_count'] = high_quality_count
        features['high_quality_indicator_density'] = high_quality_count / max(len(words), 1)
        
        # Sentiment analysis
        try:
            blob = TextBlob(clean_text)
            features['sentiment_polarity'] = blob.sentiment.polarity
            features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        except:
            features['sentiment_polarity'] = 0.0
            features['sentiment_subjectivity'] = 0.0
        
        # Content structure features
        features['has_question'] = 1 if '?' in text else 0
        features['has_exclamation'] = 1 if '!' in text else 0
        features['has_mention'] = 1 if '@' in text else 0
        features['has_hashtag'] = 1 if '#' in text else 0
        features['has_url'] = 1 if 'http' in text else 0
        
        # Readability features
        features['avg_sentence_length'] = len(words) / max(text.count('.') + text.count('!') + text.count('?'), 1)
        features['complexity_score'] = (features['avg_word_length'] * features['avg_sentence_length']) / 100
        
        # Engagement quality features
        features['engagement_quality'] = 0.0
        
        # Reward generic content (inverted scoring)
        if features['generic_phrase_density'] > 0.3:
            features['engagement_quality'] += 0.3
        elif features['generic_phrase_density'] > 0.1:
            features['engagement_quality'] += 0.1
        
        # Penalize high-quality indicators (inverted scoring)
        if features['high_quality_indicator_density'] > 0.05:
            features['engagement_quality'] -= 0.2
        
        # Penalize lexical diversity (inverted scoring)
        if features['lexical_diversity'] > 0.8:
            features['engagement_quality'] -= 0.2
        elif features['lexical_diversity'] > 0.6:
            features['engagement_quality'] -= 0.1
        
        # Reward very short responses (inverted scoring)
        if features['word_count'] < 5:
            features['engagement_quality'] += 0.2
        elif features['word_count'] < 10:
            features['engagement_quality'] += 0.1
        
        # Reward excessive punctuation (inverted scoring)
        punctuation_ratio = (text.count('!') + text.count('?')) / max(len(text), 1)
        if punctuation_ratio > 0.1:
            features['engagement_quality'] += 0.1
        
        # Overall generic score (inverted, normalized)
        features['overall_quality_score'] = max(0.0, min(1.0, 0.5 + features['engagement_quality']))
        
        return features
    
    def train_model(self, training_data: pd.DataFrame) -> Dict:
        """
        Train the generic comment detection model.
        
        Args:
            training_data: DataFrame with 'text' and 'label confidence' columns
            
        Returns:
            Dict: Training results and metrics
        """
        print("Training Generic Comment Detector...")
        
        # Extract features
        print("Extracting features...")
        feature_data = []
        for _, row in training_data.iterrows():
            features = self.extract_content_quality_features(row['text'])
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
    
    def predict_content_quality_score(self, text: str) -> float:
        """
        Predict generic content score for given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            float: Generic content score between 0 and 1 (higher = more generic)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        features = self.extract_content_quality_features(text)
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
        Comprehensive analysis of text for generic content detection.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict: Comprehensive analysis results (higher scores = more generic)
        """
        # Extract features
        features = self.extract_content_quality_features(text)
        
        # Get prediction if model is trained
        if self.is_trained:
            ml_score = self.predict_content_quality_score(text)
            rule_score = features['overall_quality_score']
            
            # Use hybrid approach: if ML score is very low but rule-based score is reasonable,
            # use the higher score
            if ml_score < 0.2 and rule_score > 0.3:
                quality_score = rule_score
                score_source = "rule_based_override"
            else:
                quality_score = ml_score
                score_source = "ml_model"
        else:
            # Use rule-based scoring as fallback
            quality_score = features['overall_quality_score']
            score_source = "rule_based"
        
        # Analyze specific patterns
        pattern_analysis = self._analyze_content_patterns(text)
        
        # Quality assessment
        quality_level = self._assess_quality_level(quality_score)
        
        return {
            'generic_content_score': quality_score,
            'genericness_level': quality_level,
            'score_source': score_source,
            'features': features,
            'pattern_analysis': pattern_analysis,
            'text_length': len(text),
            'analysis_timestamp': pd.Timestamp.now()
        }
    
    def _analyze_content_patterns(self, text: str) -> Dict:
        """
        Analyze specific content patterns in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict: Pattern analysis results
        """
        clean_text = self.preprocess_text(text)
        
        patterns_found = {}
        
        # Generic phrase analysis
        generic_found = [phrase for phrase in self.generic_phrases if phrase in clean_text]
        if generic_found:
            patterns_found['generic_phrases'] = {
                'phrases_found': generic_found,
                'count': len(generic_found)
            }
        
        # Low-quality pattern analysis
        for pattern_type, pattern_words in self.low_quality_patterns.items():
            found_words = [word for word in pattern_words if word in clean_text]
            if found_words:
                patterns_found[pattern_type] = {
                    'words_found': found_words,
                    'count': len(found_words)
                }
        
        # High-quality indicator analysis
        high_quality_found = [indicator for indicator in self.high_quality_indicators if indicator in clean_text]
        if high_quality_found:
            patterns_found['high_quality_indicators'] = {
                'indicators_found': high_quality_found,
                'count': len(high_quality_found)
            }
        
        return patterns_found
    
    def _assess_quality_level(self, quality_score: float) -> str:
        """
        Assess the genericness level of the content (inverted scoring).
        
        Args:
            quality_score: Genericness score (higher = more generic)
            
        Returns:
            str: Genericness level assessment
        """
        if quality_score > 0.7:
            return 'VERY_GENERIC'
        elif quality_score > 0.5:
            return 'GENERIC'
        elif quality_score > 0.3:
            return 'MODERATE'
        else:
            return 'HIGH_QUALITY'
    
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
        report.append("CONTENT QUALITY ANALYSIS REPORT")
        report.append("=" * 60)
        
        report.append(f"Content Quality Score: {analysis_results['content_quality_score']:.3f}")
        report.append(f"Quality Level: {analysis_results['quality_level']}")
        report.append(f"Text Length: {analysis_results['text_length']} characters")
        report.append("")
        
        # Pattern analysis
        patterns = analysis_results['pattern_analysis']
        if patterns:
            report.append("CONTENT PATTERNS DETECTED:")
            for pattern_type, pattern_data in patterns.items():
                report.append(f"  {pattern_type.replace('_', ' ').title()}:")
                if 'phrases_found' in pattern_data:
                    report.append(f"    Phrases: {', '.join(pattern_data['phrases_found'])}")
                if 'words_found' in pattern_data:
                    report.append(f"    Words: {', '.join(pattern_data['words_found'])}")
                if 'indicators_found' in pattern_data:
                    report.append(f"    Indicators: {', '.join(pattern_data['indicators_found'])}")
                report.append(f"    Count: {pattern_data['count']}")
                report.append("")
        else:
            report.append("No significant patterns detected.")
            report.append("")
        
        # Feature highlights
        features = analysis_results['features']
        report.append("KEY FEATURES:")
        report.append(f"  Lexical Diversity: {features.get('lexical_diversity', 0):.3f}")
        report.append(f"  Vocabulary Richness: {features.get('vocabulary_richness', 0):.1f}")
        report.append(f"  Generic Phrase Density: {features.get('generic_phrase_density', 0):.3f}")
        report.append(f"  High-Quality Indicators: {features.get('high_quality_indicator_count', 0)}")
        report.append(f"  Engagement Quality: {features.get('engagement_quality', 0):.3f}")
        report.append(f"  Overall Quality Score: {features.get('overall_quality_score', 0):.3f}")
        
        report.append("")
        report.append(f"Analysis Timestamp: {analysis_results['analysis_timestamp']}")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """
    Example usage of the Generic Comment Detector.
    """
    print("Generic Comment Detector")
    print("=" * 40)
    
    # Initialize detector
    detector = GenericCommentDetector()
    
    # Example text analysis
    example_texts = [
        "This is a generic response that adds no value to the conversation.",
        "Great analysis! The data clearly shows that the correlation coefficient of 0.87 indicates a strong positive relationship between these variables.",
        "nice",
        "Thanks for sharing this insightful research. The methodology appears sound, and the conclusions align with recent findings in the field."
    ]
    
    print("Example Analysis:")
    print()
    
    for i, text in enumerate(example_texts, 1):
        print(f"Example {i}:")
        print(f"Text: {text}")
        
        try:
            results = detector.analyze_text(text)
            print(f"Quality Score: {results['content_quality_score']:.3f}")
            print(f"Quality Level: {results['quality_level']}")
        except Exception as e:
            print(f"Analysis error: {str(e)}")
        
        print("-" * 40)
    
    print("\nTo train the model with your annotated data:")
    print("detector.train_model(your_dataframe)")
    print("\nTo analyze new text:")
    print("score = detector.predict_content_quality_score('your text here')")


if __name__ == "__main__":
    main()
