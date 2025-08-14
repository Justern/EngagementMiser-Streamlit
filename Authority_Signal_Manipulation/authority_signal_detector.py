"""
Authority-Signal Manipulation (ASM) - Profile Analysis Complexity

This model detects when users manipulate their authority signals by:
1. Using overly complex language to appear intelligent
2. Creating engagement patterns that don't match content quality
3. Having discrepancies between profile signals and actual engagement
4. Generating low-quality engagement (bots, fake accounts, etc.)

Author: DS Capstone Project
Date: 2025
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class AuthoritySignalDetector:
    """
    Detects authority signal manipulation in Twitter content by analyzing
    lexical diversity, semantic coherence, engagement quality, and profile mismatches.
    """
    
    def __init__(self, 
                 lexical_threshold: float = 0.6,
                 coherence_threshold: float = 0.5,
                 engagement_threshold: float = 0.4):
        """
        Initialize the Authority Signal Detector.
        
        Args:
            lexical_threshold: Threshold for lexical complexity detection
            coherence_threshold: Threshold for semantic coherence
            engagement_threshold: Threshold for engagement quality
        """
        self.lexical_threshold = lexical_threshold
        self.coherence_threshold = coherence_threshold
        self.engagement_threshold = engagement_threshold
        
        # TF-IDF vectorizer for semantic analysis
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        # Common authority signal words (overly formal, academic, etc.)
        self.authority_words = {
            'academic': ['furthermore', 'moreover', 'nevertheless', 'consequently', 'subsequently'],
            'formal': ['henceforth', 'whereas', 'notwithstanding', 'aforementioned', 'heretofore'],
            'technical': ['algorithmically', 'methodologically', 'theoretically', 'empirically'],
            'jargon': ['paradigm', 'heuristic', 'ontology', 'epistemology', 'methodology']
        }
        
        # Authority corpus for legitimate authority signal comparison
        self.authority_corpus = None
        self.authority_corpus_loaded = False
        
        # Bot-like engagement patterns
        self.bot_patterns = {
            'timing': [0, 1, 2, 3, 4, 5],  # Seconds between engagements
            'repetitive': ['great post!', 'thanks for sharing', 'interesting', 'agree'],
            'generic': ['nice', 'good', 'cool', 'awesome', 'wow']
        }
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for analysis.
        
        Args:
            text (str): Raw text to clean
            
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
    
    def load_authority_corpus(self, authority_data: pd.DataFrame) -> None:
        """
        Load authority figures corpus for legitimate authority signal comparison.
        
        Args:
            authority_data: DataFrame with authority figures data
        """
        if authority_data is not None and not authority_data.empty:
            self.authority_corpus = authority_data
            self.authority_corpus_loaded = True
            print(f"✅ Authority corpus loaded with {len(authority_data)} figures")
        else:
            print("⚠️  No authority corpus data provided")
    
    def analyze_lexical_complexity(self, text: str) -> Dict:
        """
        Analyze lexical complexity and diversity of text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict: Lexical complexity analysis results
        """
        if not text or len(text.strip()) < 10:
            return {
                'complexity_score': 0.0,
                'vocabulary_diversity': 0.0,
                'sentence_complexity': 0.0,
                'authority_word_density': 0.0,
                'is_overly_complex': False
            }
        
        # Clean text
        clean_text = self.preprocess_text(text)
        words = clean_text.split()
        
        if len(words) < 5:
            return {
                'complexity_score': 0.0,
                'vocabulary_diversity': 0.0,
                'sentence_complexity': 0.0,
                'authority_word_density': 0.0,
                'is_overly_complex': False
            }
        
        # Vocabulary diversity (Type-Token Ratio)
        unique_words = set(words)
        vocabulary_diversity = len(unique_words) / len(words)
        
        # Sentence complexity (average words per sentence)
        sentences = re.split(r'[.!?]+', clean_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_words_per_sentence = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        
        # Authority word density
        authority_word_count = 0
        for category, words_list in self.authority_words.items():
            for word in words_list:
                authority_word_count += clean_text.count(word)
        
        authority_word_density = authority_word_count / len(words)
        
        # Word length analysis
        avg_word_length = np.mean([len(word) for word in words])
        
        # Calculate overall complexity score
        complexity_score = (
            vocabulary_diversity * 0.3 +
            min(avg_words_per_sentence / 20.0, 1.0) * 0.3 +
            min(authority_word_density * 100, 1.0) * 0.2 +
            min(avg_word_length / 8.0, 1.0) * 0.2
        )
        
        is_overly_complex = complexity_score > self.lexical_threshold
        
        return {
            'complexity_score': min(complexity_score, 1.0),
            'vocabulary_diversity': vocabulary_diversity,
            'sentence_complexity': avg_words_per_sentence,
            'authority_word_density': authority_word_density,
            'avg_word_length': avg_word_length,
            'is_overly_complex': is_overly_complex
        }
    
    def analyze_semantic_coherence(self, text: str) -> Dict:
        """
        Analyze semantic coherence and topic consistency.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict: Semantic coherence analysis results
        """
        if not text or len(text.strip()) < 20:
            return {
                'coherence_score': 0.0,
                'topic_consistency': 0.0,
                'semantic_density': 0.0,
                'is_coherent': False
            }
        
        # Clean text
        clean_text = self.preprocess_text(text)
        
        # Split into sentences for analysis
        sentences = re.split(r'[.!?]+', clean_text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]
        
        if len(sentences) < 2:
            return {
                'coherence_score': 0.5,  # Neutral for single sentence
                'topic_consistency': 0.5,
                'semantic_density': 0.5,
                'is_coherent': True
            }
        
        try:
            # Vectorize sentences
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Calculate semantic coherence between consecutive sentences
            consecutive_similarities = []
            for i in range(len(similarity_matrix) - 1):
                similarity = similarity_matrix[i][i + 1]
                consecutive_similarities.append(similarity)
            
            # Topic consistency (similarity between consecutive sentences)
            topic_consistency = np.mean(consecutive_similarities) if consecutive_similarities else 0.0
            
            # Overall semantic density (average similarity across all sentence pairs)
            total_similarity = 0
            pair_count = 0
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    total_similarity += similarity_matrix[i][j]
                    pair_count += 1
            
            semantic_density = total_similarity / pair_count if pair_count > 0 else 0.0
            
            # Calculate overall coherence score
            coherence_score = (
                topic_consistency * 0.6 +
                semantic_density * 0.4
            )
            
            is_coherent = coherence_score > self.coherence_threshold
            
            return {
                'coherence_score': min(coherence_score, 1.0),
                'topic_consistency': topic_consistency,
                'semantic_density': semantic_density,
                'consecutive_similarities': consecutive_similarities,
                'is_coherent': is_coherent
            }
            
        except Exception as e:
            return {
                'coherence_score': 0.5,
                'topic_consistency': 0.5,
                'semantic_density': 0.5,
                'error': str(e),
                'is_coherent': True
            }
    
    def analyze_engagement_quality(self, engagement_data: Dict) -> Dict:
        """
        Analyze engagement quality and detect bot-like patterns.
        
        Args:
            engagement_data (Dict): Engagement metrics and patterns
            
        Returns:
            Dict: Engagement quality analysis results
        """
        # Extract engagement metrics
        followers_count = engagement_data.get('followers_count', 0)
        following_count = engagement_data.get('following_count', 0)
        tweet_count = engagement_data.get('tweet_count', 0)
        like_count = engagement_data.get('like_count', 0)
        retweet_count = engagement_data.get('retweet_count', 0)
        reply_count = engagement_data.get('reply_count', 0)
        
        # Calculate engagement ratios
        engagement_rate = (like_count + retweet_count + reply_count) / max(followers_count, 1)
        
        # Bot detection indicators
        bot_indicators = []
        bot_score = 0.0
        
        # 1. Follower/Following ratio (suspicious if too high or too low)
        if followers_count > 0 and following_count > 0:
            follower_following_ratio = followers_count / following_count
            if follower_following_ratio > 100 or follower_following_ratio < 0.01:
                bot_indicators.append('suspicious_follower_ratio')
                bot_score += 0.3
        
        # 2. Engagement rate (too high might indicate bot activity)
        if engagement_rate > 0.5:  # 50% engagement rate is suspicious
            bot_indicators.append('suspicious_engagement_rate')
            bot_score += 0.2
        
        # 3. Tweet frequency (too many tweets per day is suspicious)
        if tweet_count > 1000:  # More than 1000 tweets is suspicious
            bot_indicators.append('excessive_tweet_frequency')
            bot_score += 0.2
        
        # 4. Account age vs. tweet count (new accounts with many tweets)
        account_age_days = engagement_data.get('account_age_days', 365)
        if account_age_days < 30 and tweet_count > 100:
            bot_indicators.append('new_account_many_tweets')
            bot_score += 0.3
        
        # 5. Engagement pattern consistency (all engagement at same time)
        if engagement_data.get('engagement_timing_consistency', 0) > 0.8:
            bot_indicators.append('consistent_engagement_timing')
            bot_score += 0.2
        
        # Normalize bot score
        bot_score = min(bot_score, 1.0)
        
        # Calculate overall engagement quality
        engagement_quality = 1.0 - bot_score
        
        return {
            'engagement_quality': engagement_quality,
            'bot_score': bot_score,
            'bot_indicators': bot_indicators,
            'engagement_rate': engagement_rate,
            'follower_following_ratio': followers_count / max(following_count, 1) if following_count > 0 else 0,
            'is_low_quality': engagement_quality < self.engagement_threshold
        }
    
    def analyze_profile_content_mismatch(self, profile_data: Dict, content_quality: float) -> Dict:
        """
        Analyze mismatch between profile authority signals and content quality.
        
        Args:
            profile_data (Dict): Profile information and metrics
            content_quality (float): Overall content quality score
            
        Returns:
            Dict: Profile-content mismatch analysis
        """
        # Extract profile metrics
        followers_count = profile_data.get('followers_count', 0)
        verified = profile_data.get('verified', False)
        account_age_days = profile_data.get('account_age_days', 365)
        description_length = profile_data.get('description_length', 0)
        profile_image = profile_data.get('profile_image', False)
        banner_image = profile_data.get('banner_image', False)
        
        # Calculate authority signal score
        authority_signals = 0.0
        
        # Follower-based authority (logarithmic scale)
        if followers_count > 0:
            authority_signals += min(np.log10(followers_count) / 6.0, 1.0) * 0.3
        
        # Verification status
        if verified:
            authority_signals += 0.2
        
        # Account age (older accounts seem more authoritative)
        authority_signals += min(account_age_days / 3650.0, 1.0) * 0.2  # 10 years max
        
        # Profile completeness
        profile_completeness = 0.0
        if description_length > 50:
            profile_completeness += 0.5
        if profile_image:
            profile_completeness += 0.25
        if banner_image:
            profile_completeness += 0.25
        
        authority_signals += profile_completeness * 0.3
        
        # Calculate mismatch score
        # Higher mismatch = more suspicious (high authority signals but low content quality)
        mismatch_score = max(0, authority_signals - content_quality)
        
        # Normalize mismatch score
        normalized_mismatch = min(mismatch_score, 1.0)
        
        # Check against authority corpus if available
        corpus_comparison = self._compare_with_authority_corpus(profile_data, content_quality)
        
        return {
            'authority_signals': authority_signals,
            'content_quality': content_quality,
            'mismatch_score': normalized_mismatch,
            'profile_completeness': profile_completeness,
            'has_mismatch': normalized_mismatch > 0.3,
            'corpus_comparison': corpus_comparison
        }
    
    def analyze_tweet(self, tweet_data: Dict) -> Dict:
        """
        Analyze a single tweet for authority signal manipulation.
        
        Args:
            tweet_data (Dict): Tweet data including text and engagement metrics
            
        Returns:
            Dict: Comprehensive ASM analysis results
        """
        text = tweet_data.get('text', '')
        
        # 1. Lexical complexity analysis
        lexical_analysis = self.analyze_lexical_complexity(text)
        
        # 2. Semantic coherence analysis
        coherence_analysis = self.analyze_semantic_coherence(text)
        
        # 3. Engagement quality analysis
        engagement_analysis = self.analyze_engagement_quality(tweet_data)
        
        # 4. Profile-content mismatch analysis
        profile_data = {
            'followers_count': tweet_data.get('followers_count', 0),
            'following_count': tweet_data.get('following_count', 0),
            'verified': tweet_data.get('verified', False),
            'account_age_days': tweet_data.get('account_age_days', 365),
            'description_length': tweet_data.get('description_length', 0),
            'profile_image': tweet_data.get('profile_image', False),
            'banner_image': tweet_data.get('banner_image', False),
            'name': tweet_data.get('name', ''),
            'title': tweet_data.get('title', ''),
            'organization': tweet_data.get('organization', '')
        }
        
        # Calculate content quality (average of lexical and coherence)
        content_quality = (
            lexical_analysis['complexity_score'] * 0.5 +
            coherence_analysis['coherence_score'] * 0.5
        )
        
        mismatch_analysis = self.analyze_profile_content_mismatch(profile_data, content_quality)
        
        # 5. Calculate overall ASM score
        asm_score = self._calculate_asm_score(
            lexical_analysis,
            coherence_analysis,
            engagement_analysis,
            mismatch_analysis
        )
        
        return {
            'tweet_id': tweet_data.get('tweet_id', ''),
            'author_id': tweet_data.get('author_id', ''),
            'asm_score': asm_score,
            'is_manipulated': asm_score > 0.6,
            'risk_level': self._assess_risk_level(asm_score),
            'lexical_analysis': lexical_analysis,
            'coherence_analysis': coherence_analysis,
            'engagement_analysis': engagement_analysis,
            'mismatch_analysis': mismatch_analysis,
            'content_quality': content_quality,
            'analysis_timestamp': pd.Timestamp.now()
        }
    
    def _calculate_asm_score(self, lexical_analysis: Dict, coherence_analysis: Dict,
                            engagement_analysis: Dict, mismatch_analysis: Dict) -> float:
        """
        Calculate overall authority signal manipulation score.
        
        Args:
            lexical_analysis: Results from lexical complexity analysis
            coherence_analysis: Results from semantic coherence analysis
            engagement_analysis: Results from engagement quality analysis
            mismatch_analysis: Results from profile-content mismatch analysis
            
        Returns:
            float: ASM score between 0 and 1
        """
        score = 0.0
        
        # Lexical complexity component (30%)
        if lexical_analysis['is_overly_complex']:
            score += 0.3 * lexical_analysis['complexity_score']
        
        # Semantic coherence component (25%)
        if not coherence_analysis['is_coherent']:
            score += 0.25 * (1.0 - coherence_analysis['coherence_score'])
        
        # Engagement quality component (25%)
        if engagement_analysis['is_low_quality']:
            score += 0.25 * (1.0 - engagement_analysis['engagement_quality'])
        
        # Profile mismatch component (20%)
        if mismatch_analysis['has_mismatch']:
            score += 0.2 * mismatch_analysis['mismatch_score']
        
        return min(score, 1.0)
    
    def _compare_with_authority_corpus(self, profile_data: Dict, content_quality: float) -> Dict:
        """
        Compare user profile against legitimate authority figures corpus.
        
        Args:
            profile_data: User profile information
            content_quality: Content quality score
            
        Returns:
            Dict: Corpus comparison results
        """
        if not self.authority_corpus_loaded or self.authority_corpus is None:
            return {
                'corpus_available': False,
                'legitimacy_score': 0.5,
                'closest_match': None,
                'suspicious_indicators': []
            }
        
        # Extract profile information for comparison
        user_name = str(profile_data.get('name', '')).lower()
        user_title = str(profile_data.get('title', '')).lower()
        user_org = str(profile_data.get('organization', '')).lower()
        
        # If title/organization not available, try to extract from description or other fields
        if not user_title and profile_data.get('description'):
            desc = str(profile_data.get('description', '')).lower()
            # Look for common title indicators in description
            title_indicators = ['ceo', 'founder', 'director', 'manager', 'consultant', 'expert']
            for indicator in title_indicators:
                if indicator in desc:
                    user_title = indicator
                    break
        
        suspicious_indicators = []
        legitimacy_score = 0.5  # Base score
        
        # Check for suspicious patterns
        if user_title:
            # Check for overly grandiose titles
            grandiose_titles = ['expert', 'guru', 'master', 'specialist', 'consultant', 'advisor']
            if any(title in user_title for title in grandiose_titles):
                suspicious_indicators.append('grandiose_title')
                legitimacy_score -= 0.2
        
        if user_org:
            # Check for suspicious organization names
            suspicious_orgs = ['consulting', 'solutions', 'enterprises', 'ventures', 'holdings']
            if any(org in user_org for org in suspicious_orgs):
                suspicious_indicators.append('suspicious_organization')
                legitimacy_score -= 0.1
        
        # Find closest match in authority corpus
        closest_match = None
        best_similarity = 0.0
        
        for _, row in self.authority_corpus.iterrows():
            corpus_name = str(row.get('name', '')).lower()
            corpus_title = str(row.get('title', '')).lower()
            corpus_org = str(row.get('organization', '')).lower()
            
            # Calculate similarity scores
            name_similarity = self._calculate_text_similarity(user_name, corpus_name)
            title_similarity = self._calculate_text_similarity(user_title, corpus_title)
            org_similarity = self._calculate_text_similarity(user_org, corpus_org)
            
            # Overall similarity
            overall_similarity = (name_similarity + title_similarity + org_similarity) / 3
            
            if overall_similarity > best_similarity:
                best_similarity = overall_similarity
                closest_match = {
                    'name': row.get('name', ''),
                    'title': row.get('title', ''),
                    'organization': row.get('organization', ''),
                    'similarity': overall_similarity
                }
        
        # Adjust legitimacy score based on corpus similarity
        if closest_match and closest_match['similarity'] > 0.7:
            legitimacy_score += 0.2  # High similarity to legitimate authority
        elif closest_match and closest_match['similarity'] > 0.5:
            legitimacy_score += 0.1  # Moderate similarity
        
        # Normalize legitimacy score
        legitimacy_score = max(0.0, min(1.0, legitimacy_score))
        
        return {
            'corpus_available': True,
            'legitimacy_score': legitimacy_score,
            'closest_match': closest_match,
            'suspicious_indicators': suspicious_indicators
        }
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using simple character-based approach.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        # Simple character-based similarity
        set1 = set(text1.lower())
        set2 = set(text2.lower())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _assess_risk_level(self, asm_score: float) -> str:
        """
        Assess the risk level of authority signal manipulation.
        
        Args:
            asm_score: Authority signal manipulation score
            
        Returns:
            str: Risk level assessment
        """
        if asm_score > 0.8:
            return 'HIGH'
        elif asm_score > 0.6:
            return 'MEDIUM'
        elif asm_score > 0.4:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def generate_report(self, analysis_results: Dict) -> str:
        """
        Generate a human-readable report from analysis results.
        
        Args:
            analysis_results: Results from tweet analysis
            
        Returns:
            str: Formatted report string
        """
        if 'error' in analysis_results:
            return f"Error in analysis: {analysis_results['error']}"
        
        report = []
        report.append("=" * 60)
        report.append("AUTHORITY SIGNAL MANIPULATION (ASM) ANALYSIS REPORT")
        report.append("=" * 60)
        
        report.append(f"Tweet ID: {analysis_results['tweet_id']}")
        report.append(f"Author ID: {analysis_results['author_id']}")
        report.append(f"ASM Score: {analysis_results['asm_score']:.3f}")
        report.append(f"Risk Level: {analysis_results['risk_level']}")
        report.append(f"Is Manipulated: {analysis_results['is_manipulated']}")
        report.append("")
        
        # Lexical analysis
        lex_analysis = analysis_results['lexical_analysis']
        report.append("LEXICAL COMPLEXITY ANALYSIS:")
        report.append(f"  Complexity Score: {lex_analysis['complexity_score']:.3f}")
        report.append(f"  Vocabulary Diversity: {lex_analysis['vocabulary_diversity']:.3f}")
        report.append(f"  Sentence Complexity: {lex_analysis['sentence_complexity']:.1f} words/sentence")
        report.append(f"  Authority Word Density: {lex_analysis['authority_word_density']:.3f}")
        report.append(f"  Is Overly Complex: {lex_analysis['is_overly_complex']}")
        
        # Coherence analysis
        coh_analysis = analysis_results['coherence_analysis']
        report.append("")
        report.append("SEMANTIC COHERENCE ANALYSIS:")
        report.append(f"  Coherence Score: {coh_analysis['coherence_score']:.3f}")
        report.append(f"  Topic Consistency: {coh_analysis['topic_consistency']:.3f}")
        report.append(f"  Semantic Density: {coh_analysis['semantic_density']:.3f}")
        report.append(f"  Is Coherent: {coh_analysis['is_coherent']}")
        
        # Engagement analysis
        eng_analysis = analysis_results['engagement_analysis']
        report.append("")
        report.append("ENGAGEMENT QUALITY ANALYSIS:")
        report.append(f"  Engagement Quality: {eng_analysis['engagement_quality']:.3f}")
        report.append(f"  Bot Score: {eng_analysis['bot_score']:.3f}")
        report.append(f"  Engagement Rate: {eng_analysis['engagement_rate']:.3f}")
        report.append(f"  Bot Indicators: {', '.join(eng_analysis['bot_indicators']) if eng_analysis['bot_indicators'] else 'None'}")
        
        # Mismatch analysis
        mismatch_analysis = analysis_results['mismatch_analysis']
        report.append("")
        report.append("PROFILE-CONTENT MISMATCH ANALYSIS:")
        report.append(f"  Authority Signals: {mismatch_analysis['authority_signals']:.3f}")
        report.append(f"  Content Quality: {mismatch_analysis['content_quality']:.3f}")
        report.append(f"  Mismatch Score: {mismatch_analysis['mismatch_score']:.3f}")
        report.append(f"  Has Mismatch: {mismatch_analysis['has_mismatch']}")
        
        # Corpus comparison
        corpus_analysis = mismatch_analysis.get('corpus_comparison', {})
        if corpus_analysis.get('corpus_available', False):
            report.append("")
            report.append("AUTHORITY CORPUS COMPARISON:")
            report.append(f"  Legitimacy Score: {corpus_analysis['legitimacy_score']:.3f}")
            if corpus_analysis.get('closest_match'):
                match = corpus_analysis['closest_match']
                report.append(f"  Closest Match: {match['name']} ({match['title']} at {match['organization']})")
                report.append(f"  Match Similarity: {match['similarity']:.3f}")
            if corpus_analysis.get('suspicious_indicators'):
                report.append(f"  Suspicious Indicators: {', '.join(corpus_analysis['suspicious_indicators'])}")
        
        report.append("")
        report.append(f"Analysis Timestamp: {analysis_results['analysis_timestamp']}")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def analyze_tweet_by_id(self, tweet_id: str, connector: 'TwitterDataConnector') -> float:
        """
        Analyze a single tweet by ID and return ASM score from 0-1.
        
        Args:
            tweet_id: The tweet ID to analyze
            connector: TwitterDataConnector instance for database access
            
        Returns:
            float: ASM score between 0 and 1
        """
        try:
            # Get tweet data from database
            tweet_data = connector.get_tweet_data(tweet_id)
            
            if not tweet_data:
                return 0.0
            
            # Analyze the tweet
            analysis_results = self.analyze_tweet(tweet_data)
            
            # Return the ASM score
            return analysis_results.get('asm_score', 0.0)
            
        except Exception as e:
            print(f"Error analyzing tweet {tweet_id}: {str(e)}")
            return 0.0


def main():
    """
    Example usage of the Authority Signal Detector.
    """
    print("Authority Signal Manipulation (ASM) - Profile Analysis Complexity")
    print("=" * 60)
    
    # Initialize the detector
    detector = AuthoritySignalDetector(
        lexical_threshold=0.6,
        coherence_threshold=0.5,
        engagement_threshold=0.4
    )
    
    print("Detector initialized successfully!")
    print(f"Lexical threshold: {detector.lexical_threshold}")
    print(f"Coherence threshold: {detector.coherence_threshold}")
    print(f"Engagement threshold: {detector.engagement_threshold}")
    print("\nReady to analyze tweets for authority signal manipulation.")
    print("\nSIMPLE USAGE:")
    print("results = detector.analyze_tweet(tweet_data)")
    print("Returns: Comprehensive ASM analysis with score from 0-1")


if __name__ == "__main__":
    main()
