"""
Reply-Bait Detector (RBD) - Reply Pattern Complexity Analysis

This model detects reply-baiting behavior by analyzing:
1. Repetitive replies to own posts/comments
2. Sentiment inversion (negative post with positive replies)
3. Full thread analysis for pattern detection

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

class ReplyBaitDetector:
    """
    Detects reply-baiting behavior in Twitter conversations by analyzing
    reply patterns, sentiment inversion, and repetitive content.
    """
    
    def __init__(self, similarity_threshold: float = 0.7, 
                 sentiment_threshold: float = 0.3):
        """
        Initialize the Reply-Bait Detector.
        
        Args:
            similarity_threshold: Threshold for considering replies similar
            sentiment_threshold: Threshold for sentiment inversion detection
        """
        self.similarity_threshold = similarity_threshold
        self.sentiment_threshold = sentiment_threshold
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
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
    
    def calculate_sentiment(self, text: str) -> float:
        """
        Calculate sentiment score using TextBlob.
        
        Args:
            text: Input text
            
        Returns:
            Sentiment score between -1 (negative) and 1 (positive)
        """
        if not text or pd.isna(text):
            return 0.0
        
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def detect_repetitive_replies(self, replies: List[str]) -> Dict:
        """
        Detect repetitive reply patterns.
        
        Args:
            replies: List of reply texts
            
        Returns:
            Dictionary with repetitive reply analysis
        """
        if len(replies) < 2:
            return {
                'is_repetitive': False,
                'similarity_score': 0.0,
                'repetitive_pairs': [],
                'overall_similarity': 0.0
            }
        
        # Preprocess all replies
        processed_replies = [self.preprocess_text(reply) for reply in replies]
        
        # Remove empty replies
        processed_replies = [reply for reply in processed_replies if reply.strip()]
        
        if len(processed_replies) < 2:
            return {
                'is_repetitive': False,
                'similarity_score': 0.0,
                'repetitive_pairs': [],
                'overall_similarity': 0.0
            }
        
        # Vectorize replies
        try:
            tfidf_matrix = self.vectorizer.fit_transform(processed_replies)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find similar pairs
            repetitive_pairs = []
            total_similarity = 0
            pair_count = 0
            
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    similarity = similarity_matrix[i][j]
                    total_similarity += similarity
                    pair_count += 1
                    
                    if similarity >= self.similarity_threshold:
                        repetitive_pairs.append({
                            'reply1_idx': i,
                            'reply2_idx': j,
                            'similarity': similarity,
                            'reply1': replies[i],
                            'reply2': replies[j]
                        })
            
            overall_similarity = total_similarity / pair_count if pair_count > 0 else 0.0
            is_repetitive = len(repetitive_pairs) > 0
            
            return {
                'is_repetitive': is_repetitive,
                'similarity_score': overall_similarity,
                'repetitive_pairs': repetitive_pairs,
                'overall_similarity': overall_similarity,
                'total_replies': len(replies)
            }
            
        except Exception as e:
            return {
                'is_repetitive': False,
                'similarity_score': 0.0,
                'repetitive_pairs': [],
                'overall_similarity': 0.0,
                'error': str(e)
            }
    
    def detect_sentiment_inversion(self, main_post: str, replies: List[str]) -> Dict:
        """
        Detect sentiment inversion between main post and replies.
        
        Args:
            main_post: Main post text
            replies: List of reply texts
            
        Returns:
            Dictionary with sentiment inversion analysis
        """
        main_sentiment = self.calculate_sentiment(main_post)
        reply_sentiments = [self.calculate_sentiment(reply) for reply in replies]
        
        if not reply_sentiments:
            return {
                'has_inversion': False,
                'main_sentiment': main_sentiment,
                'avg_reply_sentiment': 0.0,
                'inversion_score': 0.0,
                'inverted_replies': []
            }
        
        avg_reply_sentiment = np.mean(reply_sentiments)
        
        # Check for sentiment inversion
        has_inversion = False
        inverted_replies = []
        
        for i, reply_sentiment in enumerate(reply_sentiments):
            # Detect inversion: negative main post with positive replies
            if main_sentiment < -self.sentiment_threshold and reply_sentiment > self.sentiment_threshold:
                has_inversion = True
                inverted_replies.append({
                    'reply_idx': i,
                    'reply_text': replies[i],
                    'reply_sentiment': reply_sentiment,
                    'inversion_type': 'negative_to_positive'
                })
            
            # Detect inversion: positive main post with negative replies
            elif main_sentiment > self.sentiment_threshold and reply_sentiment < -self.sentiment_threshold:
                has_inversion = True
                inverted_replies.append({
                    'reply_idx': i,
                    'reply_text': replies[i],
                    'reply_sentiment': reply_sentiment,
                    'inversion_type': 'positive_to_negative'
                })
        
        # Calculate inversion score
        if has_inversion:
            inversion_score = abs(main_sentiment - avg_reply_sentiment)
        else:
            inversion_score = 0.0
        
        return {
            'has_inversion': has_inversion,
            'main_sentiment': main_sentiment,
            'avg_reply_sentiment': avg_reply_sentiment,
            'inversion_score': inversion_score,
            'inverted_replies': inverted_replies,
            'total_replies': len(replies)
        }
    
    def analyze_conversation_thread(self, conversation_data: pd.DataFrame) -> Dict:
        """
        Analyze a complete conversation thread for reply-baiting patterns.
        
        Args:
            conversation_data: DataFrame with conversation data
            
        Returns:
            Dictionary with comprehensive thread analysis
        """
        if conversation_data.empty:
            return {'error': 'No conversation data provided'}
        
        # Sort by timestamp to maintain conversation order
        if 'created_at' in conversation_data.columns:
            conversation_data = conversation_data.sort_values('created_at')
        
        # Find the main post (first tweet in conversation)
        main_post = conversation_data.iloc[0]
        main_post_text = main_post.get('text', '')
        main_post_id = main_post.get('tweet_id', '')
        main_author = main_post.get('author_id', '')
        
        # Find replies (subsequent tweets in conversation)
        replies_data = conversation_data.iloc[1:]
        
        # Filter replies by the same author that are replies to their own main post
        # For reply-baiting, we want replies where the author is replying to their own tweet
        own_replies = replies_data[
            (replies_data['author_id'] == main_author) & 
            (replies_data['in_reply_to_user_id'] == main_author)
        ]
        other_replies = replies_data[
            (replies_data['author_id'] != main_author) | 
            (replies_data['in_reply_to_user_id'] != main_author)
        ]
        
        own_reply_texts = own_replies['text'].tolist() if not own_replies.empty else []
        other_reply_texts = other_replies['text'].tolist() if not other_replies.empty else []
        
        # Analyze repetitive patterns in own replies
        repetitive_analysis = self.detect_repetitive_replies(own_reply_texts)
        
        # Analyze sentiment inversion
        sentiment_analysis = self.detect_sentiment_inversion(main_post_text, own_reply_texts)
        
        # Calculate reply-baiting score
        reply_bait_score = self._calculate_reply_bait_score(
            repetitive_analysis, sentiment_analysis
        )
        
        return {
            'conversation_id': main_post.get('conversation_id', ''),
            'main_post_id': main_post_id,
            'main_author': main_author,
            'main_post_text': main_post_text,
            'total_replies': len(replies_data),
            'own_replies_count': len(own_replies),
            'other_replies_count': len(other_replies),
            'repetitive_analysis': repetitive_analysis,
            'sentiment_analysis': sentiment_analysis,
            'reply_bait_score': reply_bait_score,
            'is_reply_bait': reply_bait_score > 0.6,
            'analysis_timestamp': pd.Timestamp.now()
        }
    
    def _calculate_reply_bait_score(self, repetitive_analysis: Dict, 
                                   sentiment_analysis: Dict) -> float:
        """
        Calculate overall reply-baiting score.
        
        Args:
            repetitive_analysis: Results from repetitive reply detection
            sentiment_analysis: Results from sentiment inversion detection
            
        Returns:
            Score between 0 and 1 indicating likelihood of reply-baiting
        """
        score = 0.0
        
        # Repetitive reply component (40% weight)
        if repetitive_analysis['is_repetitive']:
            score += 0.4 * min(repetitive_analysis['overall_similarity'], 1.0)
        
        # Sentiment inversion component (40% weight)
        if sentiment_analysis['has_inversion']:
            score += 0.4 * min(sentiment_analysis['inversion_score'], 1.0)
        
        # Volume component (20% weight) - high number of own replies
        if repetitive_analysis.get('total_replies', 0) > 0:
            volume_score = min(repetitive_analysis['total_replies'] / 10.0, 1.0)
            score += 0.2 * volume_score
        
        return min(score, 1.0)
    
    def analyze_user_patterns(self, user_data: pd.DataFrame, 
                             min_conversations: int = 5) -> Dict:
        """
        Analyze reply-baiting patterns for a specific user across multiple conversations.
        
        Args:
            user_data: DataFrame with user's conversation data
            min_conversations: Minimum conversations required for analysis
            
        Returns:
            Dictionary with user-level reply-baiting analysis
        """
        if user_data.empty:
            return {'error': 'No user data provided'}
        
        # Group by conversation
        conversations = user_data.groupby('conversation_id')
        
        if len(conversations) < min_conversations:
            return {
                'error': f'Insufficient conversations ({len(conversations)} < {min_conversations})',
                'conversations_analyzed': len(conversations)
            }
        
        conversation_scores = []
        total_reply_bait_count = 0
        
        for conv_id, conv_data in conversations:
            conv_analysis = self.analyze_conversation_thread(conv_data)
            
            if 'error' not in conv_analysis:
                conversation_scores.append(conv_analysis['reply_bait_score'])
                if conv_analysis['is_reply_bait']:
                    total_reply_bait_count += 1
        
        if not conversation_scores:
            return {'error': 'No valid conversations found'}
        
        avg_score = np.mean(conversation_scores)
        reply_bait_rate = total_reply_bait_count / len(conversation_scores)
        
        return {
            'user_id': user_data['author_id'].iloc[0] if 'author_id' in user_data.columns else 'unknown',
            'total_conversations': len(conversations),
            'conversations_analyzed': len(conversation_scores),
            'average_reply_bait_score': avg_score,
            'reply_bait_rate': reply_bait_rate,
            'conversation_scores': conversation_scores,
            'risk_level': self._assess_risk_level(avg_score, reply_bait_rate),
            'analysis_timestamp': pd.Timestamp.now()
        }
    
    def _assess_risk_level(self, avg_score: float, reply_bait_rate: float) -> str:
        """
        Assess the risk level of reply-baiting behavior.
        
        Args:
            avg_score: Average reply-baiting score
            reply_bait_rate: Rate of conversations flagged as reply-bait
            
        Returns:
            Risk level assessment
        """
        if avg_score > 0.8 or reply_bait_rate > 0.7:
            return 'HIGH'
        elif avg_score > 0.6 or reply_bait_rate > 0.5:
            return 'MEDIUM'
        elif avg_score > 0.4 or reply_bait_rate > 0.3:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def generate_report(self, analysis_results: Dict) -> str:
        """
        Generate a human-readable report from analysis results.
        
        Args:
            analysis_results: Results from conversation or user analysis
            
        Returns:
            Formatted report string
        """
        if 'error' in analysis_results:
            return f"Error in analysis: {analysis_results['error']}"
        
        report = []
        report.append("=" * 60)
        report.append("REPLY-BAIT DETECTOR ANALYSIS REPORT")
        report.append("=" * 60)
        
        if 'conversation_id' in analysis_results:
            # Single conversation report
            report.append(f"Conversation ID: {analysis_results['conversation_id']}")
            report.append(f"Main Post ID: {analysis_results['main_post_id']}")
            report.append(f"Author: {analysis_results['main_author']}")
            report.append(f"Total Replies: {analysis_results['total_replies']}")
            report.append(f"Own Replies: {analysis_results['own_replies_count']}")
            report.append("")
            
            # Repetitive analysis
            rep_analysis = analysis_results['repetitive_analysis']
            report.append("REPETITIVE REPLY ANALYSIS:")
            report.append(f"  Is Repetitive: {rep_analysis['is_repetitive']}")
            report.append(f"  Overall Similarity: {rep_analysis['overall_similarity']:.3f}")
            if rep_analysis['repetitive_pairs']:
                report.append(f"  Repetitive Pairs Found: {len(rep_analysis['repetitive_pairs'])}")
            
            # Sentiment analysis
            sent_analysis = analysis_results['sentiment_analysis']
            report.append("")
            report.append("SENTIMENT ANALYSIS:")
            report.append(f"  Main Post Sentiment: {sent_analysis['main_sentiment']:.3f}")
            report.append(f"  Average Reply Sentiment: {sent_analysis['avg_reply_sentiment']:.3f}")
            report.append(f"  Has Sentiment Inversion: {sent_analysis['has_inversion']}")
            if sent_analysis['inverted_replies']:
                report.append(f"  Inverted Replies: {len(sent_analysis['inverted_replies'])}")
            
            # Overall assessment
            report.append("")
            report.append("OVERALL ASSESSMENT:")
            report.append(f"  Reply-Bait Score: {analysis_results['reply_bait_score']:.3f}")
            report.append(f"  Is Reply-Bait: {analysis_results['is_reply_bait']}")
            
        else:
            # User-level report
            report.append(f"User ID: {analysis_results['user_id']}")
            report.append(f"Total Conversations: {analysis_results['total_conversations']}")
            report.append(f"Conversations Analyzed: {analysis_results['conversations_analyzed']}")
            report.append("")
            report.append(f"Average Reply-Bait Score: {analysis_results['average_reply_bait_score']:.3f}")
            report.append(f"Reply-Bait Rate: {analysis_results['reply_bait_rate']:.3f}")
            report.append(f"Risk Level: {analysis_results['risk_level']}")
        
        report.append("")
        report.append(f"Analysis Timestamp: {analysis_results['analysis_timestamp']}")
        report.append("=" * 60)
        
        return "\n".join(report)

    def analyze_tweet(self, tweet_id: str, connector: 'TwitterDataConnector') -> float:
        """
        Analyze a single tweet and return a reply-bait score from 0-1.
        
        Args:
            tweet_id: The tweet ID to analyze
            connector: TwitterDataConnector instance for database access
            
        Returns:
            float: Reply-bait score between 0 and 1
        """
        try:
            # Get the conversation thread for this tweet
            conversation_data = connector.get_conversation_thread(tweet_id)
            
            if conversation_data.empty:
                return 0.0
            
            # Analyze the conversation thread
            analysis_results = self.analyze_conversation_thread(conversation_data)
            
            if 'error' in analysis_results:
                return 0.0
            
            # Return the reply-bait score
            return analysis_results.get('reply_bait_score', 0.0)
            
        except Exception as e:
            print(f"Error analyzing tweet {tweet_id}: {str(e)}")
            return 0.0


def main():
    """
    Example usage of the Reply-Bait Detector.
    """
    print("Reply-Bait Detector (RBD) - Reply Pattern Complexity Analysis")
    print("=" * 60)
    
    # Initialize the detector
    detector = ReplyBaitDetector(
        similarity_threshold=0.7,
        sentiment_threshold=0.3
    )
    
    print("Detector initialized successfully!")
    print(f"Similarity threshold: {detector.similarity_threshold}")
    print(f"Sentiment threshold: {detector.sentiment_threshold}")
    print("\nReady to analyze Twitter conversations for reply-baiting patterns.")
    print("\nSIMPLE USAGE:")
    print("score = detector.analyze_tweet('tweet_id', connector)")
    print("Returns: Single score from 0-1 (0 = no reply-baiting, 1 = high reply-baiting)")
    print("\nADVANCED USAGE:")
    print("Use the analyze_conversation_thread() method for detailed conversation analysis")
    print("Use the analyze_user_patterns() method for user-level analysis")


if __name__ == "__main__":
    main()
