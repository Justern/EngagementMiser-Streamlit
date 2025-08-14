import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import pyodbc
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CoordinatedAccountNetworkModel:
    """
    Coordinated Account Network Model for detecting coordinated behavior
    using user interaction patterns and follower networks.
    """
    
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.is_trained = False
        self.scaler = StandardScaler()
        self.clustering_model = None
        self.feature_columns = None
        
    def preprocess_network_data(self, df):
        """
        Preprocess network data for analysis.
        
        Args:
            df (pd.DataFrame): Raw network data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        if df.empty:
            return df
            
        # Convert to numeric where possible
        numeric_columns = ['follower_count', 'following_count', 'tweet_count', 
                          'like_count', 'retweet_count', 'reply_count']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Handle missing values
        df = df.fillna(0)
        
        return df
    
    def extract_network_features(self, df):
        """
        Extract features for network analysis.
        
        Args:
            df (pd.DataFrame): Preprocessed network data
            
        Returns:
            pd.DataFrame: Features for clustering
        """
        if df.empty:
            return pd.DataFrame()
        
        features = df.copy()
        
        # Basic network metrics
        if 'follower_count' in features.columns and 'following_count' in features.columns:
            features['follower_following_ratio'] = features['follower_count'] / (features['following_count'] + 1)
            features['network_balance'] = abs(features['follower_count'] - features['following_count'])
        
        # Engagement metrics
        if all(col in features.columns for col in ['like_count', 'retweet_count', 'reply_count']):
            features['total_engagement'] = features['like_count'] + features['retweet_count'] + features['reply_count']
            features['engagement_rate'] = features['total_engagement'] / (features['tweet_count'] + 1)
        
        # Activity patterns
        if 'tweet_count' in features.columns:
            features['activity_level'] = np.log1p(features['tweet_count'])
        
        # Account age (if available)
        if 'created_at' in features.columns:
            try:
                features['created_at'] = pd.to_datetime(features['created_at'])
                features['account_age_days'] = (datetime.now() - features['created_at']).dt.days
                features['tweets_per_day'] = features['tweet_count'] / (features['account_age_days'] + 1)
            except:
                features['account_age_days'] = 0
                features['tweets_per_day'] = 0
        
        # Select numeric features for clustering
        numeric_features = features.select_dtypes(include=[np.number])
        
        # Remove infinite values
        numeric_features = numeric_features.replace([np.inf, -np.inf], 0)
        
        return numeric_features
    
    def build_interaction_graph(self, interactions_df):
        """
        Build interaction graph from user interactions.
        
        Args:
            interactions_df (pd.DataFrame): User interaction data
            
        Returns:
            networkx.Graph: Interaction graph
        """
        if interactions_df.empty:
            return nx.Graph()
        
        G = nx.Graph()
        
        # Add nodes (users)
        unique_users = set(interactions_df['user_id'].unique()) | set(interactions_df['interacted_with_id'].unique())
        G.add_nodes_from(unique_users)
        
        # Add edges (interactions)
        for _, row in interactions_df.iterrows():
            user_id = row['user_id']
            interacted_with_id = row['interacted_with_id']
            weight = row.get('interaction_count', 1)
            
            if user_id != interacted_with_id:
                if G.has_edge(user_id, interacted_with_id):
                    G[user_id][interacted_with_id]['weight'] += weight
                else:
                    G.add_edge(user_id, interacted_with_id, weight=weight)
        
        return G
    
    def calculate_network_metrics(self, G):
        """
        Calculate network-level metrics for coordination detection.
        
        Args:
            G (networkx.Graph): Interaction graph
            
        Returns:
            dict: Network metrics
        """
        if not G.nodes():
            return {
                'node_count': 0,
                'edge_count': 0,
                'density': 0,
                'clustering_coefficient': 0,
                'average_degree': 0,
                'modularity': 0
            }
        
        metrics = {}
        
        # Basic metrics
        metrics['node_count'] = G.number_of_nodes()
        metrics['edge_count'] = G.number_of_edges()
        metrics['density'] = nx.density(G)
        metrics['clustering_coefficient'] = nx.average_clustering(G)
        metrics['average_degree'] = sum(dict(G.degree()).values()) / G.number_of_nodes()
        
        # Community detection for modularity
        try:
            communities = nx.community.greedy_modularity_communities(G)
            modularity = nx.community.modularity(G, communities)
            metrics['modularity'] = modularity
        except:
            metrics['modularity'] = 0
        
        return metrics
    
    def detect_coordinated_clusters(self, features_df, method='dbscan'):
        """
        Detect coordinated account clusters using clustering algorithms.
        
        Args:
            features_df (pd.DataFrame): Network features
            method (str): Clustering method ('dbscan' or 'kmeans')
            
        Returns:
            tuple: (cluster_labels, clustering_model)
        """
        if features_df.empty or features_df.shape[0] < 2:
            return np.zeros(features_df.shape[0]), None
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features_df)
        
        if method == 'dbscan':
            # DBSCAN for density-based clustering
            clustering = DBSCAN(eps=0.5, min_samples=2)
        else:
            # K-means for centroid-based clustering
            n_clusters = min(5, features_df.shape[0] // 2)
            clustering = KMeans(n_clusters=max(2, n_clusters), random_state=42)
        
        cluster_labels = clustering.fit_predict(scaled_features)
        self.clustering_model = clustering
        self.feature_columns = features_df.columns.tolist()
        
        return cluster_labels, clustering
    
    def analyze_user_coordination(self, user_id, user_data, interactions_data):
        """
        Analyze coordination patterns for a specific user.
        
        Args:
            user_id (str): User ID to analyze
            user_data (pd.DataFrame): User profile data
            interactions_data (pd.DataFrame): User interaction data
            
        Returns:
            dict: Coordination analysis results
        """
        if user_data.empty:
            return {
                'coordination_score': 0.0,
                'coordination_level': 'NO_DATA',
                'cluster_id': None,
                'network_metrics': {},
                'suspicious_patterns': []
            }
        
        # Build interaction graph
        G = self.build_interaction_graph(interactions_data)
        
        # Calculate network metrics
        network_metrics = self.calculate_network_metrics(G)
        
        # Extract features for the user
        features = self.extract_network_features(user_data)
        
        if features.empty:
            return {
                'coordination_score': 0.0,
                'coordination_level': 'NO_FEATURES',
                'cluster_id': None,
                'network_metrics': network_metrics,
                'suspicious_patterns': []
            }
        
        # Detect clusters
        cluster_labels, _ = self.detect_coordinated_clusters(features)
        
        # Find user's cluster
        user_index = user_data.index[user_data.index == user_id]
        if len(user_index) > 0:
            cluster_id = cluster_labels[user_index[0]] if user_index[0] < len(cluster_labels) else -1
        else:
            cluster_id = -1
        
        # Calculate coordination score based on multiple factors
        coordination_score = self._calculate_coordination_score(
            user_data.iloc[0] if not user_data.empty else None,
            network_metrics,
            cluster_id,
            interactions_data
        )
        
        # Determine coordination level
        coordination_level = self._assess_coordination_level(coordination_score)
        
        # Identify suspicious patterns
        suspicious_patterns = self._identify_suspicious_patterns(
            user_data.iloc[0] if not user_data.empty else None,
            network_metrics,
            interactions_data
        )
        
        return {
            'coordination_score': coordination_score,
            'coordination_level': coordination_level,
            'cluster_id': int(cluster_id) if cluster_id is not None else None,
            'network_metrics': network_metrics,
            'suspicious_patterns': suspicious_patterns
        }
    
    def _calculate_coordination_score(self, user_profile, network_metrics, cluster_id, interactions_data):
        """
        Calculate coordination score based on multiple factors.
        
        Args:
            user_profile (pd.Series): User profile data
            network_metrics (dict): Network-level metrics
            cluster_id (int): User's cluster ID
            interactions_data (pd.DataFrame): Interaction data
            
        Returns:
            float: Coordination score (0-1)
        """
        score = 0.0
        
        # Network density factor (higher density = more coordination)
        if network_metrics['density'] > 0:
            score += min(network_metrics['density'] * 2, 0.3)
        
        # Clustering coefficient factor (higher clustering = more coordination)
        if network_metrics['clustering_coefficient'] > 0:
            score += min(network_metrics['clustering_coefficient'] * 1.5, 0.2)
        
        # Modularity factor (higher modularity = more distinct communities)
        if network_metrics['modularity'] > 0:
            score += min(network_metrics['modularity'] * 1.0, 0.2)
        
        # Cluster membership factor
        if cluster_id is not None and cluster_id >= 0:
            score += 0.1
        
        # Suspicious pattern factors
        if user_profile is not None:
            # Account age suspiciousness
            if 'account_age_days' in user_profile:
                if user_profile['account_age_days'] < 30:  # Very new account
                    score += 0.1
                elif user_profile['account_age_days'] < 90:  # New account
                    score += 0.05
            
            # Follower-following ratio suspiciousness
            if 'follower_following_ratio' in user_profile:
                if user_profile['follower_following_ratio'] > 10:  # Suspicious ratio
                    score += 0.1
                elif user_profile['follower_following_ratio'] > 5:
                    score += 0.05
            
            # Activity pattern suspiciousness
            if 'tweets_per_day' in user_profile:
                if user_profile['tweets_per_day'] > 50:  # Excessive activity
                    score += 0.1
                elif user_profile['tweets_per_day'] > 20:
                    score += 0.05
        
        # Interaction pattern suspiciousness
        if not interactions_data.empty:
            # Check for repetitive interactions
            if len(interactions_data) > 100:  # High interaction volume
                score += 0.1
            
            # Check for interaction timing patterns
            if 'created_at' in interactions_data.columns:
                try:
                    interactions_data['created_at'] = pd.to_datetime(interactions_data['created_at'])
                    time_diffs = interactions_data['created_at'].diff().dropna()
                    if len(time_diffs) > 0:
                        avg_time_diff = time_diffs.mean()
                        if avg_time_diff < timedelta(minutes=1):  # Very rapid interactions
                            score += 0.1
                        elif avg_time_diff < timedelta(minutes=5):  # Rapid interactions
                            score += 0.05
                except:
                    pass
        
        return min(score, 1.0)
    
    def _assess_coordination_level(self, score):
        """
        Assess coordination level based on score.
        
        Args:
            score (float): Coordination score (0-1)
            
        Returns:
            str: Coordination level
        """
        if score >= 0.8:
            return 'VERY_HIGH_COORDINATION'
        elif score >= 0.6:
            return 'HIGH_COORDINATION'
        elif score >= 0.4:
            return 'MODERATE_COORDINATION'
        elif score >= 0.2:
            return 'LOW_COORDINATION'
        else:
            return 'NO_COORDINATION'
    
    def _identify_suspicious_patterns(self, user_profile, network_metrics, interactions_data):
        """
        Identify suspicious patterns that indicate coordination.
        
        Args:
            user_profile (pd.Series): User profile data
            network_metrics (dict): Network metrics
            interactions_data (pd.DataFrame): Interaction data
            
        Returns:
            list: List of suspicious patterns
        """
        patterns = []
        
        # Network-level patterns
        if network_metrics['density'] > 0.8:
            patterns.append("Extremely high network density")
        elif network_metrics['density'] > 0.5:
            patterns.append("High network density")
        
        if network_metrics['clustering_coefficient'] > 0.8:
            patterns.append("Extremely high clustering coefficient")
        elif network_metrics['clustering_coefficient'] > 0.6:
            patterns.append("High clustering coefficient")
        
        if network_metrics['modularity'] > 0.7:
            patterns.append("High network modularity")
        
        # User-level patterns
        if user_profile is not None:
            if 'account_age_days' in user_profile and user_profile['account_age_days'] < 30:
                patterns.append("Very new account (< 30 days)")
            elif 'account_age_days' in user_profile and user_profile['account_age_days'] < 90:
                patterns.append("New account (< 90 days)")
            
            if 'follower_following_ratio' in user_profile and user_profile['follower_following_ratio'] > 10:
                patterns.append("Suspicious follower-following ratio")
            
            if 'tweets_per_day' in user_profile and user_profile['tweets_per_day'] > 50:
                patterns.append("Excessive posting activity")
        
        # Interaction patterns
        if not interactions_data.empty:
            if len(interactions_data) > 100:
                patterns.append("High interaction volume")
            
            if 'created_at' in interactions_data.columns:
                try:
                    interactions_data['created_at'] = pd.to_datetime(interactions_data['created_at'])
                    time_diffs = interactions_data['created_at'].diff().dropna()
                    if len(time_diffs) > 0:
                        avg_time_diff = time_diffs.mean()
                        if avg_time_diff < timedelta(minutes=1):
                            patterns.append("Very rapid interaction timing")
                        elif avg_time_diff < timedelta(minutes=5):
                            patterns.append("Rapid interaction timing")
                except:
                    pass
        
        return patterns
    
    def analyze_tweet_coordination(self, tweet_id, user_data, interactions_data):
        """
        Analyze coordination patterns for a specific tweet.
        
        Args:
            tweet_id (str): Tweet ID to analyze
            user_data (pd.DataFrame): User data related to the tweet
            interactions_data (pd.DataFrame): Interaction data related to the tweet
            
        Returns:
            dict: Tweet coordination analysis results
        """
        if user_data.empty:
            return {
                'coordination_score': 0.0,
                'coordination_level': 'NO_DATA',
                'network_metrics': {},
                'suspicious_patterns': []
            }
        
        # Build interaction graph for this tweet
        G = self.build_interaction_graph(interactions_data)
        
        # Calculate network metrics
        network_metrics = self.calculate_network_metrics(G)
        
        # Extract features
        features = self.extract_network_features(user_data)
        
        if features.empty:
            return {
                'coordination_score': 0.0,
                'coordination_level': 'NO_FEATURES',
                'cluster_id': None,
                'network_metrics': network_metrics,
                'suspicious_patterns': []
            }
        
        # Detect clusters
        cluster_labels, _ = self.detect_coordinated_clusters(features)
        
        # Calculate coordination score
        coordination_score = self._calculate_coordination_score(
            user_data.iloc[0] if not user_data.empty else None,
            network_metrics,
            cluster_labels[0] if len(cluster_labels) > 0 else -1,
            interactions_data
        )
        
        # Determine coordination level
        coordination_level = self._assess_coordination_level(coordination_score)
        
        # Identify suspicious patterns
        suspicious_patterns = self._identify_suspicious_patterns(
            user_data.iloc[0] if not user_data.empty else None,
            network_metrics,
            interactions_data
        )
        
        return {
            'coordination_score': coordination_score,
            'coordination_level': coordination_level,
            'cluster_id': int(cluster_labels[0]) if len(cluster_labels) > 0 else None,
            'network_metrics': network_metrics,
            'suspicious_patterns': suspicious_patterns
        }
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        model_data = {
            'scaler': self.scaler,
            'clustering_model': self.clustering_model,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        try:
            model_data = joblib.load(filepath)
            self.scaler = model_data['scaler']
            self.clustering_model = model_data['clustering_model']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
