"""
Rapid Engagement Spike Detector

This model detects unusual engagement spikes and anomalies in social media data by analyzing:
1. Time series engagement metrics (likes, retweets, replies, etc.)
2. Statistical anomalies and changepoints
3. Rapid engagement velocity changes
4. Unusual engagement patterns over time
5. Bot-like or coordinated engagement behavior

Author: DS Capstone Project
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Time series analysis libraries
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

# Changepoint detection
try:
    from ruptures import Binseg, Pelt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    print("Warning: ruptures library not available. Install with: pip install ruptures")

class RapidEngagementSpikeDetector:
    """
    Detects rapid engagement spikes and anomalies using time series analysis
    and statistical methods.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Rapid Engagement Spike Detector.
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        # Initialize components
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        # Spike detection parameters
        self.spike_threshold = 2.0  # Standard deviations for spike detection
        self.velocity_threshold = 1.5  # Threshold for rapid velocity changes
        self.min_spike_duration = 3  # Minimum time points for a spike
        
        # Changepoint detection
        self.changepoint_penalty = 10  # Penalty for changepoint detection
        
        # Load pre-trained model if provided
        if model_path:
            self.load_model(model_path)
        else:
            self.is_trained = False
    
    def preprocess_engagement_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess engagement time series data.
        
        Args:
            data: DataFrame with timestamp and engagement columns
            
        Returns:
            pd.DataFrame: Cleaned and preprocessed data
        """
        if data.empty:
            return data
        
        # Ensure timestamp column exists and is datetime
        if 'timestamp' not in data.columns and 'created_at' in data.columns:
            data = data.rename(columns={'created_at': 'timestamp'})
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates and handle missing values
        data = data.drop_duplicates(subset=['timestamp'])
        data = data.dropna()
        
        # Ensure we have engagement metrics
        engagement_cols = ['likes', 'retweets', 'replies', 'quotes', 'total_engagements']
        available_cols = [col for col in engagement_cols if col in data.columns]
        
        if not available_cols:
            raise ValueError("No engagement columns found in data")
        
        # Fill missing engagement values with 0
        for col in available_cols:
            data[col] = data[col].fillna(0)
        
        return data
    
    def calculate_engagement_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional engagement metrics and features.
        
        Args:
            data: Preprocessed engagement data
            
        Returns:
            pd.DataFrame: Data with additional metrics
        """
        if data.empty:
            return data
        
        # Calculate total engagements if not present
        if 'total_engagements' not in data.columns:
            engagement_cols = ['likes', 'retweets', 'replies', 'quotes']
            available_cols = [col for col in engagement_cols if col in data.columns]
            if available_cols:
                data['total_engagements'] = data[available_cols].sum(axis=1)
        
        # Calculate engagement velocity (rate of change)
        data['engagement_velocity'] = data['total_engagements'].diff()
        data['engagement_acceleration'] = data['engagement_velocity'].diff()
        
        # Calculate rolling statistics
        window_sizes = [5, 10, 20]
        for window in window_sizes:
            if len(data) >= window:
                data[f'rolling_mean_{window}'] = data['total_engagements'].rolling(window=window).mean()
                data[f'rolling_std_{window}'] = data['total_engagements'].rolling(window=window).std()
                data[f'rolling_zscore_{window}'] = (
                    (data['total_engagements'] - data[f'rolling_mean_{window}']) / 
                    data[f'rolling_std_{window}']
                )
        
        # Calculate percentage changes
        data['engagement_pct_change'] = data['total_engagements'].pct_change() * 100
        
        # Calculate cumulative engagement
        data['cumulative_engagement'] = data['total_engagements'].cumsum()
        
        return data
    
    def detect_statistical_anomalies(self, data: pd.DataFrame) -> Dict:
        """
        Detect statistical anomalies in engagement data.
        
        Args:
            data: Engagement data with calculated metrics
            
        Returns:
            Dict: Anomaly detection results
        """
        if data.empty or 'total_engagements' not in data.columns:
            return {'anomalies': [], 'anomaly_score': 0.0}
        
        # Prepare features for anomaly detection
        feature_cols = ['total_engagements', 'engagement_velocity', 'engagement_acceleration']
        available_features = [col for col in feature_cols if col in data.columns]
        
        if not available_features:
            return {'anomalies': [], 'anomaly_score': 0.0}
        
        # Extract features and handle infinite values
        features = data[available_features].copy()
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Detect anomalies using Isolation Forest
        anomaly_labels = self.isolation_forest.fit_predict(features_scaled)
        anomaly_scores = self.isolation_forest.decision_function(features_scaled)
        
        # Find anomaly indices
        anomaly_indices = np.where(anomaly_labels == -1)[0]
        
        # Calculate overall anomaly score
        anomaly_score = np.mean(np.abs(anomaly_scores)) if len(anomaly_scores) > 0 else 0.0
        
        # Get anomaly details
        anomalies = []
        for idx in anomaly_indices:
            if idx < len(data):
                anomalies.append({
                    'timestamp': data.iloc[idx]['timestamp'],
                    'total_engagements': data.iloc[idx]['total_engagements'],
                    'anomaly_score': abs(anomaly_scores[idx]),
                    'features': features.iloc[idx].to_dict()
                })
        
        return {
            'anomalies': anomalies,
            'anomaly_score': anomaly_score,
            'anomaly_count': len(anomalies)
        }
    
    def detect_engagement_spikes(self, data: pd.DataFrame) -> Dict:
        """
        Detect rapid engagement spikes using statistical methods.
        
        Args:
            data: Engagement data with calculated metrics
            
        Returns:
            Dict: Spike detection results
        """
        if data.empty or 'total_engagements' not in data.columns:
            return {'spikes': [], 'spike_score': 0.0}
        
        spikes = []
        spike_scores = []
        
        # Method 1: Z-score based spike detection
        if 'rolling_zscore_10' in data.columns:
            zscore_spikes = data[data['rolling_zscore_10'].abs() > self.spike_threshold]
            for _, row in zscore_spikes.iterrows():
                spikes.append({
                    'timestamp': row['timestamp'],
                    'total_engagements': row['total_engagements'],
                    'zscore': row['rolling_zscore_10'],
                    'method': 'zscore',
                    'spike_score': abs(row['rolling_zscore_10'])
                })
                spike_scores.append(abs(row['rolling_zscore_10']))
        
        # Method 2: Velocity-based spike detection
        if 'engagement_velocity' in data.columns:
            velocity_data = data[data['engagement_velocity'].notna()].copy()
            if not velocity_data.empty:
                velocity_mean = velocity_data['engagement_velocity'].mean()
                velocity_std = velocity_data['engagement_velocity'].std()
                
                if velocity_std > 0:
                    velocity_zscore = (velocity_data['engagement_velocity'] - velocity_mean) / velocity_std
                    velocity_spikes = velocity_data[velocity_zscore.abs() > self.velocity_threshold]
                    
                    for _, row in velocity_spikes.iterrows():
                        spikes.append({
                            'timestamp': row['timestamp'],
                            'total_engagements': row['total_engagements'],
                            'velocity': row['engagement_velocity'],
                            'velocity_zscore': velocity_zscore.loc[row.name],
                            'method': 'velocity',
                            'spike_score': abs(velocity_zscore.loc[row.name])
                        })
                        spike_scores.append(abs(velocity_zscore.loc[row.name]))
        
        # Method 3: Peak detection using scipy
        try:
            engagement_series = data['total_engagements'].values
            if len(engagement_series) > 10:
                peaks, properties = find_peaks(engagement_series, height=np.mean(engagement_series))
                
                for peak_idx in peaks:
                    if peak_idx < len(data):
                        row = data.iloc[peak_idx]
                        peak_height = engagement_series[peak_idx]
                        peak_score = peak_height / np.mean(engagement_series) if np.mean(engagement_series) > 0 else 0
                        
                        spikes.append({
                            'timestamp': row['timestamp'],
                            'total_engagements': row['total_engagements'],
                            'peak_height': peak_height,
                            'method': 'peak_detection',
                            'spike_score': peak_score
                        })
                        spike_scores.append(peak_score)
        except Exception as e:
            print(f"Peak detection error: {e}")
        
        # Remove duplicate spikes (same timestamp)
        unique_spikes = {}
        for spike in spikes:
            timestamp_key = spike['timestamp']
            if timestamp_key not in unique_spikes or spike['spike_score'] > unique_spikes[timestamp_key]['spike_score']:
                unique_spikes[timestamp_key] = spike
        
        spikes = list(unique_spikes.values())
        
        # Calculate overall spike score
        spike_score = np.mean(spike_scores) if spike_scores else 0.0
        
        return {
            'spikes': spikes,
            'spike_score': spike_score,
            'spike_count': len(spikes)
        }
    
    def detect_changepoints(self, data: pd.DataFrame) -> Dict:
        """
        Detect changepoints in engagement time series.
        
        Args:
            data: Engagement data
            
        Returns:
            Dict: Changepoint detection results
        """
        if not RUPTURES_AVAILABLE or data.empty or 'total_engagements' not in data.columns:
            return {'changepoints': [], 'changepoint_score': 0.0}
        
        try:
            # Prepare data for changepoint detection
            engagement_series = data['total_engagements'].values
            
            if len(engagement_series) < 10:
                return {'changepoints': [], 'changepoint_score': 0.0}
            
            # Use Pelt algorithm for changepoint detection
            algo = Pelt(model="l2", jump=1)
            algo.fit(engagement_series)
            changepoint_indices = algo.predict(pen=self.changepoint_penalty)
            
            # Remove the last index (end of series)
            if changepoint_indices and changepoint_indices[-1] == len(engagement_series):
                changepoint_indices = changepoint_indices[:-1]
            
            changepoints = []
            for idx in changepoint_indices:
                if idx < len(data):
                    row = data.iloc[idx]
                    changepoints.append({
                        'timestamp': row['timestamp'],
                        'total_engagements': row['total_engagements'],
                        'index': idx,
                        'changepoint_score': 1.0  # Binary detection
                    })
            
            changepoint_score = len(changepoints) / max(len(engagement_series), 1)
            
            return {
                'changepoints': changepoints,
                'changepoint_score': changepoint_score,
                'changepoint_count': len(changepoints)
            }
            
        except Exception as e:
            print(f"Changepoint detection error: {e}")
            return {'changepoints': [], 'changepoint_score': 0.0}
    
    def analyze_engagement_patterns(self, data: pd.DataFrame) -> Dict:
        """
        Analyze overall engagement patterns and trends.
        
        Args:
            data: Engagement data
            
        Returns:
            Dict: Pattern analysis results
        """
        if data.empty or 'total_engagements' not in data.columns:
            return {'pattern_score': 0.0, 'trends': {}}
        
        patterns = {}
        
        # Basic statistics
        patterns['total_engagement'] = data['total_engagements'].sum()
        patterns['mean_engagement'] = data['total_engagements'].mean()
        patterns['std_engagement'] = data['total_engagements'].std()
        patterns['max_engagement'] = data['total_engagements'].max()
        patterns['min_engagement'] = data['total_engagements'].min()
        
        # Trend analysis
        if len(data) > 1:
            # Linear trend
            x = np.arange(len(data))
            y = data['total_engagements'].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            patterns['trend_slope'] = slope
            patterns['trend_r_squared'] = r_value ** 2
            patterns['trend_p_value'] = p_value
            
            # Trend direction
            if slope > 0.01:
                patterns['trend_direction'] = 'increasing'
            elif slope < -0.01:
                patterns['trend_direction'] = 'decreasing'
            else:
                patterns['trend_direction'] = 'stable'
        
        # Volatility analysis
        if 'engagement_pct_change' in data.columns:
            volatility = data['engagement_pct_change'].std()
            patterns['volatility'] = volatility
            
            # Volatility classification
            if volatility > 50:
                patterns['volatility_level'] = 'high'
            elif volatility > 20:
                patterns['volatility_level'] = 'medium'
            else:
                patterns['volatility_level'] = 'low'
        
        # Calculate overall pattern score
        pattern_score = 0.0
        
        # Higher score for more volatile/unusual patterns
        if patterns.get('volatility_level') == 'high':
            pattern_score += 0.4
        elif patterns.get('volatility_level') == 'medium':
            pattern_score += 0.2
        
        # Higher score for strong trends
        if patterns.get('trend_r_squared', 0) > 0.7:
            pattern_score += 0.3
        elif patterns.get('trend_r_squared', 0) > 0.4:
            pattern_score += 0.2
        
        # Higher score for high engagement variance
        if patterns.get('std_engagement', 0) > patterns.get('mean_engagement', 1):
            pattern_score += 0.3
        
        patterns['pattern_score'] = min(pattern_score, 1.0)
        
        return patterns
    
    def analyze_tweet_engagement(self, tweet_id: str, engagement_data: pd.DataFrame) -> Dict:
        """
        Comprehensive analysis of engagement patterns for a specific tweet.
        
        Args:
            tweet_id: ID of the tweet to analyze
            engagement_data: Time series engagement data
            
        Returns:
            Dict: Comprehensive analysis results
        """
        print(f"🔍 Analyzing engagement patterns for tweet: {tweet_id}")
        
        # Preprocess data
        processed_data = self.preprocess_engagement_data(engagement_data)
        if processed_data.empty:
            return {'error': 'No valid engagement data found'}
        
        # Calculate additional metrics
        enhanced_data = self.calculate_engagement_metrics(processed_data)
        
        # Perform various analyses
        anomaly_results = self.detect_statistical_anomalies(enhanced_data)
        spike_results = self.detect_engagement_spikes(enhanced_data)
        changepoint_results = self.detect_changepoints(enhanced_data)
        pattern_results = self.analyze_engagement_patterns(enhanced_data)
        
        # Calculate overall spike detection score
        spike_detection_score = 0.0
        
        # Weight different components
        if anomaly_results['anomaly_count'] > 0:
            spike_detection_score += 0.3 * min(anomaly_results['anomaly_score'], 1.0)
        
        if spike_results['spike_count'] > 0:
            spike_detection_score += 0.4 * min(spike_results['spike_score'], 1.0)
        
        if changepoint_results['changepoint_count'] > 0:
            spike_detection_score += 0.2 * changepoint_results['changepoint_score']
        
        if pattern_results.get('pattern_score', 0) > 0:
            spike_detection_score += 0.1 * pattern_results['pattern_score']
        
        # Normalize to 0-1 range
        spike_detection_score = min(spike_detection_score, 1.0)
        
        # Determine spike level
        if spike_detection_score > 0.7:
            spike_level = 'HIGH_SPIKE'
        elif spike_detection_score > 0.4:
            spike_level = 'MODERATE_SPIKE'
        elif spike_detection_score > 0.2:
            spike_level = 'LOW_SPIKE'
        else:
            spike_level = 'NO_SPIKE'
        
        return {
            'tweet_id': tweet_id,
            'spike_detection_score': spike_detection_score,
            'spike_level': spike_level,
            'analysis_timestamp': pd.Timestamp.now(),
            'data_points': len(enhanced_data),
            'time_range': {
                'start': enhanced_data['timestamp'].min(),
                'end': enhanced_data['timestamp'].max(),
                'duration_hours': (enhanced_data['timestamp'].max() - enhanced_data['timestamp'].min()).total_seconds() / 3600
            },
            'anomaly_detection': anomaly_results,
            'spike_detection': spike_results,
            'changepoint_detection': changepoint_results,
            'pattern_analysis': pattern_results,
            'raw_data': enhanced_data.to_dict('records') if len(enhanced_data) <= 100 else None
        }
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
        """
        import joblib
        model_data = {
            'scaler': self.scaler,
            'isolation_forest': self.isolation_forest,
            'spike_threshold': self.spike_threshold,
            'velocity_threshold': self.velocity_threshold,
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
        
        self.scaler = model_data['scaler']
        self.isolation_forest = model_data['isolation_forest']
        self.spike_threshold = model_data.get('spike_threshold', 2.0)
        self.velocity_threshold = model_data.get('velocity_threshold', 1.5)
        self.is_trained = model_data.get('is_trained', True)
        
        print(f"Model loaded from {model_path}")


def main():
    """
    Example usage of the Rapid Engagement Spike Detector.
    """
    print("Rapid Engagement Spike Detector")
    print("=" * 50)
    
    # Initialize detector
    detector = RapidEngagementSpikeDetector()
    
    # Example synthetic data
    print("Creating example engagement data...")
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    
    # Normal engagement pattern with some spikes
    base_engagement = 100
    normal_pattern = base_engagement + np.random.normal(0, 20, 100)
    
    # Add some spikes
    normal_pattern[30:35] += np.random.normal(200, 50, 5)  # Spike 1
    normal_pattern[60:65] += np.random.normal(300, 80, 5)  # Spike 2
    
    example_data = pd.DataFrame({
        'timestamp': dates,
        'total_engagements': normal_pattern,
        'likes': normal_pattern * 0.7,
        'retweets': normal_pattern * 0.2,
        'replies': normal_pattern * 0.1
    })
    
    print(f"Example data created with {len(example_data)} data points")
    print(f"Time range: {example_data['timestamp'].min()} to {example_data['timestamp'].max()}")
    
    # Analyze the data
    results = detector.analyze_tweet_engagement("example_tweet_123", example_data)
    
    print("\nAnalysis Results:")
    print(f"Spike Detection Score: {results['spike_detection_score']:.3f}")
    print(f"Spike Level: {results['spike_level']}")
    print(f"Anomalies Detected: {results['anomaly_detection']['anomaly_count']}")
    print(f"Spikes Detected: {results['spike_detection']['spike_count']}")
    print(f"Changepoints Detected: {results['changepoint_detection']['changepoint_count']}")
    
    print("\nTo use with real data:")
    print("detector.analyze_tweet_engagement(tweet_id, your_engagement_dataframe)")


if __name__ == "__main__":
    main()
