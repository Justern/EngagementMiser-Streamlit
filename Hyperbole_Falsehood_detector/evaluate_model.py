#!/usr/bin/env python3
"""
Model Evaluation Script for Hyperbole & Falsehood Detector
========================================================

This script evaluates the trained model and provides comprehensive metrics including:
- Precision, Recall, F1-Score, Accuracy
- Confusion Matrix
- ROC Curve and AUC
- Detailed classification report
- Threshold analysis for optimal performance

Usage: python evaluate_model.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sqlalchemy import create_engine, text as sql_text
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelEvaluator:
    def __init__(self):
        """Initialize the model evaluator."""
        self.model = None
        self.tokenizer = None
        self.device = None
        self.engine = None
        self.setup_database()
        
    def setup_database(self):
        """Setup database connection."""
        SQL_SERVER = "localhost"
        SQL_DB = "EngagementMiser"
        SQL_DRIVER = "ODBC Driver 18 for SQL Server"
        
        CONN_STR = (
            f"mssql+pyodbc://@{SQL_SERVER}/{SQL_DB}"
            f"?driver={SQL_DRIVER.replace(' ', '+')}"
            "&Trusted_Connection=yes"
            "&TrustServerCertificate=yes"
        )
        
        try:
            self.engine = create_engine(CONN_STR)
            print("✅ Database connection established")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            sys.exit(1)
    
    def load_model(self):
        """Load the trained model and tokenizer."""
        checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "text_softlabel_roberta")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        
        print("🔄 Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Model loaded on {self.device}")
    
    def load_evaluation_data(self, test_size=0.2, random_state=42):
        """Load data for evaluation from the annotated dataset."""
        print("🔄 Loading evaluation data...")
        
        query = """
        SELECT tweet_id, text, label_confidence
        FROM [EngagementMiser].[dbo].[Hyperbole_Falsehood_tweets_annot]
        WHERE text IS NOT NULL 
        AND LEN(text) > 10 
        AND label_confidence IS NOT NULL
        """
        
        try:
            df = pd.read_sql_query(sql_text(query), self.engine)
            print(f"✅ Loaded {len(df)} annotated tweets")
            
            # Convert to binary labels (threshold at 0.5)
            df['binary_label'] = (df['label_confidence'] >= 0.5).astype(int)
            
            # Split into train/test
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state, 
                stratify=df['binary_label']
            )
            
            print(f"📊 Train set: {len(train_df)} tweets")
            print(f"📊 Test set: {len(test_df)} tweets")
            print(f"📊 Class distribution in test set:")
            print(test_df['binary_label'].value_counts().sort_index())
            
            return test_df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def predict_batch(self, texts, max_len=128):
        """Make predictions on a batch of texts."""
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                enc = self.tokenizer(
                    str(text),
                    padding="max_length",
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                
                # Predict
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                score = torch.sigmoid(logits.squeeze()).item()
                predictions.append(score)
        
        return np.array(predictions)
    
    def evaluate_model(self, test_df):
        """Evaluate the model and calculate all metrics."""
        print("🔄 Running model evaluation...")
        
        # Get predictions
        test_texts = test_df['text'].tolist()
        y_pred_proba = self.predict_batch(test_texts)
        y_true = test_df['binary_label'].values
        
        # Convert probabilities to binary predictions (threshold 0.5)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate ROC and AUC
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Calculate Precision-Recall curve
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        # Store results
        self.results = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve,
            'avg_precision': avg_precision
        }
        
        return self.results
    
    def print_metrics(self):
        """Print all evaluation metrics."""
        if not hasattr(self, 'results'):
            print("❌ No evaluation results available. Run evaluate_model() first.")
            return
        
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION RESULTS")
        print("="*60)
        
        # Basic metrics
        print(f"\n🎯 Basic Metrics:")
        print(f"   Accuracy:  {self.results['accuracy']:.4f}")
        print(f"   Precision: {self.results['precision']:.4f}")
        print(f"   Recall:    {self.results['recall']:.4f}")
        print(f"   F1-Score:  {self.results['f1']:.4f}")
        
        # ROC and PR AUC
        print(f"\n📈 Advanced Metrics:")
        print(f"   ROC AUC:           {self.results['roc_auc']:.4f}")
        print(f"   Average Precision: {self.results['avg_precision']:.4f}")
        
        # Confusion matrix
        print(f"\n🔍 Confusion Matrix:")
        cm = self.results['confusion_matrix']
        print(f"   Predicted")
        print(f"Actual    0    1")
        print(f"    0   {cm[0,0]:4d} {cm[0,1]:4d}")
        print(f"    1   {cm[1,0]:4d} {cm[1,1]:4d}")
        
        # Detailed classification report
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(
            self.results['y_true'], 
            self.results['y_pred'],
            target_names=['Truthful (0)', 'Hyperbole/False (1)']
        ))
    
    def plot_results(self, save_plots=True):
        """Create comprehensive visualization plots."""
        if not hasattr(self, 'results'):
            print("❌ No evaluation results available. Run evaluate_model() first.")
            return
        
        print("🔄 Creating visualization plots...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Hyperbole & Falsehood Detector - Model Evaluation', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix Heatmap
        ax1 = axes[0, 0]
        cm = self.results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Truthful', 'Hyperbole/False'],
                   yticklabels=['Truthful', 'Hyperbole/False'],
                   ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # 2. ROC Curve
        ax2 = axes[0, 1]
        ax2.plot(self.results['fpr'], self.results['tpr'], 
                color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {self.results["roc_auc"]:.3f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('Receiver Operating Characteristic (ROC)')
        ax2.legend(loc="lower right")
        ax2.grid(True)
        
        # 3. Precision-Recall Curve
        ax3 = axes[1, 0]
        ax3.plot(self.results['recall_curve'], self.results['precision_curve'], 
                color='blue', lw=2, 
                label=f'PR curve (AP = {self.results["avg_precision"]:.3f})')
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve')
        ax3.legend(loc="lower left")
        ax3.grid(True)
        
        # 4. Prediction Distribution
        ax4 = axes[1, 1]
        y_true = self.results['y_true']
        y_pred_proba = self.results['y_pred_proba']
        
        # Plot histograms for each class
        ax4.hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7, 
                label='Truthful (0)', color='green', density=True)
        ax4.hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7, 
                label='Hyperbole/False (1)', color='red', density=True)
        ax4.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
        ax4.set_xlabel('Predicted Probability')
        ax4.set_ylabel('Density')
        ax4.set_title('Prediction Probability Distribution')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = "model_evaluation_plots.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plots saved to: {plot_path}")
        
        plt.show()
    
    def threshold_analysis(self):
        """Analyze model performance across different thresholds."""
        if not hasattr(self, 'results'):
            print("❌ No evaluation results available. Run evaluate_model() first.")
            return
        
        print("🔄 Performing threshold analysis...")
        
        thresholds = np.arange(0.1, 1.0, 0.05)
        metrics = []
        
        for threshold in thresholds:
            y_pred_thresh = (self.results['y_pred_proba'] >= threshold).astype(int)
            
            precision = precision_score(self.results['y_true'], y_pred_thresh, zero_division=0)
            recall = recall_score(self.results['y_true'], y_pred_thresh, zero_division=0)
            f1 = f1_score(self.results['y_true'], y_pred_thresh, zero_division=0)
            accuracy = accuracy_score(self.results['y_true'], y_pred_thresh)
            
            metrics.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            })
        
        metrics_df = pd.DataFrame(metrics)
        
        print(f"\n📊 Threshold Analysis:")
        print(f"Optimal thresholds based on different metrics:")
        print(f"   Best F1-Score:     {metrics_df.loc[metrics_df['f1'].idxmax(), 'threshold']:.2f}")
        print(f"   Best Precision:    {metrics_df.loc[metrics_df['precision'].idxmax(), 'threshold']:.2f}")
        print(f"   Best Recall:       {metrics_df.loc[metrics_df['recall'].idxmax(), 'threshold']:.2f}")
        print(f"   Best Accuracy:     {metrics_df.loc[metrics_df['accuracy'].idxmax(), 'threshold']:.2f}")
        
        # Plot threshold analysis
        plt.figure(figsize=(12, 8))
        plt.plot(metrics_df['threshold'], metrics_df['precision'], 'o-', label='Precision', linewidth=2)
        plt.plot(metrics_df['threshold'], metrics_df['recall'], 's-', label='Recall', linewidth=2)
        plt.plot(metrics_df['threshold'], metrics_df['f1'], '^-', label='F1-Score', linewidth=2)
        plt.plot(metrics_df['threshold'], metrics_df['accuracy'], 'd-', label='Accuracy', linewidth=2)
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Model Performance vs. Classification Threshold')
        plt.legend()
        plt.grid(True)
        plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Default Threshold (0.5)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("threshold_analysis.png", dpi=300, bbox_inches='tight')
        print(f"✅ Threshold analysis plot saved to: threshold_analysis.png")
        plt.show()
        
        return metrics_df

def main():
    """Main evaluation function."""
    print("🔍 Hyperbole & Falsehood Detector - Model Evaluation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    try:
        # Load model
        evaluator.load_model()
        
        # Load evaluation data
        test_df = evaluator.load_evaluation_data()
        if test_df is None:
            print("❌ Failed to load evaluation data")
            return
        
        # Run evaluation
        results = evaluator.evaluate_model(test_df)
        
        # Print metrics
        evaluator.print_metrics()
        
        # Create plots
        evaluator.plot_results()
        
        # Threshold analysis
        evaluator.threshold_analysis()
        
        print("\n✅ Evaluation completed successfully!")
        print("📁 Check the generated plot files for visualizations")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


