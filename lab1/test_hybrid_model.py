"""
This is the testing script for my hybrid model against the baseline, with the framework of your choosing.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import time
import os
import argparse
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import models
from baseline_model import BaselineModel
from hybrid_model import HybridModel
from preprocessing import TextPreprocessor


def load_data(dataset_path, sample_size=None):

    print(f"Loading data from {dataset_path}...")
    

    df = pd.read_csv(dataset_path)
    

    label_col = 'class' if 'class' in df.columns else 'related'
    

    if sample_size and sample_size < len(df):

        df_pos = df[df[label_col] == 1]
        df_neg = df[df[label_col] == 0]
        

        pos_size = min(len(df_pos), max(20, int(sample_size * 0.3)))
        neg_size = min(len(df_neg), sample_size - pos_size)
        

        if len(df_pos) > pos_size:
            df_pos = df_pos.sample(pos_size, random_state=42)
        if len(df_neg) > neg_size:
            df_neg = df_neg.sample(neg_size, random_state=42)
        

        df = pd.concat([df_pos, df_neg])
        print(f"Sampled {len(df)} records for faster testing")
    
   
    df['text'] = df['Title'].fillna('') + ' ' + df['Body'].fillna('')
    

    X = df['text'].tolist()
    y = df[label_col].astype(int).tolist()
    

    pos_count = sum(y)
    neg_count = len(y) - pos_count
    print(f"Loaded {len(X)} samples with {pos_count} positive ({pos_count/len(y)*100:.1f}%) and {neg_count} negative examples")
    
    return X, y


def test_models(framework='tensorflow', random_state=42, test_size=0.3, sample_size=None, fast_mode=True):
    """Test baseline and hybrid models with optimized parameters."""
    start_total = time.time()
    

    dataset_path = os.path.join('datasets', f"{framework}.csv")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset for {framework} not found at {dataset_path}")
        return None
    
  
    X, y = load_data(dataset_path, sample_size)
    
 
    print("Preprocessing text...")
    try:
        preprocessor = TextPreprocessor(remove_code=False, keep_tech_terms=True)
        X_processed = [preprocessor.preprocess_text(text, framework) for text in X]
    except Exception as e:
        print(f"Warning: Error in framework-specific preprocessing: {e}")
        print("Falling back to basic preprocessing...")

        X_processed = [text.lower() for text in X]
    

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    results = {}
    

    minority_count = min(sum(y_train), len(y_train) - sum(y_train))
    use_smote = minority_count >= 6
    
    # Test baseline model
    print("\n===== Testing Baseline Model =====")
    baseline_model = BaselineModel(use_smote=use_smote)
    
    # Train and time
    start_time = time.time()
    baseline_model.fit(X_train, y_train)
    baseline_train_time = time.time() - start_time
    
    # Predict and time
    start_time = time.time()
    baseline_pred = baseline_model.predict(X_test)
    baseline_pred_time = time.time() - start_time
    
    # Calculate metrics
    baseline_metrics = {
        'precision': precision_score(y_test, baseline_pred),
        'recall': recall_score(y_test, baseline_pred),
        'f1_score': f1_score(y_test, baseline_pred),
        'training_time': baseline_train_time,
        'prediction_time': baseline_pred_time
    }
    
    # Print results
    print("\nBaseline Model Results:")
    print(classification_report(y_test, baseline_pred))
    print(f"Training time: {baseline_train_time:.2f}s, Prediction time: {baseline_pred_time:.2f}s")
    
    # Test HybridModel
    print(f"\n===== Testing HybridModel on {framework} dataset =====")
    
    # Configure HybridModel with optimized parameters for speed
    model_params = {
        'framework': framework,
        'use_smote': use_smote,
        'max_features': 3000 if fast_mode else 5000,
        'ngram_range': (1, 2),
        'use_meta_features': fast_mode,  # Simpler features in fast mode
        'use_pattern_features': True,    # These are important for performance detection
        'use_code_features': not fast_mode  # Code features can be slow to process
    }
    
    hybrid_model = HybridModel(**model_params)
    
    # Train and time
    start_time = time.time()
    hybrid_model.fit(X_train, y_train)
    hybrid_train_time = time.time() - start_time
    
    # Predict and time
    start_time = time.time()
    hybrid_pred = hybrid_model.predict(X_test)
    hybrid_pred_time = time.time() - start_time
    
    # Calculate metrics
    hybrid_metrics = {
        'precision': precision_score(y_test, hybrid_pred),
        'recall': recall_score(y_test, hybrid_pred),
        'f1_score': f1_score(y_test, hybrid_pred),
        'training_time': hybrid_train_time,
        'prediction_time': hybrid_pred_time
    }
    
    # Print results
    print("\nHybridModel Results:")
    print(classification_report(y_test, hybrid_pred))
    print(f"Training time: {hybrid_train_time:.2f}s, Prediction time: {hybrid_pred_time:.2f}s")
    
    # Compare models
    print("\n===== Model Comparison =====")
    print(f"Framework: {framework}")
    print(f"Precision: Baseline: {baseline_metrics['precision']:.3f}, HybridModel: {hybrid_metrics['precision']:.3f}")
    print(f"Recall: Baseline: {baseline_metrics['recall']:.3f}, HybridModel: {hybrid_metrics['recall']:.3f}")
    print(f"F1 Score: Baseline: {baseline_metrics['f1_score']:.3f}, HybridModel: {hybrid_metrics['f1_score']:.3f}")
    
    # Calculate improvement percentages
    if baseline_metrics['precision'] > 0:
        prec_imp = (hybrid_metrics['precision'] - baseline_metrics['precision']) / baseline_metrics['precision'] * 100
        print(f"Precision improvement: {prec_imp:.1f}%")
    
    if baseline_metrics['recall'] > 0:
        recall_imp = (hybrid_metrics['recall'] - baseline_metrics['recall']) / baseline_metrics['recall'] * 100
        print(f"Recall improvement: {recall_imp:.1f}%")
    
    if baseline_metrics['f1_score'] > 0:
        f1_imp = (hybrid_metrics['f1_score'] - baseline_metrics['f1_score']) / baseline_metrics['f1_score'] * 100
        print(f"F1 Score improvement: {f1_imp:.1f}%")
    
    print(f"\nTotal test time: {time.time() - start_total:.2f} seconds")
    
    # Return all results
    return {
        'baseline': baseline_metrics,
        'hybrid': hybrid_metrics,
        'framework': framework
    }


def main():
    """Parse command line arguments and run tests."""
    parser = argparse.ArgumentParser(description='Test HybridModel against baseline')
    parser.add_argument('--framework', type=str, default='tensorflow',
                        choices=['tensorflow', 'pytorch', 'keras', 'mxnet', 'caffe'],
                        help='DL framework to test')
    parser.add_argument('--test_size', type=float, default=0.3,
                        help='Proportion of data for testing')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of samples to use (default: all)')
    parser.add_argument('--fast', action='store_true',
                        help='Enable fast mode with simplified features')
    
    args = parser.parse_args()
    
    # Run tests with parameters
    results = test_models(
        framework=args.framework,
        random_state=args.random_state,
        test_size=args.test_size,
        sample_size=args.sample_size,
        fast_mode=args.fast
    )
    
    return results


if __name__ == "__main__":
    main()