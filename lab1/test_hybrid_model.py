"""
Test script for evaluating the HybridModel model on a single framework.

This script tests the HybridModel model against the baseline model (Naive Bayes + TF-IDF)
on the specified framework dataset to verify performance improvements.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import time
import os
import argparse

# Import your baseline model
from baseline_model import BaselineModel

# Import our new model
from hybrid_model import HybridModel

# Import preprocessing utilities
from preprocessing import TextPreprocessor


def load_data(dataset_path):
    """
    Load a dataset from the specified path.
    
    Args:
        dataset_path: Path to the dataset CSV file
        
    Returns:
        X: List of bug report texts
        y: List of labels (0 for non-performance bugs, 1 for performance bugs)
    """
    print(f"Loading data from {dataset_path}...")
    
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Create a combined text field
    df['text'] = df['Title'].fillna('') + ' ' + df['Body'].fillna('')
    
    # Extract features and labels - check for 'class' or 'related' column
    label_col = 'class' if 'class' in df.columns else 'related'
    X = df['text'].tolist()
    y = df[label_col].astype(int).tolist()
    
    # Print class distribution
    positive_count = sum(y)
    negative_count = len(y) - positive_count
    print(f"Loaded {len(X)} samples with {positive_count} positive ({positive_count/len(y)*100:.1f}%) and {negative_count} negative examples")
    
    return X, y


def test_models(framework='tensorflow', random_state=42, test_size=0.3):
    """
    Test both models on the specified framework dataset.
    
    Args:
        framework: Name of the framework to test on
        random_state: Random seed for reproducibility
        test_size: Proportion of data to use for testing
        
    Returns:
        dict: Dictionary of evaluation results
    """
    # Construct the dataset path
    dataset_path = os.path.join('datasets', f"{framework}.csv")
    
    # Load data
    X, y = load_data(dataset_path)
    
    # Initialize the preprocessing
    preprocessor = TextPreprocessor(remove_code=False, keep_tech_terms=True)
    
    # Preprocess text
    print("Preprocessing text...")
    X_processed = [preprocessor.preprocess_text(text, framework) for text in X]
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, 
        random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    results = {}
    
    # Test the baseline model
    print("\n===== Testing Baseline Model =====")
    
    # Check if we have enough samples for SMOTE
    minority_count = min(sum(y_train), len(y_train) - sum(y_train))
    use_smote = minority_count >= 6
    
    if not use_smote:
        print(f"Warning: Not enough minority samples ({minority_count}) for SMOTE. Using regular training.")
    
    baseline_model = BaselineModel(use_smote=use_smote)
    
    # Train and time the model
    start_time = time.time()
    baseline_model.fit(X_train, y_train)
    baseline_training_time = time.time() - start_time
    
    # Make predictions and time them
    start_time = time.time()
    baseline_pred = baseline_model.predict(X_test)
    baseline_prediction_time = time.time() - start_time
    
    # Calculate metrics
    baseline_precision = precision_score(y_test, baseline_pred)
    baseline_recall = recall_score(y_test, baseline_pred)
    baseline_f1 = f1_score(y_test, baseline_pred)
    
    # Print results
    print("\nBaseline Model Results:")
    print(classification_report(y_test, baseline_pred))
    print(f"Training time: {baseline_training_time:.2f} seconds")
    print(f"Prediction time: {baseline_prediction_time:.2f} seconds")
    
    # Store results
    results['baseline'] = {
        'precision': baseline_precision,
        'recall': baseline_recall,
        'f1_score': baseline_f1,
        'training_time': baseline_training_time,
        'prediction_time': baseline_prediction_time
    }
    
    # Test the HybridModel model
    print(f"\n===== Testing HybridModel Model on {framework} dataset =====")
    hybrid_model = HybridModel(framework=framework, use_smote=use_smote)
    
    # Train and time the model
    start_time = time.time()
    hybrid_model.fit(X_train, y_train)
    hybrid_training_time = time.time() - start_time
    
    # Make predictions and time them
    start_time = time.time()
    hybrid_pred = hybrid_model.predict(X_test)
    hybrid_prediction_time = time.time() - start_time
    
    # Calculate metrics
    hybrid_precision = precision_score(y_test, hybrid_pred)
    hybrid_recall = recall_score(y_test, hybrid_pred)
    hybrid_f1 = f1_score(y_test, hybrid_pred)
    
    # Print results
    print("\nHybridModel Model Results:")
    print(classification_report(y_test, hybrid_pred))
    print(f"Training time: {hybrid_training_time:.2f} seconds")
    print(f"Prediction time: {hybrid_prediction_time:.2f} seconds")
    
    # Store results
    results['hybrid'] = {
        'precision': hybrid_precision,
        'recall': hybrid_recall,
        'f1_score': hybrid_f1,
        'training_time': hybrid_training_time,
        'prediction_time': hybrid_prediction_time
    }
    
    # Compare results
    print("\n===== Model Comparison =====")
    print(f"Framework: {framework}")
    print(f"Precision: Baseline: {baseline_precision:.3f}, HybridModel: {hybrid_precision:.3f}")
    print(f"Recall: Baseline: {baseline_recall:.3f}, HybridModel: {hybrid_recall:.3f}")
    print(f"F1 Score: Baseline: {baseline_f1:.3f}, HybridModel: {hybrid_f1:.3f}")
    print(f"Training Time: Baseline: {baseline_training_time:.2f}s, HybridModel: {hybrid_training_time:.2f}s")
    print(f"Prediction Time: Baseline: {baseline_prediction_time:.2f}s, HybridModel: {hybrid_prediction_time:.2f}s")
    
    # Calculate improvement percentages
    precision_improvement = (hybrid_precision - baseline_precision) / baseline_precision * 100
    recall_improvement = (hybrid_recall - baseline_recall) / baseline_recall * 100
    f1_improvement = (hybrid_f1 - baseline_f1) / baseline_f1 * 100
    
    print(f"\nImprovement over baseline:")
    print(f"Precision: {precision_improvement:.1f}%")
    print(f"Recall: {recall_improvement:.1f}%")
    print(f"F1 Score: {f1_improvement:.1f}%")
    
    return results


def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description='Test HybridModel against baseline for bug report classification')
    parser.add_argument('--framework', type=str, default='tensorflow',
                        choices=['tensorflow', 'pytorch', 'keras', 'mxnet', 'caffe'],
                        help='DL framework to use for testing')
    parser.add_argument('--test_size', type=float, default=0.3,
                        help='Proportion of data to use for testing')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Run tests with the specified parameters
    results = test_models(
        framework=args.framework,
        random_state=args.random_state,
        test_size=args.test_size
    )
    
    return results


if __name__ == "__main__":
    main()