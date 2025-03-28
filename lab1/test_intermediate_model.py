"""This is a test script for comparing my intermediate model against the baseline model, with the framework of your choosing."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
import time
import argparse
import os

from baseline_model import BaselineModel
from intermediate_model import IntermediateModel


def preprocess_text(text):
    """Simple text preprocessing."""
    if not isinstance(text, str):
        return ""
    return text.lower()

def tokenize_text(text):
    """Simple tokenization by splitting on spaces."""
    return text.split()

def load_data(dataset_name, sample_size=None):
    """Load dataset with optional sampling."""
    file_path = f'datasets/{dataset_name}.csv'
    print(f"Loading data from {file_path}...")
    
    df = pd.read_csv(file_path)
    

    label_col = 'class' if 'class' in df.columns else 'related'
    

    df['text'] = df['Title'].fillna('') + ' ' + df['Body'].fillna('')
    

    unique_labels = df[label_col].unique()
    if not all(isinstance(label, (int, np.integer)) for label in unique_labels):
        label_mapping = {val: i for i, val in enumerate(sorted(unique_labels))}
        df[label_col] = df[label_col].map(label_mapping)
    

    if sample_size and sample_size < len(df):
        pos_df = df[df[label_col] == 1]
        neg_df = df[df[label_col] == 0]
        
        pos_sample = min(len(pos_df), sample_size // 2)
        neg_sample = min(len(neg_df), sample_size // 2)
        
        df = pd.concat([
            pos_df.sample(pos_sample, random_state=42),
            neg_df.sample(neg_sample, random_state=42)
        ])
    

    X = df['text'].tolist()
    y = df[label_col].astype(int).tolist()
    
    print(f"Loaded {len(X)} samples with {sum(y)} positive and {len(y) - sum(y)} negative examples")
    return X, y

def main():
    parser = argparse.ArgumentParser(description='Test models for bug report classification')
    parser.add_argument('--framework', type=str, default='tensorflow',
                        choices=['tensorflow', 'pytorch', 'keras', 'mxnet', 'caffe'],
                        help='DL framework to use for testing')
    parser.add_argument('--sample_size', type=int, default=300,
                        help='Number of samples to use for testing')
    
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    

    X, y = load_data(args.framework, sample_size=args.sample_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    

    print(f"\nFramework: {args.framework}, Training: {len(X_train)}, Test: {len(X_test)}")
    

    X_train_processed = [preprocess_text(text) for text in X_train]
    X_test_processed = [preprocess_text(text) for text in X_test]
    

    X_train_tokens = [tokenize_text(text) for text in X_train_processed]
    X_test_tokens = [tokenize_text(text) for text in X_test_processed]
    
    # Test baseline model
    print("\n===== Testing Baseline Model =====")
    minority_count = min(sum(y_train), len(y_train) - sum(y_train))
    use_smote = minority_count >= 6
    
    start_time = time.time()
    baseline = BaselineModel(use_smote=use_smote)
    baseline.fit(X_train_processed, y_train)
    baseline_train_time = time.time() - start_time
    
    start_time = time.time()
    baseline_pred = baseline.predict(X_test_processed)
    baseline_pred_time = time.time() - start_time
    
    print(classification_report(y_test, baseline_pred))
    print(f"Training time: {baseline_train_time:.2f}s, Prediction time: {baseline_pred_time:.2f}s")
    
    # Test intermediate model
    print("\n===== Testing Intermediate Model =====")
    start_time = time.time()
    intermediate = IntermediateModel(use_smote=use_smote)
    intermediate.fit(X_train_tokens, y_train, train_word2vec=True)
    intermediate_train_time = time.time() - start_time
    
    start_time = time.time()
    intermediate_pred = intermediate.predict(X_test_tokens)
    intermediate_pred_time = time.time() - start_time
    
    print(classification_report(y_test, intermediate_pred))
    print(f"Training time: {intermediate_train_time:.2f}s, Prediction time: {intermediate_pred_time:.2f}s")
    
    # Compare results
    print("\n===== Model Comparison =====")
    from sklearn.metrics import f1_score
    baseline_f1 = f1_score(y_test, baseline_pred, average='macro')
    intermediate_f1 = f1_score(y_test, intermediate_pred, average='macro')
    
    print(f"Baseline F1: {baseline_f1:.4f}, Intermediate F1: {intermediate_f1:.4f}")
    print(f"Training time: Baseline: {baseline_train_time:.2f}s, Intermediate: {intermediate_train_time:.2f}s")
    print(f"Prediction time: Baseline: {baseline_pred_time:.2f}s, Intermediate: {intermediate_pred_time:.2f}s")

if __name__ == "__main__":
    main()