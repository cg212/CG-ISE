"""
This script evaluates three models (Baseline, Intermediate, and HybridModel)
on the available deep learning framework datasets, with optimizations for faster execution.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from scipy import stats
import time
import warnings
from tqdm import tqdm

from preprocessing import TextPreprocessor

# Import models
from baseline_model import BaselineModel
from intermediate_model import IntermediateModel
from hybrid_model import HybridModel

# Configure settings
RESULTS_DIR = 'comprehensive_results'
DATA_DIR = 'datasets'
RANDOM_STATE = 42
TEST_SIZE = 0.3
SAMPLE_RATIO = 0.4  # Sample 40% of data for faster evaluation
N_RUNS = 3  # Number of evaluation runs

# Define the order for models in reports
MODEL_ORDER = ["Baseline", "Intermediate", "HybridModel"]

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'plots'), exist_ok=True)

# Suppress warnings
warnings.filterwarnings('ignore')


def load_dataset(framework):
    """Load and preprocess a dataset for a specific framework."""
    print(f"Loading dataset for {framework}...")
    
    # Construct the file path
    file_path = os.path.join(DATA_DIR, f"{framework}.csv")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Sample data for faster evaluation
    if SAMPLE_RATIO < 1.0:
        # Sample separately for positive and negative classes
        df_pos = df[df['class'] == 1]
        df_neg = df[df['class'] == 0]
        
        # Calculate sample sizes (minimum of 30 positive examples if possible)
        pos_sample_size = max(int(len(df_pos) * SAMPLE_RATIO), min(30, len(df_pos)))
        neg_sample_size = max(int(len(df_neg) * SAMPLE_RATIO), min(100, len(df_neg)))
        
        # Sample from each class
        if len(df_pos) > pos_sample_size:
            df_pos = df_pos.sample(pos_sample_size, random_state=RANDOM_STATE)
        if len(df_neg) > neg_sample_size:
            df_neg = df_neg.sample(neg_sample_size, random_state=RANDOM_STATE)
        
        # Combine the samples
        df = pd.concat([df_pos, df_neg])
    
    # Create a combined text field from Title and Body
    df['text'] = df['Title'].fillna('') + ' ' + df['Body'].fillna('')
    
    # Extract features and labels
    X = df['text'].tolist()
    y = df['class'].astype(int).tolist()
    
    # Print class distribution
    positive_count = sum(y)
    negative_count = len(y) - positive_count
    print(f"Loaded {len(X)} samples with {positive_count} positive ({positive_count/len(y)*100:.1f}%) and {negative_count} negative examples")
    
    return X, y


def preprocess_data(X, framework):
    """Preprocess the text data."""
    print(f"Preprocessing {len(X)} text samples...")
    preprocessor = TextPreprocessor(remove_code=False, keep_tech_terms=True)
    return [preprocessor.preprocess_text(text, framework) for text in X]


def evaluate_model(model_class, model_name, X_train, y_train, X_test, y_test, eval_framework, **kwargs):
    """Train and evaluate a model on the given dataset."""
    print(f"Evaluating {model_name} on {eval_framework} dataset...")
    
    if model_name == "HybridModel":
        kwargs.update({
            'max_features': 5000,  # Reduced from default 10000
            'ngram_range': (1, 2)  # Reduced from (1, 3)
        })
    
    # Initialize the model
    model = model_class(**kwargs)
    
    # Track training time
    start_time = time.time()
    
    # Fit the model
    if model_name == "Intermediate":
        # For the intermediate model, we need tokenized text
        tokenized_X_train = [text.split() for text in X_train]
        model.fit(tokenized_X_train, y_train)
    else:
        # For other models
        model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Track prediction time
    start_time = time.time()
    
    # Generate predictions
    if model_name == "Intermediate":
        # For the intermediate model, we need tokenized text
        tokenized_X_test = [text.split() for text in X_test]
        y_pred = model.predict(tokenized_X_test)
    else:
        # For other models
        y_pred = model.predict(X_test)
    
    prediction_time = time.time() - start_time
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print evaluation report
    print(f"\nClassification Report for {model_name} on {eval_framework}:")
    print(classification_report(y_test, y_pred))
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Prediction time: {prediction_time:.2f} seconds")
    
    # Return metrics
    return {
        'model': model_name,
        'framework': eval_framework,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'training_time': training_time,
        'prediction_time': prediction_time
    }


def run_evaluation(framework, models_to_evaluate=None):
    """Run evaluation for a specific framework across all models."""
    if models_to_evaluate is None:
        models_to_evaluate = MODEL_ORDER.copy()
    
    print(f"Running evaluations for {framework}...")
    
    # Load and preprocess dataset once
    X, y = load_dataset(framework)
    X_processed = preprocess_data(X, framework)
    
    all_results = []
    
    for i in range(N_RUNS):
        print(f"\nRun {i+1}/{N_RUNS}")
        
        # Use a different random seed for each run
        run_seed = RANDOM_STATE + i
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=TEST_SIZE,
            random_state=run_seed, stratify=y
        )
        
        # Evaluate each model
        if "Baseline" in models_to_evaluate:
            baseline_results = evaluate_model(
                BaselineModel, "Baseline", X_train, y_train, X_test, y_test, 
                eval_framework=framework, use_smote=True
            )
            baseline_results['run'] = i
            all_results.append(baseline_results)
        
        if "Intermediate" in models_to_evaluate:
            intermediate_results = evaluate_model(
                IntermediateModel, "Intermediate", X_train, y_train, X_test, y_test, 
                eval_framework=framework, use_smote=True
            )
            intermediate_results['run'] = i
            all_results.append(intermediate_results)
        
        if "HybridModel" in models_to_evaluate:
            hybrid_results = evaluate_model(
                HybridModel, "HybridModel", X_train, y_train, X_test, y_test, 
                eval_framework=framework, framework=framework
            )
            hybrid_results['run'] = i
            all_results.append(hybrid_results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, f"{framework}_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    return results_df


def run_statistical_test(results_df, metric='f1_score'):
    """Run statistical tests to compare model performance."""
    # Group by model and calculate statistics
    model_stats = results_df.groupby('model')[metric].agg(['mean', 'std'])
    
    # Ensure models are in the defined order
    model_stats = model_stats.reindex(index=[m for m in MODEL_ORDER if m in model_stats.index])
    
    print(f"\nModel performance statistics for {metric}:")
    print(model_stats)
    
    # Get the metric values for each model
    models = results_df['model'].unique()
    model_metrics = {model: results_df[results_df['model'] == model][metric].values 
                     for model in models}
    
    # Run statistical tests
    test_results = {}
    
    # Perform Mann-Whitney U test (non-parametric) between baseline and each other model
    if 'Baseline' in models:
        baseline_metric = model_metrics['Baseline']
        
        for model in [m for m in MODEL_ORDER if m in models and m != 'Baseline']:
            model_metric = model_metrics[model]
            stat, p_value = stats.mannwhitneyu(model_metric, baseline_metric, alternative='two-sided')
            
            # Calculate effect size (r = Z / sqrt(N))
            n1, n2 = len(model_metric), len(baseline_metric)
            z = stat
            r = z / np.sqrt(n1 + n2)
            
            test_results[model] = {
                'p_value': p_value,
                'stat': stat,
                'effect_size': r,
                'significant': p_value < 0.05
            }
            
            # Print results
            print(f"\nMann-Whitney U test: Baseline vs {model}")
            print(f"U statistic: {stat}")
            print(f"P-value: {p_value:.5f}")
            print(f"Effect size r: {r:.5f}")
            print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
    
    return test_results


def plot_results(results_df, framework):
    """Create visualizations of the evaluation results."""
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(RESULTS_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Ensure model order is consistent
    # Create a categorical type with our custom ordering
    cat_type = pd.CategoricalDtype(categories=MODEL_ORDER, ordered=True)
    results_df['model'] = results_df['model'].astype(cat_type)
    results_df = results_df.sort_values('model')
    
    # Plot F1 Score
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model', y='f1_score', data=results_df, palette='viridis')
    plt.title(f"F1 Score Comparison for {framework}")
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(plots_dir, f"{framework}_f1_score.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Precision and Recall
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model', y='precision', data=results_df, palette='Blues_d', label='Precision')
    sns.barplot(x='model', y='recall', data=results_df, palette='Reds_d', label='Recall', alpha=0.7)
    plt.title(f"Precision and Recall Comparison for {framework}")
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(plots_dir, f"{framework}_precision_recall.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Training Time (log scale)
    plt.figure(figsize=(12, 6))
    plt.yscale('log')
    sns.barplot(x='model', y='training_time', data=results_df, palette='YlOrBr')
    plt.title(f"Training Time Comparison for {framework} (Log Scale)")
    plt.xlabel('Model')
    plt.ylabel('Training Time (seconds, log scale)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(plots_dir, f"{framework}_training_time.png"), dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_report(all_results):
    """Create a summary report of all evaluation results."""
    # Check if results are empty
    if not all_results:
        print("Cannot create summary report: No results available")
        return ""
    
    # Combine all results
    all_results_df = pd.concat(all_results.values())
    
    # Ensure model order is consistent by creating a categorical type
    cat_type = pd.CategoricalDtype(categories=MODEL_ORDER, ordered=True)
    all_results_df['model'] = all_results_df['model'].astype(cat_type)
    
    # Calculate mean metrics by model and framework
    summary = all_results_df.groupby(['framework', 'model']).agg({
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'training_time': 'mean',
        'prediction_time': 'mean'
    }).reset_index()
    
    # Format the summary report
    report = "# Bug Report Classification Evaluation Summary\n\n"
    
    # Get available frameworks
    frameworks = all_results_df['framework'].unique()
    
    for framework in frameworks:
        framework_results = summary[summary['framework'] == framework]
        
        if len(framework_results) == 0:
            continue
        
        # Sort by model order
        framework_results = framework_results.sort_values('model')
            
        report += f"## {framework.capitalize()} Framework\n\n"
        report += "| Model | Precision | Recall | F1 Score | Training Time (s) | Prediction Time (s) |\n"
        report += "| ----- | --------- | ------ | -------- | ---------------- | ------------------ |\n"
        
        for _, row in framework_results.iterrows():
            model = row['model']
            precision = f"{row[('precision', 'mean')]:.3f} ± {row[('precision', 'std')]:.3f}"
            recall = f"{row[('recall', 'mean')]:.3f} ± {row[('recall', 'std')]:.3f}"
            f1 = f"{row[('f1_score', 'mean')]:.3f} ± {row[('f1_score', 'std')]:.3f}"
            train_time = f"{row[('training_time', 'mean')]:.2f}"
            pred_time = f"{row[('prediction_time', 'mean')]:.2f}"
            
            report += f"| {model} | {precision} | {recall} | {f1} | {train_time} | {pred_time} |\n"
        
        report += "\n"
    
    # Add overall comparison
    report += "## Overall Model Performance\n\n"
    
    # Use categorical dtype to ensure model order
    overall = all_results_df.groupby('model').agg({
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'training_time': 'mean',
        'prediction_time': 'mean'
    }).reset_index()
    
    # Sort by the predetermined model order
    overall = overall.sort_values('model')
    
    report += "| Model | Precision | Recall | F1 Score | Training Time (s) | Prediction Time (s) |\n"
    report += "| ----- | --------- | ------ | -------- | ---------------- | ------------------ |\n"
    
    for _, row in overall.iterrows():
        model = row['model']
        precision = f"{row[('precision', 'mean')]:.3f} ± {row[('precision', 'std')]:.3f}"
        recall = f"{row[('recall', 'mean')]:.3f} ± {row[('recall', 'std')]:.3f}"
        f1 = f"{row[('f1_score', 'mean')]:.3f} ± {row[('f1_score', 'std')]:.3f}"
        train_time = f"{row[('training_time', 'mean')]:.2f}"
        pred_time = f"{row[('prediction_time', 'mean')]:.2f}"
        
        report += f"| {model} | {precision} | {recall} | {f1} | {train_time} | {pred_time} |\n"
    
    # Save the report
    report_path = os.path.join(RESULTS_DIR, "evaluation_summary.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Summary report saved to {report_path}")
    
    return report


def main():
    """Main function to run comprehensive evaluation."""
    # List of frameworks to evaluate
    frameworks = ['tensorflow', 'pytorch', 'keras', 'caffe']
    
    # Dictionary to store results for each framework
    all_framework_results = {}
    
    # Check which frameworks have data available
    available_frameworks = []
    for fw in frameworks:
        file_path = os.path.join(DATA_DIR, f"{fw}.csv")
        if os.path.exists(file_path):
            available_frameworks.append(fw)
        else:
            print(f"Warning: Dataset for {fw} not found at {file_path}")
    
    # For each available framework
    for framework in available_frameworks:
        try:
            print(f"\n{'='*60}\nEvaluating {framework.upper()} dataset\n{'='*60}")
            
            # Run evaluation and get results
            results_df = run_evaluation(framework)
            
            # Run statistical tests
            run_statistical_test(results_df, 'f1_score')
            
            # Create visualizations
            plot_results(results_df, framework)
            
            # Store results
            all_framework_results[framework] = results_df
            
        except Exception as e:
            print(f"Error evaluating {framework}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate summary report
    create_summary_report(all_framework_results)
    
    # Create combined performance plot across all frameworks
    if all_framework_results:
        all_results_df = pd.concat(all_framework_results.values())
        
        # Ensure model order is consistent
        cat_type = pd.CategoricalDtype(categories=MODEL_ORDER, ordered=True)
        all_results_df['model'] = all_results_df['model'].astype(cat_type)
        all_results_df = all_results_df.sort_values('model')
        
        # Plot overall F1 scores by model and framework
        plt.figure(figsize=(14, 8))
        sns.barplot(x='framework', y='f1_score', hue='model', data=all_results_df, palette='viridis', hue_order=MODEL_ORDER)
        plt.title("F1 Score Comparison Across Frameworks")
        plt.xlabel('Framework')
        plt.ylabel('F1 Score')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Model', loc='lower right')
        plt.savefig(os.path.join(RESULTS_DIR, 'plots', "all_frameworks_f1_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Evaluation complete! Results saved to {RESULTS_DIR} directory.")


if __name__ == "__main__":
    main()