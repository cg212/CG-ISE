# Note - this is the baseline model from lab1, but adapted so that it fits with my test scripts and visualisations.
# it still uses the same functions and methods as the original baseline model, but has been adapted to fit with the test scripts and visualisations.
# still uses naive bayes with tf-idf, but has been adapted to fit with the test scripts and visualisations.

"""
Baseline model for bug report classification using Naive Bayes with TF-IDF.

This module implements a simple baseline classifier that uses TF-IDF features
with a Multinomial Naive Bayes classifier to detect performance-related bug reports.
"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as IMBPipeline

class BaselineModel:

    
    def __init__(self, max_features=5000, ngram_range=(1, 2), alpha=1.0, 
                 use_smote=True, class_weight=None):

        self.max_features = max_features
        self.ngram_range = ngram_range
        self.alpha = alpha
        self.use_smote = use_smote
        self.class_weight = class_weight
        self.model = None
        self.tfidf_vectorizer = None
    
    def build(self):

        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words='english'
        )
        
        # Create Naive Bayes classifier
        nb_classifier = MultinomialNB(alpha=self.alpha, class_prior=None)
        
        # Create pipeline based on whether SMOTE is used
        if self.use_smote:
            self.model = IMBPipeline([
                ('tfidf', self.tfidf_vectorizer),
                ('smote', SMOTE(random_state=42)),
                ('classifier', nb_classifier)
            ])
        else:
            self.model = Pipeline([
                ('tfidf', self.tfidf_vectorizer),
                ('classifier', nb_classifier)
            ])
        
        return self
    
    def optimize_hyperparameters(self, X_train, y_train, cv=5):
        """
        Optimize hyperparameters using grid search.
        
        Args:
            X_train (list): Training data (list of bug report texts)
            y_train (list): Target values (0 for non-performance bugs, 1 for performance bugs)
            cv (int): Number of cross-validation folds
            
        Returns:
            self: The BaselineModel instance with optimized parameters
        """
        # Define pipeline based on whether SMOTE is used
        if self.use_smote:
            pipeline = IMBPipeline([
                ('tfidf', TfidfVectorizer(stop_words='english')),
                ('smote', SMOTE(random_state=42)),
                ('classifier', MultinomialNB())
            ])
            
            param_grid = {
                'tfidf__max_features': [3000, 5000, 10000],
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'classifier__alpha': [0.1, 0.5, 1.0, 2.0]
            }
        else:
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english')),
                ('classifier', MultinomialNB())
            ])
            
            param_grid = {
                'tfidf__max_features': [3000, 5000, 10000],
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'classifier__alpha': [0.1, 0.5, 1.0, 2.0]
            }
        
        # Perform grid search with F1 score as the optimization metric
        print("Starting grid search for hyperparameter optimization...")
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=cv, 
            scoring='f1', 
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        # Set the best parameters
        best_params = grid_search.best_params_
        print(f"Best parameters found: {best_params}")
        
        self.max_features = best_params['tfidf__max_features']
        self.ngram_range = best_params['tfidf__ngram_range']
        self.alpha = best_params['classifier__alpha']
        
        # Build the model with the best parameters
        self.build()
        
        return self
    
    def fit(self, X_train, y_train):
        """
        Fit the model to the training data.
        
        Args:
            X_train (list): Training data (list of bug report texts)
            y_train (list): Target values (0 for non-performance bugs, 1 for performance bugs)
            
        Returns:
            self: The fitted BaselineModel instance
        """
        if self.model is None:
            self.build()
        
        print(f"Fitting baseline model with {len(X_train)} samples...")
        print(f"Class distribution: {np.bincount(y_train)}")
        
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.
        
        Args:
            X_test (list): Test data (list of bug report texts)
            
        Returns:
            numpy.ndarray: Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
            
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Args:
            X_test (list): Test data (list of bug report texts)
            
        Returns:
            numpy.ndarray: Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
            
        return self.model.predict_proba(X_test)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (list): Test data (list of bug report texts)
            y_test (list): True class labels
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def get_feature_names(self):
        """
        Get the feature names from the TF-IDF vectorizer.
        
        Returns:
            list: Feature names (words) used by the TF-IDF vectorizer
        """
        if self.tfidf_vectorizer is None or not hasattr(self.tfidf_vectorizer, 'get_feature_names_out'):
            raise ValueError("TF-IDF vectorizer has not been fitted yet.")
            
        return self.tfidf_vectorizer.get_feature_names_out()
    
    def get_top_features(self, top_n=20):
        """
        Get the top features (words) that are most indicative of performance-related bugs.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            list: List of (feature_name, coefficient) tuples for the top features
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Get feature names
        feature_names = self.get_feature_names()
        

        if self.use_smote:
            # If using SMOTE, the classifier is the third step in the pipeline
            coefficients = self.model.named_steps['classifier'].feature_log_prob_
        else:
            # Otherwise, it's the second step
            coefficients = self.model.named_steps['classifier'].feature_log_prob_
        

        log_prob_diffs = coefficients[1] - coefficients[0]
        

        top_indices = np.argsort(log_prob_diffs)[-top_n:]
        

        return [(feature_names[i], log_prob_diffs[i]) for i in top_indices[::-1]]