"""my intermediate model for bug report classification using SVM with Word2Vec."""

import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as IMBPipeline
from gensim.models import Word2Vec
import joblib
import os

class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):

    
    def __init__(self, word2vec_model):
        self.word2vec_model = word2vec_model
        self.vector_size = word2vec_model.wv.vector_size
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array([
            np.mean([self.word2vec_model.wv[word] for word in words 
                    if word in self.word2vec_model.wv] or 
                   [np.zeros(self.vector_size)], axis=0)
            for words in X
        ])


class IntermediateModel:
    """SVM with Word2Vec features for bug report classification."""
    
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4,
                 use_smote=True, C=1.0, kernel='rbf', gamma='scale', class_weight=None):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.use_smote = use_smote
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.class_weight = class_weight
        
        self.word2vec_model = None
        self.vectorizer = None
        self.model = None
    
    def train_word2vec(self, tokenized_texts, save_path=None):

        self.word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=1  
        )
        
        self.vectorizer = MeanEmbeddingVectorizer(self.word2vec_model)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.word2vec_model.save(save_path)
        
        return self
    
    def load_word2vec(self, model_path):

        self.word2vec_model = Word2Vec.load(model_path)
        self.vector_size = self.word2vec_model.wv.vector_size
        self.vectorizer = MeanEmbeddingVectorizer(self.word2vec_model)
        return self
    
    def build(self):

        svm_classifier = SVC(
            C=self.C, kernel=self.kernel, gamma=self.gamma,
            probability=True, class_weight=self.class_weight, random_state=42
        )
        
        self.model = (
            IMBPipeline([('smote', SMOTE(random_state=42)), ('classifier', svm_classifier)])
            if self.use_smote else svm_classifier
        )
        
        return self
    
    def optimize_hyperparameters(self, X_transformed, y_train, cv=5):

        from sklearn.model_selection import GridSearchCV
        
        if self.use_smote:
            pipeline = IMBPipeline([('smote', SMOTE(random_state=42)), ('classifier', SVC(probability=True))])
            param_grid = {
                'classifier__C': [0.1, 1.0, 10.0, 100.0],
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1.0]
            }
            param_prefix = 'classifier__'
        else:
            pipeline = SVC(probability=True)
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.01, 0.1, 1.0]
            }
            param_prefix = ''
        
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1)
        grid_search.fit(X_transformed, y_train)
        
        best_params = grid_search.best_params_
        self.C = best_params[f'{param_prefix}C']
        self.kernel = best_params[f'{param_prefix}kernel']
        self.gamma = best_params[f'{param_prefix}gamma']
        
        return self.build()
    
    def transform_text(self, tokenized_texts):

        if self.vectorizer is None:
            raise ValueError("Word2Vec model has not been trained or loaded yet.")
        return self.vectorizer.transform(tokenized_texts)
    
    def fit(self, tokenized_texts, y_train, train_word2vec=True, word2vec_path=None):

        # Train or load Word2Vec
        if train_word2vec:
            self.train_word2vec(tokenized_texts, save_path=word2vec_path)
        elif word2vec_path and self.word2vec_model is None:
            self.load_word2vec(word2vec_path)
        
        # Transform texts
        X_transformed = self.transform_text(tokenized_texts)
        
        # Build model if not built
        if self.model is None:
            self.build()
        
        # Fit the model
        self.model.fit(X_transformed, y_train)
        return self
    
    def predict(self, tokenized_texts):

        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return self.model.predict(self.transform_text(tokenized_texts))
    
    def predict_proba(self, tokenized_texts):

        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return self.model.predict_proba(self.transform_text(tokenized_texts))
    
    def evaluate(self, tokenized_texts, y_test):

        from sklearn.metrics import precision_score, recall_score, f1_score
        
        y_pred = self.predict(tokenized_texts)
        return {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
    
    def save_model(self, model_path, word2vec_path=None):

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        
        if word2vec_path and self.word2vec_model:
            os.makedirs(os.path.dirname(word2vec_path), exist_ok=True)
            self.word2vec_model.save(word2vec_path)
    
    def load_model(self, model_path, word2vec_path=None):

        if word2vec_path:
            self.load_word2vec(word2vec_path)
        
        self.model = joblib.load(model_path)
        return self