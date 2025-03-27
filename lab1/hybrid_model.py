"""HybridModel for performance-related bug report classification."""

import numpy as np
import re
import os
import spacy
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from scipy.sparse import issparse

warnings.filterwarnings("ignore", message=r"\[W007\]", category=UserWarning)

class HybridModel(BaseEstimator, ClassifierMixin):
    def __init__(self, framework='tensorflow', use_meta_features=True, 
                 use_pattern_features=True, use_code_features=True,
                 perf_terms_weight=4.0, max_features=10000, ngram_range=(1, 3),
                 smote_sampling=True, use_smote=None, voting_weights=None, classification_threshold=0.4):
        self.framework = framework
        self.use_meta_features = use_meta_features
        self.use_pattern_features = use_pattern_features
        self.use_code_features = use_code_features
        self.perf_terms_weight = perf_terms_weight
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.smote_sampling = use_smote if use_smote is not None else smote_sampling
        self.use_smote = self.smote_sampling
        self.voting_weights = voting_weights
        self.classification_threshold = classification_threshold
        
        # Model components
        self.tfidf_vectorizer = self.tfidf_classifier = self.meta_model = None
        self.pattern_model = self.recall_optimizer = self.ensemble_model = self.meta_scaler = None
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import sys
            os.system(f"{sys.executable} -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        self._init_framework_resources()
    
    def _init_framework_resources(self):
        # Common performance terms across all frameworks
        self.common_perf_terms = {
            'slow', 'fast', 'speed', 'performance', 'latency', 'throughput', 'memory',
            'leak', 'efficient', 'inefficient', 'bottleneck', 'overhead', 'optimization',
            'optimize', 'optimizing', 'optimized', 'faster', 'slower', 'speedup', 'cpu',
            'gpu', 'cuda', 'profile', 'profiling', 'benchmark', 'benchmarking', 'time',
            'consumption', 'runtime', 'timeout', 'hang', 'crash', 'response', 'responsive',
            'unresponsive', 'resource', 'utilization', 'usage', 'intensive', 'compute', 
            'computing', 'acceleration', 'accelerated', 'accelerate', 'inference'
        }
        
        # Framework-specific performance terms
        framework_terms = {
            'tensorflow': {'xla', 'tpu', 'eager', 'graph', 'session', 'tensorrt', 'distribute', 
                'distribution', 'parallel', 'tf.function', 'autograph', 'trace', 'jit', 
                'compile', 'profiler', 'timeline', 'device', 'placement', 'allocator'},
            'pytorch': {'jit', 'trace', 'script', 'torchscript', 'autograd', 'cuda', 'cudnn', 
                'nvtx', 'backward', 'execution', 'profile', 'amp', 'autocast', 'scaler',
                'ddp', 'dataparallel', 'onnx', 'fuser', 'fusion'},
            'keras': {'compile', 'fit', 'predict', 'evaluate', 'backend', 'worker', 'distribution',
                'multiprocessing', 'generator', 'callback', 'batch', 'profile'},
            'mxnet': {'hybridize', 'symbol', 'imperative', 'executor', 'profile', 
                'kvstore', 'dist', 'distributed', 'amp', 'gluon', 'gpu', 'context'},
            'incubatormxnet': {'hybridize', 'symbol', 'imperative', 'executor', 'profile', 
                'kvstore', 'dist', 'distributed', 'amp', 'gluon', 'gpu', 'context'},
            'caffe': {'solver', 'forward', 'backward', 'layer', 'blob', 'syncedmem', 'gpu', 
                'allocate', 'prefetch', 'batch', 'multi', 'thread'}
        }
        
        # Combine common and framework-specific terms
        self.perf_terms = self.common_perf_terms.copy()
        if self.framework in framework_terms:
            self.perf_terms.update(framework_terms[self.framework])
        
        # Performance-related regex patterns
        self.perf_patterns = [
            r'takes?( too)? (long|much) time', r'slow(er|ly)?', r'(high|excessive) latency',
            r'(fast|faster|quickly|quick)', r'(execution|run|running) time',
            r'memory (leak|usage|consumption|intensive)', r'out of memory', r'oom',
            r'(cpu|gpu) utilization', r'(not|isn\'t) efficient', r'(more|less) efficient',
            r'performance (issue|problem|bug|degradation)',
            r'(slow|fast) (training|inference)', r'batch size', r'throughput',
            r'(\d+) times (slower|faster)', r'compared to', r'performance (regression|improvement)',
            r'(profile|benchmark|measure)(ing|ed)?', r'(before|after) optimization',
            r'tak(e|es|ing) (too )?much (time|memory)', r'(very|too|extremely) (slow|sluggish)',
            r'hang(s|ing)?', r'(poor|bad|sub-optimal) performance',
            r'(large|high) (memory )?footprint', r'(long|slow) (load|start)(ing)? time',
            r'resource (usage|consumption)', r'freezes?', r'unresponsive(ness)?'
        ]
        
        # Add framework-specific patterns
        if self.framework == 'tensorflow':
            self.perf_patterns.extend([r'(eager|graph) mode', r'xla compilation', r'tf\.function',
                r'session runtime', r'out of memory error', r'resource exhausted'])
        elif self.framework == 'pytorch':
            self.perf_patterns.extend([r'torch(script|jit)', r'cuda (performance|synchronization)',
                r'autograd', r'backward pass', r'cuda out of memory'])
        elif self.framework == 'keras':
            self.perf_patterns.extend([r'steps? per second', r'epoch time', r'training time'])
        
        # Compile patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.perf_patterns]
        
        # Code patterns
        self.code_patterns = {
            'markdown_block': re.compile(r'```[\s\S]*?```'),
            'inline_code': re.compile(r'`.*?`'),
            'html_code': re.compile(r'<code>[\s\S]*?</code>'),
            'stack_trace': re.compile(r'Traceback \(most recent call last\):[\s\S]*?(?:\n\n|\Z)')
        }
    
    def _custom_tokenizer(self, text):
        if not isinstance(text, str) or not text:
            return []
        
        # Extract and save code blocks
        code_blocks = []
        for pattern in self.code_patterns.values():
            matches = pattern.findall(text)
            for match in matches:
                code_blocks.append(match)
                text = text.replace(match, f" __CODE_BLOCK_{len(code_blocks)}__ ")
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Extract tokens
        tokens = []
        for token in doc:
            if token.is_punct or len(token.text) < 2: 
                continue
            
            # Weight performance terms higher
            if token.text.lower() in self.perf_terms:
                tokens.extend([token.lemma_.lower()] * int(self.perf_terms_weight))
            elif not token.is_stop:
                tokens.append(token.lemma_.lower())
        
        # Process code blocks if enabled
        if self.use_code_features and code_blocks:
            for block in code_blocks:
                code_tokens = self._extract_code_tokens(block)
                tokens.extend([f"code_{t}" for t in code_tokens])
        
        return tokens
    
    def _extract_code_tokens(self, code_block):
        # Remove code block markers
        code = re.sub(r'```.*?\n|```|`|<code>|</code>', '', code_block)
        
        # Extract useful elements
        methods = [m.rstrip('(') for m in re.findall(r'\b\w+\s*\(', code)]
        variables = [v.rstrip('=').strip() for v in re.findall(r'\b\w+\s*=', code)]
        api_refs = re.findall(r'\b\w+\.\w+', code)
        imports = re.findall(r'import\s+(\w+)', code) + re.findall(r'from\s+(\w+)', code)
        
        # Performance keywords
        perf_code_keywords = ['cuda', 'gpu', 'cpu', 'device', 'memory', 'profile', 
                           'benchmark', 'time', 'optimize', 'performance',
                           'latency', 'throughput', 'duration', 'oom', 'leak']
        
        perf_words = [word for word in code.split() if word.lower() in perf_code_keywords]
        
        # Combine and filter
        code_tokens = methods + variables + api_refs + imports + perf_words
        return [token.lower() for token in code_tokens if len(token) > 1]
    
    def _extract_pattern_features(self, text):
        if not isinstance(text, str) or not self.use_pattern_features:
            return {}
        
        features = {}
        
        # Count pattern matches
        for i, pattern in enumerate(self.compiled_patterns):
            features[f'pattern_{i}'] = min(len(pattern.findall(text)), 3)
        
        # Count performance terms
        perf_term_count = sum(len(re.findall(r'\b' + re.escape(term) + r'\b', text.lower())) 
                              for term in self.perf_terms)
        
        features['perf_term_count'] = min(perf_term_count, 20)
        features['perf_term_density'] = perf_term_count / (len(text.split()) + 1)
        
        return features
    
    def _extract_meta_features(self, text):
        if not isinstance(text, str) or not self.use_meta_features:
            return {}
        
        features = {}
        words = text.split()
        
        # Basic length features
        features['text_length'] = len(text)
        features['word_count'] = len(words)
        
        # Code-related features
        if self.use_code_features:
            code_blocks = self.code_patterns['markdown_block'].findall(text)
            inline_codes = self.code_patterns['inline_code'].findall(text)
            stack_traces = self.code_patterns['stack_trace'].findall(text)
            
            features['code_block_count'] = len(code_blocks)
            features['inline_code_count'] = len(inline_codes)
            features['has_stack_trace'] = 1 if stack_traces else 0
            
            code_text = ''.join(code_blocks + inline_codes + stack_traces)
            features['code_ratio'] = len(code_text) / (len(text) + 1)
        
        # Misc features
        features['question_mark_count'] = text.count('?')
        features['exclamation_mark_count'] = text.count('!')
        
        # Sentence complexity
        sentences = text.split('.')
        features['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences if s.strip()]) if len(sentences) > 1 else len(words)
        
        # Capitalization
        uppercase_words = sum(1 for word in words if word.isupper() and len(word) > 1)
        features['uppercase_ratio'] = uppercase_words / (len(words) + 1)
        
        return features
    
    def fit(self, X, y):
        # Validation
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        X = [x if isinstance(x, str) else str(x) for x in X]
        pos_count = sum(y)
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=self._custom_tokenizer,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True
        )
        X_tfidf = self.tfidf_vectorizer.fit_transform(X)
        self.tfidf_classifier = MultinomialNB(class_prior=[0.4, 0.6])
        
        # Extract meta and pattern features
        meta_features = None
        pattern_features = None
        
        if self.use_meta_features:
            meta_features = np.array([list(self._extract_meta_features(text).values()) for text in X])
            if meta_features.size > 0:
                self.meta_scaler = StandardScaler()
                meta_features = self.meta_scaler.fit_transform(meta_features)
                self.meta_model = RandomForestClassifier(n_estimators=100, class_weight={0: 1, 1: 1.5}, random_state=42)
        
        if self.use_pattern_features:
            pattern_features = np.array([list(self._extract_pattern_features(text).values()) for text in X])
            if pattern_features.size > 0:
                self.pattern_model = SVC(probability=True, class_weight={0: 1, 1: 2}, random_state=42)
        
        # Train models with SMOTE if enabled
        if self.smote_sampling:
            # TF-IDF model
            X_tfidf_array = X_tfidf.toarray() if issparse(X_tfidf) else X_tfidf
            X_tfidf_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_tfidf_array, y)
            self.tfidf_classifier.fit(X_tfidf_resampled, y_resampled)
            
            # Meta model
            if self.use_meta_features and meta_features is not None and meta_features.size > 0:
                meta_features_resampled, y_meta_resampled = SMOTE(random_state=42).fit_resample(meta_features, y)
                self.meta_model.fit(meta_features_resampled, y_meta_resampled)
            
            # Pattern model
            if self.use_pattern_features and pattern_features is not None and pattern_features.size > 0:
                pattern_features_resampled, y_pattern_resampled = SMOTE(random_state=42).fit_resample(pattern_features, y)
                self.pattern_model.fit(pattern_features_resampled, y_pattern_resampled)
        else:
            self.tfidf_classifier.fit(X_tfidf, y)
            if self.use_meta_features and meta_features is not None and meta_features.size > 0:
                self.meta_model.fit(meta_features, y)
            if self.use_pattern_features and pattern_features is not None and pattern_features.size > 0:
                self.pattern_model.fit(pattern_features, y)
        
        # Prepare ensemble feature array
        feature_arrays = [X_tfidf.toarray() if issparse(X_tfidf) else X_tfidf]
        if self.use_meta_features and meta_features is not None and meta_features.size > 0:
            feature_arrays.append(meta_features)
        if self.use_pattern_features and pattern_features is not None and pattern_features.size > 0:
            feature_arrays.append(pattern_features)
        
        X_ensemble = np.hstack(feature_arrays) if len(feature_arrays) > 1 else feature_arrays[0]
        
        # Create ensemble model
        estimators = [
            ('lr_tfidf', LogisticRegression(max_iter=1000, class_weight={0: 1, 1: 1.5}, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, class_weight={0: 1, 1: 1.5}, random_state=42)),
            ('svm', SVC(probability=True, class_weight='balanced', random_state=42))
        ]
        
        self.recall_optimizer = LogisticRegression(max_iter=1000, class_weight={0: 1, 1: 2.5}, C=0.5, random_state=42)
        estimators.append(('recall_optimizer', self.recall_optimizer))
        
        weights = self.voting_weights or [2, 1, 1, 1.5][:len(estimators)]
        
        self.ensemble_model = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
        
        # Train ensemble
        if self.smote_sampling:
            X_ensemble_resampled, y_ensemble_resampled = SMOTE(random_state=42).fit_resample(X_ensemble, y)
            self.ensemble_model.fit(X_ensemble_resampled, y_ensemble_resampled)
        else:
            self.ensemble_model.fit(X_ensemble, y)
        
        return self
    
    def predict(self, X):
        if self.tfidf_vectorizer is None or self.ensemble_model is None:
            raise ValueError("Model has not been trained yet.")
        
        probas = self.predict_proba(X)
        return (probas[:, 1] >= self.classification_threshold).astype(int)
    
    def predict_proba(self, X):
        if self.tfidf_vectorizer is None or self.ensemble_model is None:
            raise ValueError("Model has not been trained yet.")
        
        X = [x if isinstance(x, str) else str(x) for x in X]
        X_tfidf = self.tfidf_vectorizer.transform(X)
        
        # Prepare feature arrays
        feature_arrays = [X_tfidf.toarray() if issparse(X_tfidf) else X_tfidf]
        
        if self.use_meta_features and hasattr(self, 'meta_scaler'):
            meta_features = np.array([list(self._extract_meta_features(text).values()) for text in X])
            meta_features = self.meta_scaler.transform(meta_features)
            feature_arrays.append(meta_features)
        
        if self.use_pattern_features:
            pattern_features = np.array([list(self._extract_pattern_features(text).values()) for text in X])
            feature_arrays.append(pattern_features)
        
        X_ensemble = np.hstack(feature_arrays) if len(feature_arrays) > 1 else feature_arrays[0]
        return self.ensemble_model.predict_proba(X_ensemble)
    
    def evaluate(self, X, y):
        from sklearn.metrics import precision_score, recall_score, f1_score
        y_pred = self.predict(X)
        
        return {
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred)
        }
    
    def get_top_features(self, n=20):
        if self.tfidf_vectorizer is None or self.tfidf_classifier is None:
            raise ValueError("Model has not been trained yet.")
        
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # For MultinomialNB, log probability ratios show feature importance
        if isinstance(self.tfidf_classifier, MultinomialNB):
            feature_importances = self.tfidf_classifier.feature_log_prob_[1] - self.tfidf_classifier.feature_log_prob_[0]
            top_indices = np.argsort(feature_importances)[-n:]
            return [(feature_names[i], feature_importances[i]) for i in top_indices[::-1]]
        
        # For other classifiers with coef_ attribute
        elif hasattr(self.tfidf_classifier, 'coef_'):
            feature_importances = self.tfidf_classifier.coef_[0]
            top_indices = np.argsort(feature_importances)[-n:]
            return [(feature_names[i], feature_importances[i]) for i in top_indices[::-1]]
            
        return []