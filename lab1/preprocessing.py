import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download required NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextPreprocessor:
    """
    A class for preprocessing text data in bug reports with special handling for
    technical content and code snippets.
    """
    
    def __init__(self, remove_code=True, keep_tech_terms=True):
        """
        Initialize the text preprocessor.
        
        Parameters:
        -----------
        remove_code : bool
            Whether to remove code blocks during preprocessing
        keep_tech_terms : bool
            Whether to preserve technical terms (don't remove as stopwords)
        """
        self.remove_code = remove_code
        self.keep_tech_terms = keep_tech_terms
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add custom stop words that might be common in bug reports but not informative
        self.custom_stop_words = {'error', 'bug', 'issue', 'problem', 'fix',
                                 'please', 'thank', 'thanks', 'help', 'using',
                                 'example', 'code', 'file', 'used', 'following',
                                 'see', 'version', 'get', 'use', 'try', 'tried',
                                 'like', 'run', 'running', 'want', 'need', 'make'}
        
        # Combine regular stop words with custom ones
        self.stop_words = self.stop_words.union(self.custom_stop_words)
        
        # Technical terms by project that should be preserved
        self.tech_terms = {
            "tensorflow": ["tf", "tensor", "graph", "session", "keras", "estimator", 
                          "tpu", "tensorboard", "eager", "gradient", "optimizer", 
                          "training", "inference", "loss", "accuracy", "checkpoint",
                          "serving", "saved_model", "hub", "dataset", "queues"],
            
            "pytorch": ["torch", "tensor", "autograd", "nn", "optim", "cuda", "jit", 
                       "distributed", "multiprocessing", "backward", "parameter", 
                       "module", "dataloader", "dataset", "sampler", "transform",
                       "scripting", "tracing", "amp", "hub", "torchaudio", "torchvision"],
            
            "keras": ["model", "layer", "sequential", "functional", "backend", "callback",
                     "optimizer", "loss", "metrics", "activation", "batch", "epoch", 
                     "training", "validation", "compile", "evaluate", "predict", 
                     "checkpoint", "regularizer", "initializer"],
            
            "incubator-mxnet": ["mx", "symbol", "ndarray", "gluon", "kvstore", "executor",
                               "module", "optimizer", "trainer", "context", "gpu", "cpu",
                               "hybridize", "forward", "backward", "gradient", "parameter",
                               "block", "network"],
            
            "caffe": ["blob", "solver", "layer", "net", "proto", "caffe", "model",
                     "weight", "data", "forward", "backward", "train", "test",
                     "deploy", "loss", "accuracy", "gpu", "cpu", "memory", "batch"]
        }
        
        # Compile regex patterns for code detection
        self.code_patterns = {
            'markdown_block': re.compile(r'```[\s\S]*?```'),
            'inline_code': re.compile(r'`.*?`'),
            'html_code': re.compile(r'<code>[\s\S]*?</code>'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        }
    
    def remove_html(self, text):
        """
        Remove HTML tags from text.
        
        Parameters:
        -----------
        text : str
            Input text
            
        Returns:
        --------
        str
            Text with HTML tags removed
        """
        html_pattern = re.compile(r'<.*?>')
        return html_pattern.sub('', text)
    
    def remove_emojis(self, text):
        """
        Remove emojis from text.
        
        Parameters:
        -----------
        text : str
            Input text
            
        Returns:
        --------
        str
            Text with emojis removed
        """
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"  # enclosed characters
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub('', text)
    
    def extract_code_features(self, text):
        """
        Extract features related to code presence in the text.
        
        Parameters:
        -----------
        text : str
            Input text
            
        Returns:
        --------
        dict
            Dictionary of code-related features
        """
        if pd.isna(text) or text is None:
            text = ""
        else:
            text = str(text)
            
        features = {}
        
        # Check for code blocks
        code_blocks = self.code_patterns['markdown_block'].findall(text)
        features['has_code_block'] = 1 if len(code_blocks) > 0 else 0
        features['num_code_blocks'] = len(code_blocks)
        
        # Check for inline code
        inline_code = self.code_patterns['inline_code'].findall(text)
        features['has_inline_code'] = 1 if len(inline_code) > 0 else 0
        features['num_inline_code'] = len(inline_code)
        
        # Check for HTML code tags
        html_code = self.code_patterns['html_code'].findall(text)
        features['has_html_code'] = 1 if len(html_code) > 0 else 0
        
        # Check for stack traces
        stack_trace_patterns = [
            r'Traceback \(most recent call last\)',
            r'Exception in thread',
            r'at[\s\S]*?\.java:\d+\)',
            r'File ".*?", line \d+'
        ]
        
        has_stack_trace = any(re.search(pattern, text) is not None for pattern in stack_trace_patterns)
        features['has_stack_trace'] = 1 if has_stack_trace else 0
        
        return features
    
    def preprocess_text(self, text, project=None):
        """
        Preprocess text with special handling for code snippets and technical terms.
        
        Parameters:
        -----------
        text : str
            Input text
        project : str, optional
            Project name for preserving technical terms
            
        Returns:
        --------
        str
            Preprocessed text
        """
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove URLs
        text = self.code_patterns['url'].sub(' URL ', text)
        
        # Handle code blocks based on configuration
        if self.remove_code:
            # Replace code blocks with placeholder
            text = self.code_patterns['markdown_block'].sub(' CODE_BLOCK ', text)
            text = self.code_patterns['inline_code'].sub(' CODE_SNIPPET ', text)
            text = self.code_patterns['html_code'].sub(' CODE_BLOCK ', text)
        
        # Remove HTML tags
        text = self.remove_html(text)
        
        # Remove emojis
        text = self.remove_emojis(text)
        
        # Create a set of technical terms to preserve if project is specified
        project_tech_terms = set()
        if self.keep_tech_terms and project is not None:
            if project in self.tech_terms:
                project_tech_terms = set(self.tech_terms[project])
        
        # Tokenize words (simple split instead of word_tokenize)
        tokens = text.lower().split()
        
        # Remove punctuation, stopwords, and lemmatize
        processed_tokens = []
        for token in tokens:
            # Remove punctuation at start and end
            token = token.strip(string.punctuation)
            
            # Skip empty tokens or single characters
            if not token or len(token) < 2:
                continue
                
            # Keep technical terms even if they're in stopwords
            if token in project_tech_terms:
                processed_tokens.append(token)
                continue
                
            # Skip stopwords
            if token in self.stop_words:
                continue
                
            # Apply lemmatization
            lemma = self.lemmatizer.lemmatize(token)
            processed_tokens.append(lemma)
        
        return ' '.join(processed_tokens)
    
    def preprocess_dataframe(self, df, text_columns=None, project=None):
        """
        Preprocess text columns in a dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        text_columns : list, optional
            List of text columns to preprocess
        project : str, optional
            Project name for preserving technical terms
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with preprocessed text columns
        """
        if text_columns is None:
            text_columns = ['Title', 'Body']
        
        # Create a copy of the dataframe
        processed_df = df.copy()
        
        # Process each text column
        for col in text_columns:
            if col in processed_df.columns:
                processed_df[f'{col}_processed'] = processed_df[col].apply(
                    lambda x: self.preprocess_text(x, project)
                )
        
        # Extract code features if 'Body' is one of the columns
        if 'Body' in text_columns and 'Body' in processed_df.columns:
            code_features = processed_df['Body'].apply(self.extract_code_features)
            
            # Extract each feature into a separate column
            for feature in ['has_code_block', 'num_code_blocks', 'has_inline_code', 
                           'num_inline_code', 'has_html_code', 'has_stack_trace']:
                processed_df[feature] = code_features.apply(lambda x: x.get(feature, 0))
        
        return processed_df
    
    def extract_structural_features(self, row):
        """
        Extract structural features from a bug report.
        
        Parameters:
        -----------
        row : pd.Series
            Row from the dataframe representing a bug report
            
        Returns:
        --------
        dict
            Dictionary of structural features
        """
        features = {}
        
        # Extract length-based features
        title = str(row['Title']) if pd.notna(row.get('Title')) else ""
        body = str(row['Body']) if pd.notna(row.get('Body')) else ""
        
        features['title_length'] = len(title.split())
        features['body_length'] = len(body.split())
        features['title_body_ratio'] = features['title_length'] / max(features['body_length'], 1)
        
        # Check for question marks in title (often indicates confusion/issues)
        features['title_has_question'] = 1 if '?' in title else 0
        
        # Number of comments if available
        if 'Comments' in row and pd.notna(row['Comments']):
            comments = str(row['Comments'])
            features['has_comments'] = 1
            features['comment_length'] = len(comments.split())
        else:
            features['has_comments'] = 0
            features['comment_length'] = 0
        
        # Extract code features from body
        code_features = self.extract_code_features(body)
        features.update(code_features)
        
        return features


# Example usage of the class (commented out)
if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    # Example text with various elements
    text = """
    I'm having an issue with TensorFlow 2.3.0 on GPU. 
    When I try to run the following code:
    
    ```python
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    ```
    
    I get this error:
    ```
    Traceback (most recent call last):
      File "test.py", line 2, in <module>
        import tensorflow as tf
      File "/usr/local/lib/python3.8/site-packages/tensorflow/__init__.py", line 41, in <module>
        from tensorflow.python.eager import context
    ImportError: cannot import name 'context' from 'tensorflow.python.eager'
    ```
    
    Has anyone else encountered this? I'm using CUDA 10.1 and cuDNN 7.6.5.
    """
    
    # Preprocess the text
    processed_text = preprocessor.preprocess_text(text, project="tensorflow")
    print(f"Processed text:\n{processed_text}")
    
    # Extract code features
    code_features = preprocessor.extract_code_features(text)
    print(f"\nCode features:\n{code_features}")