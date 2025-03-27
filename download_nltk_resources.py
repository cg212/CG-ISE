"""
Script to download all required NLTK resources.

Run this script once before testing the models to ensure all necessary
NLTK resources are available.
"""

import nltk

def download_nltk_resources():
    """
    Download all required NLTK resources.
    """
    print("Downloading NLTK resources...")
    
    # Download essential resources
    resources = [
        'punkt',       # Tokenizer
        'stopwords',   # Stopwords corpus
        'wordnet',     # WordNet lexical database
        'omw-1.4'      # Open Multilingual WordNet
    ]
    
    for resource in resources:
        try:
            nltk.download(resource)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {e}")
    
    print("Download complete!")

if __name__ == "__main__":
    download_nltk_resources()