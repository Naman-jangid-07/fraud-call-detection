"""
Download required NLTK data
"""

import nltk

print("Downloading NLTK data...")

# Download required datasets
datasets = [
    'punkt',
    'stopwords',
    'wordnet',
    'averaged_perceptron_tagger',
    'omw-1.4'
]

for dataset in datasets:
    try:
        nltk.download(dataset)
        print(f"✓ Downloaded {dataset}")
    except Exception as e:
        print(f"✗ Error downloading {dataset}: {e}")

print("\nNLTK data download complete!")