import nltk
from nltk.tokenize import word_tokenize

print("NLTK version:", nltk.__version__)
print("Import successful!")

# Test tokenization
text = "This is a test sentence."
tokens = word_tokenize(text)
print("Tokens:", tokens)