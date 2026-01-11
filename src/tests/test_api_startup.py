import os

# Get the directory where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Move up one level to the project root, then into data/models
model_path = os.path.join(base_dir, "..", "data", "models", "best_model.pt")
tokenizer_path = os.path.join(base_dir, "..", "data", "models", "tokenizer")

print(f"Checking model files at: {os.path.abspath(model_path)}")
print(f"Model exists: {os.path.isfile(model_path)}")
print(f"Tokenizer exists: {os.path.exists(tokenizer_path)}")