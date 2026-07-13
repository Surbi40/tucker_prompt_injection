import json
import numpy as np
from src.preprocessing.tensor_builder import build_vocabulary, build_dataset_tensor
from src.decomposition.tucker_decomposer import decompose_tucker, reconstruction_error
from src.utils.config import load_config, get_project_root

config = load_config()
ROOT = get_project_root()

with open(ROOT / "datasets" / "prompt_injection" / "ds1_deepset_clean.json") as f:
    data = json.load(f)

texts = [row["text"] for row in data[:50]]
labels = [row["label"] for row in data[:50]]

vocab = build_vocabulary(texts, max_vocab_size=config["max_vocab_size"])
tensor = build_dataset_tensor(texts, vocab, window_size=config["window_size"])
print(f"Dataset tensor shape: {tensor.shape}")

# third rank must be <= number of prompts (M=50 here)
rank = [20, 20, 10]  # V-mode, V-mode, prompt-mode ranks
core, factors = decompose_tucker(tensor, rank=rank)

print(f"Core shape: {core.shape}")
print(f"Factor 0 (vocab dim 1): {factors[0].shape}")
print(f"Factor 1 (vocab dim 2): {factors[1].shape}")
print(f"Factor 2 (prompt embeddings): {factors[2].shape}")

error = reconstruction_error(tensor, core, factors)
print(f"Reconstruction error: {error:.4f}")

# actual per-prompt feature matrix for classification
prompt_embeddings = factors[2]
print(f"\nPrompt embeddings shape: {prompt_embeddings.shape}")  # (50, 10)
print(f"First embedding (label={labels[0]}): {prompt_embeddings[0]}")
print(f"Second embedding (label={labels[1]}): {prompt_embeddings[1]}")