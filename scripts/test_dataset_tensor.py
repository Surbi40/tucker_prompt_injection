# scripts/test_dataset_tensor.py
import json
from src.preprocessing.tensor_builder import build_vocabulary, build_dataset_tensor
from src.utils.config import load_config, get_project_root

config = load_config()
ROOT = get_project_root()

with open(ROOT / "datasets" / "prompt_injection" / "ds1_deepset_clean.json") as f:
    data = json.load(f)

texts = [row["text"] for row in data[:50]] 

vocab = build_vocabulary(texts, max_vocab_size=config["max_vocab_size"])
tensor = build_dataset_tensor(texts, vocab, window_size=config["window_size"])

print(f"Vocab size: {len(vocab)}")
print(f"Dataset tensor shape: {tensor.shape}")  # should be (V, V, 50)
print(f"Non-zero fraction: {(tensor > 0).sum() / tensor.size:.4%}")