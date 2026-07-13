import json
from src.preprocessing.tensor_builder import build_vocabulary, cooccurrence_tensor
from src.decomposition.tucker_decomposer import decompose_tucker, extract_tucker_features, reconstruction_error
from src.utils.config import load_config, get_project_root

config = load_config()
ROOT = get_project_root()

with open(ROOT/"datasets"/"prompt_injection"/"ds1_deepset_clean.json") as f:
    data = json.load(f)

texts = [row["text"] for row in data[:20]]
labels = [row["label"] for row in data[:20]]

print(f"Sample texts (first 3):")
for t, l in zip(texts[:3], labels[:3]):
    print(f"  [{l}] {t[:80]}...")

vocab = build_vocabulary(texts, max_vocab_size=config["max_vocab_size"])
print(f"\nVocab size from 20 samples: {len(vocab)}")

tensor = cooccurrence_tensor(texts[0], vocab, window_size=config["window_size"])
print(f"Tensor shape for first sample: {tensor.shape}")
print(f"Non-zero entries: {(tensor > 0).sum()} / {tensor.size}")

core, factors = decompose_tucker(tensor, rank=config["tucker_rank"])
error = reconstruction_error(tensor, core, factors)
print(f"\nReconstruction error on real text: {error:.4f}")

feature = extract_tucker_features(core, factors)
print(f"Feature vector length: {len(feature)}")
