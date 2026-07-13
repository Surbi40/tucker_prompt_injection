import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

from src.preprocessing.tensor_builder import build_vocabulary, build_dataset_tensor
from src.decomposition.tucker_decomposer import decompose_tucker
from src.utils.config import load_config, get_project_root, set_seed

config = load_config()
set_seed(config["random_seed"])
ROOT = get_project_root()

with open(ROOT / "datasets" / "prompt_injection" / "ds1_deepset_clean.json") as f:
    data = json.load(f)

texts = [row["text"] for row in data]
labels = np.array([row["label"] for row in data])

print(f"Total samples: {len(texts)}")

# Build vocab + tensor on FULL dataset for this quick test
# (proper train-only vocab comes in the real CV step next)
vocab = build_vocabulary(texts, max_vocab_size=config["max_vocab_size"])
tensor = build_dataset_tensor(texts, vocab, window_size=config["window_size"])
print(f"Tensor shape: {tensor.shape}")

rank = [30, 30, 15]
core, factors = decompose_tucker(tensor, rank=rank)
embeddings = factors[2]  # (num_prompts, 15)

X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=config["random_seed"], stratify=labels
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, preds):.3f}")
print(f"F1: {f1_score(y_test, preds):.3f}")
print(f"Majority-class baseline (always predict most common label): "
      f"{max(np.bincount(y_test)) / len(y_test):.3f}")