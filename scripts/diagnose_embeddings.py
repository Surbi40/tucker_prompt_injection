import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

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

vocab = build_vocabulary(texts, max_vocab_size=config["max_vocab_size"])
tensor = build_dataset_tensor(texts, vocab, window_size=config["window_size"])

rank = [30, 30, 15]
core, factors = decompose_tucker(tensor, rank=rank)
embeddings = factors[2]

norms = np.linalg.norm(embeddings, axis=1)
print("Embedding L2 norm stats:")
print(f"  min={norms.min():.6f}  max={norms.max():.6f}  "
      f"mean={norms.mean():.6f}  median={np.median(norms):.6f}")
print(f"  Norms below 0.01: {(norms < 0.01).sum()} / {len(norms)}")

lengths = np.array([len(t.split()) for t in texts])
correlation = np.corrcoef(norms, lengths)[0, 1]
print(f"\nCorrelation between embedding norm and prompt word-count: {correlation:.3f}")


X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=config["random_seed"], stratify=labels
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_scaled, y_train)
preds = clf.predict(X_test_scaled)

print(f"\n--- With StandardScaler + class_weight='balanced' ---")
print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
print(f"F1: {f1_score(y_test, preds):.3f}")
print(f"Confusion matrix:\n{confusion_matrix(y_test, preds)}")
print(classification_report(y_test, preds))