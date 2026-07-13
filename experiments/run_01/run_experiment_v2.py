import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

from src.preprocessing.tensor_builder import build_vocabulary, build_dataset_tensor
from src.decomposition.tucker_decomposer import decompose_tucker
from src.utils.config import load_config, get_project_root, set_seed

config = load_config()
set_seed(config["random_seed"])
ROOT = get_project_root()

with open(ROOT / "datasets" / "prompt_injection" / "ds1_deepset_clean.json") as f:
    data = json.load(f)

texts = np.array([row["text"] for row in data])
labels = np.array([row["label"] for row in data])

rank = [30, 30, 15]
skf = StratifiedKFold(n_splits=config["n_folds"], shuffle=True, random_state=config["random_seed"])

fold_results = []

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
    train_texts, test_texts = texts[train_idx], texts[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]

    # Vocab from TRAIN split only
    vocab = build_vocabulary(list(train_texts), max_vocab_size=config["max_vocab_size"])

    all_texts = list(train_texts) + list(test_texts)
    tensor = build_dataset_tensor(all_texts, vocab, window_size=config["window_size"])

    fold_rank = [rank[0], rank[1], min(rank[2], tensor.shape[2] - 1)]
    core, factors = decompose_tucker(tensor, rank=fold_rank)
    embeddings = factors[2]

    X_train = embeddings[:len(train_texts)]
    X_test = embeddings[len(train_texts):]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_scaled, train_labels)
    preds = clf.predict(X_test_scaled)

    f1 = f1_score(test_labels, preds)
    acc = accuracy_score(test_labels, preds)
    fold_results.append({"fold": fold_idx, "f1": f1, "accuracy": acc})
    print(f"Fold {fold_idx}: F1={f1:.3f}  Accuracy={acc:.3f}")

f1_scores = [r["f1"] for r in fold_results]
print(f"\nMean F1: {np.mean(f1_scores):.3f} (+/- {np.std(f1_scores):.3f})")

out_dir = ROOT / "results"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "tucker_ds1_results.json", "w") as f:
    json.dump(fold_results, f, indent=2)