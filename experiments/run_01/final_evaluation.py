import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from src.preprocessing.tensor_builder import build_vocabulary, build_dataset_tensor
from src.decomposition.tucker_decomposer import decompose_tucker
from src.utils.config import load_config, get_project_root, set_seed

config = load_config()
set_seed(config["random_seed"])
ROOT = get_project_root()

BEST_WINDOW = 10
BEST_RANK_3 = 10
FIXED_VOCAB_RANK = 30
LABEL_FRACTIONS = [0.20, 0.10, 0.05, 0.02, 0.01, 0.005]
SEEDS = [42, 123, 7]

with open(ROOT / "results" / "joint_search_ds1.json") as f:
    search = json.load(f)

train_texts = np.array(search["train_texts"])
train_labels = np.array(search["train_labels"])
test_texts = np.array(search["test_texts"])
test_labels = np.array(search["test_labels"])

results = {frac: [] for frac in LABEL_FRACTIONS}

vocab = build_vocabulary(list(train_texts), max_vocab_size=config["max_vocab_size"])
all_texts = list(train_texts) + list(test_texts)
tensor = build_dataset_tensor(all_texts, vocab, window_size=BEST_WINDOW)
rank = [FIXED_VOCAB_RANK, FIXED_VOCAB_RANK, min(BEST_RANK_3, tensor.shape[2] - 1)]
core, factors = decompose_tucker(tensor, rank=rank)
embeddings = factors[2]
n_train = len(train_texts)

for seed in SEEDS:
    for frac in LABEL_FRACTIONS:
        n_labeled = max(4, int(n_train * frac))
        rng = np.random.RandomState(seed)
        positions = rng.choice(n_train, size=n_labeled, replace=False)
        mask = np.zeros(n_train, dtype=bool)
        mask[positions] = True
        labels_subset = train_labels[mask]
        if len(set(labels_subset)) < 2:
            continue

        X_train = embeddings[:n_train][mask]
        X_test = embeddings[n_train:]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_train_s, labels_subset)
        preds = clf.predict(X_test_s)
        f1 = f1_score(test_labels, preds)
        results[frac].append(f1)
        print(f"seed={seed} frac={frac*100}%: F1={f1:.3f}")

print(f"\n{'Label %':<10}{'Mean F1':<10}{'Std'}")
for frac in LABEL_FRACTIONS:
    vals = results[frac]
    print(f"{frac*100:<10.1f}{np.mean(vals):<10.3f}{np.std(vals):.3f}")

with open(ROOT / "results" / "final_ds1_results.json", "w") as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=2)