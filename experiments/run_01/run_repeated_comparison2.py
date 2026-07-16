import json
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import wilcoxon

from src.preprocessing.tensor_builder import build_vocabulary, build_dataset_tensor
from src.decomposition.tucker_decomposer import decompose_tucker
from src.decomposition.cp_decomposer import decompose_cp
from src.utils.config import load_config, get_project_root

config = load_config()
ROOT = get_project_root()

with open(ROOT / "datasets" / "prompt_injection" / "ds2_wambosec_clean.json") as f:
    data = json.load(f)

texts = np.array([row["text"] for row in data])
labels = np.array([row["label"] for row in data])

SUBSAMPLE_SIZE = 1200
if len(texts) > SUBSAMPLE_SIZE:
    texts, _, labels, _ = train_test_split(
        texts, labels,
        train_size=SUBSAMPLE_SIZE,
        stratify=labels,
        random_state=config["random_seed"]
    )
print(f"Using subsample: {len(texts)} rows")

tucker_rank = [30, 30, 15]
cp_rank = 15
N_REPEATS = 3
SEEDS = [42, 123, 7]

all_tucker_f1 = []
all_cp_f1 = []


def evaluate(embeddings, train_labels, test_labels, n_train):
    X_train = embeddings[:n_train]
    X_test = embeddings[n_train:]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_s, train_labels)
    preds = clf.predict(X_test_s)
    return f1_score(test_labels, preds)


for repeat_idx, seed in enumerate(SEEDS):
    print(f"\n=== Repeat {repeat_idx + 1}/{N_REPEATS} (seed={seed}) ===")
    skf = StratifiedKFold(n_splits=config["n_folds"], shuffle=True, random_state=seed)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
        train_texts, test_texts = texts[train_idx], texts[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]

        vocab = build_vocabulary(list(train_texts), max_vocab_size=config["max_vocab_size"])
        all_texts = list(train_texts) + list(test_texts)
        tensor = build_dataset_tensor(all_texts, vocab, window_size=config["window_size"])

        fold_tucker_rank = [tucker_rank[0], tucker_rank[1], min(tucker_rank[2], tensor.shape[2] - 1)]
        core, factors = decompose_tucker(tensor, rank=fold_tucker_rank)
        tucker_f1 = evaluate(factors[2], train_labels, test_labels, len(train_texts))

        fold_cp_rank = min(cp_rank, tensor.shape[2] - 1)
        weights, cp_factors = decompose_cp(tensor, rank=fold_cp_rank)
        cp_f1 = evaluate(cp_factors[2], train_labels, test_labels, len(train_texts))

        all_tucker_f1.append(tucker_f1)
        all_cp_f1.append(cp_f1)
        print(f"  Fold {fold_idx}: Tucker={tucker_f1:.3f}  CP={cp_f1:.3f}")

print(f"\n--- Final Summary (n={len(all_tucker_f1)} paired folds) ---")
print(f"Tucker: mean={np.mean(all_tucker_f1):.3f}  std={np.std(all_tucker_f1):.3f}")
print(f"CP:     mean={np.mean(all_cp_f1):.3f}  std={np.std(all_cp_f1):.3f}")

stat, p_value = wilcoxon(all_tucker_f1, all_cp_f1)
print(f"\nWilcoxon statistic: {stat}, p-value: {p_value:.4f}")
print("Significant at p<0.05" if p_value < 0.05 else "Not significant at p<0.05")

wins = sum(1 for t, c in zip(all_tucker_f1, all_cp_f1) if t > c)
print(f"Tucker wins: {wins}/{len(all_tucker_f1)} folds")

out_dir = ROOT / "results"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "tucker_vs_cp_repeated_ds2.json", "w") as f:
    json.dump({
        "tucker_f1": all_tucker_f1,
        "cp_f1": all_cp_f1,
        "wilcoxon_stat": float(stat),
        "p_value": float(p_value)
    }, f, indent=2)