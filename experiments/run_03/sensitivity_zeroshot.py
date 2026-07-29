import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.preprocessing.tensor_builder import build_vocab, build_tensor
from src.decomposition.tucker_decomposer import run_tucker, score_by_centroid_distance
from src.utils.config import load_config, get_project_root, set_seed

config = load_config()
set_seed(config["random_seed"])
ROOT = get_project_root()

# Smaller, targeted grid — not the full 30-combo sweep, just enough to check
# whether window=10/rank=30 (chosen for a DIFFERENT task) still holds up here
WINDOW_SIZES = [5, 10, 15]
RANKS = [10, 30, 50]
VOCAB_SIZE = 500
TUCKER_KWARGS = dict(init="random", tol=1e-4, n_iter_max=25)
SEED = 42

with open(ROOT / "datasets" / "prompt_injection" / "ds1_deepset_clean.json", "r", encoding="utf-8") as f:
    data = json.load(f)
texts = np.array([row["text"] for row in data])
labels = np.array([row["label"] for row in data])

benign_idx = np.where(labels == 0)[0]
benign_pool, benign_holdout = train_test_split(benign_idx, test_size=0.3, random_state=SEED)
test_idx = np.concatenate([benign_holdout, np.where(labels == 1)[0]])
test_texts = texts[test_idx]
test_labels = labels[test_idx]
train_texts = texts[benign_pool]

print(f"Benign-only fit set: {len(train_texts)} | Test set (mixed): {len(test_texts)}\n")

results = []

for window_size in WINDOW_SIZES:
    for rank_val in RANKS:
        vocab = build_vocab(list(train_texts), vocab_size=VOCAB_SIZE, min_freq=2)
        all_texts = list(train_texts) + list(test_texts)
        tensor = build_tensor(all_texts, vocab, window=window_size)

        fold_rank = [
            min(rank_val, tensor.shape[0]),
            min(rank_val, tensor.shape[1]),
            min(rank_val, tensor.shape[2] - 1),
        ]

        embeddings, core, factors, decomp_time = run_tucker(tensor, ranks=fold_rank, **TUCKER_KWARGS)

        train_emb = embeddings[:len(train_texts)]
        test_emb = embeddings[len(train_texts):]
        scores = score_by_centroid_distance(train_emb, test_emb)

        auc = roc_auc_score(test_labels, scores)
        results.append({"window": window_size, "rank": rank_val, "auc": float(auc), "decomp_time": float(decomp_time)})
        print(f"window={window_size}, rank={rank_val}: AUC={auc:.3f}  [decomp {decomp_time:.2f}s]")

best = max(results, key=lambda r: r["auc"])
print(f"\nBest config for zero-shot anomaly detection: window={best['window']}, rank={best['rank']}, AUC={best['auc']:.3f}")

out_dir = ROOT / "results" / "run_03"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "sensitivity_zero_shot_ds1.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved to {out_dir / 'sensitivity_zero_shot_ds1.json'}")