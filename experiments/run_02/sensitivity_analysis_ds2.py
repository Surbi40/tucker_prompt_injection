import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import json
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from src.preprocessing.tensor_builder import build_vocab, build_tensor
from src.decomposition.tucker_decomposer import run_tucker
from src.utils.config import load_config, get_project_root, set_seed

config = load_config()
set_seed(config["random_seed"])
ROOT = get_project_root()

WINDOW_SIZES = [3, 5, 10, 15, 20, 25]
RANKS = [1, 10, 30, 50, 80]
N_FOLDS = 5
SEED = 42
VOCAB_SIZE = 500
SEARCH_SUBSAMPLE_SIZE = 2000  # matching DS3/DS4 experiment scale

TUCKER_KWARGS = dict(init="random", tol=1e-4, n_iter_max=25)

with open(ROOT / "datasets" / "prompt_injection" / "ds2_wambosec_clean.json", "r", encoding="utf-8") as f:
    data = json.load(f)
texts = np.array([row["text"] for row in data])
labels = np.array([row["label"] for row in data])

if len(texts) > SEARCH_SUBSAMPLE_SIZE:
    texts, _, labels, _ = train_test_split(
        texts, labels, train_size=SEARCH_SUBSAMPLE_SIZE,
        stratify=labels, random_state=SEED
    )
    print(f"DS2 subsampled to {len(texts)} rows for sensitivity search "
          f"(full DS2 is too large for a 30-combination grid)")

results = []

for window_size in WINDOW_SIZES:
    for rank_val in RANKS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_f1s = []
        fold_decomp_times = []
        start = time.time()

        for train_idx, test_idx in skf.split(texts, labels):
            train_texts, test_texts = texts[train_idx], texts[test_idx]
            train_labels, test_labels = labels[train_idx], labels[test_idx]

            vocab = build_vocab(list(train_texts), vocab_size=VOCAB_SIZE, min_freq=2)
            all_texts = list(train_texts) + list(test_texts)
            tensor = build_tensor(all_texts, vocab, window=window_size)

            fold_rank = [
                min(rank_val, tensor.shape[0]),
                min(rank_val, tensor.shape[1]),
                min(rank_val, tensor.shape[2] - 1),
            ]

            embeddings, core, factors, decomp_time = run_tucker(tensor, ranks=fold_rank, **TUCKER_KWARGS)
            fold_decomp_times.append(decomp_time)

            X_train = embeddings[:len(train_texts)]
            X_test = embeddings[len(train_texts):]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(X_train_s, train_labels)
            preds = clf.predict(X_test_s)
            fold_f1s.append(f1_score(test_labels, preds))

        elapsed = time.time() - start
        mean_f1 = np.mean(fold_f1s)
        mean_decomp_time = np.mean(fold_decomp_times)

        results.append({
            "window_size": window_size,
            "rank": rank_val,
            "mean_f1": float(mean_f1),
            "std_f1": float(np.std(fold_f1s)),
            "runtime_sec": elapsed,
            "mean_decomp_time_sec": float(mean_decomp_time),
        })
        print(f"window={window_size}, rank={rank_val}: F1={mean_f1:.3f} (+/-{np.std(fold_f1s):.3f})  "
              f"[total {elapsed:.1f}s, decomp avg {mean_decomp_time:.2f}s/fold]")

out_dir = ROOT / "results" / "run_02"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "sensitivity_analysis_ds2.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to results/run_02/sensitivity_analysis_ds2.json")
print(f"Total combinations tested: {len(results)}")