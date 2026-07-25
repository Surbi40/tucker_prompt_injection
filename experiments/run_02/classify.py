import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import json
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import f1_score

from src.preprocessing.tensor_builder import build_vocab, build_tensor
from src.decomposition.tucker_decomposer import run_tucker
from src.utils.config import load_config, get_project_root, set_seed

config = load_config()
set_seed(config["random_seed"])
ROOT = get_project_root()

# --- Primary config: validated as correct for low-label regime (see sensitivity_analysis_ds1.json
# and classify_ds1_both_configs.json — "efficient" outperformed "best_accuracy" at every label
# fraction while being ~2.4x faster to decompose) ---
WINDOW = 10
RANK = 30
VOCAB_SIZE = 500
LABEL_FRACTIONS = [0.20, 0.10, 0.05, 0.02, 0.01, 0.005]
SEEDS = [42, 123, 7]
N_FOLDS = 5

TUCKER_KWARGS = dict(init="random", tol=1e-4, n_iter_max=25)

# LabelSpreading is empirically confirmed (2 independent test rounds) to underperform
# Logistic Regression substantially at every label fraction and exhibits numerical
# instability. Kept here, gated off, for reproducibility/documentation only.
RUN_LABEL_SPREADING = False


def eval_logistic_regression(embeddings, train_labels, mask, test_labels, n_train):
    X_train = embeddings[:n_train][mask]
    X_test = embeddings[n_train:]
    y_train = train_labels[mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    start = time.perf_counter()
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    fit_time = time.perf_counter() - start

    return f1_score(test_labels, preds), fit_time


def eval_label_spreading(embeddings, train_labels, mask, test_labels, n_train, n_neighbors=7):
    X_train = embeddings[:n_train]
    X_test = embeddings[n_train:]

    y_semi = np.full(n_train, -1)
    y_semi[mask] = train_labels[mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_labeled = mask.sum()
    k = min(n_neighbors, max(2, n_labeled - 1))

    start = time.perf_counter()
    model = LabelSpreading(kernel="knn", n_neighbors=k)
    model.fit(X_train_s, y_semi)
    preds = model.predict(X_test_s)
    fit_time = time.perf_counter() - start

    return f1_score(test_labels, preds), fit_time


def make_labeled_mask(n_train, labels_train, fraction, seed):
    n_labeled = max(4, int(n_train * fraction))
    rng = np.random.RandomState(seed)
    positions = rng.choice(n_train, size=n_labeled, replace=False)
    mask = np.zeros(n_train, dtype=bool)
    mask[positions] = True
    if len(set(labels_train[mask])) < 2:
        return None
    return mask


def run_dataset(dataset_path, dataset_name, window=WINDOW, rank=RANK, vocab_size=VOCAB_SIZE):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = np.array([row["text"] for row in data])
    labels = np.array([row["label"] for row in data])

    print(f"\n{'='*70}\n{dataset_name} (n={len(texts)}, window={window}, rank={rank})\n{'='*70}")

    results = {frac: {"lr": [], "lr_time": [], "decomp_time": []} for frac in LABEL_FRACTIONS}
    if RUN_LABEL_SPREADING:
        for frac in LABEL_FRACTIONS:
            results[frac]["labelspread"] = []
            results[frac]["ls_time"] = []

    fold_num = 0
    total_folds = len(SEEDS) * N_FOLDS

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(texts, labels):
            fold_num += 1
            train_texts, test_texts = texts[train_idx], texts[test_idx]
            train_labels, test_labels = labels[train_idx], labels[test_idx]
            n_train = len(train_texts)

            vocab = build_vocab(list(train_texts), vocab_size=vocab_size, min_freq=2)

            # Transductive design — Tucker decomposition fit jointly on train+test texts.
            # Test samples contribute to co-occurrence tensor structure (unsupervised),
            # but their labels are never used during fitting.
            all_texts = list(train_texts) + list(test_texts)
            tensor = build_tensor(all_texts, vocab, window=window)

            fold_rank = [
                min(rank, tensor.shape[0]),
                min(rank, tensor.shape[1]),
                min(rank, tensor.shape[2] - 1),
            ]

            embeddings, core, factors, decomp_time = run_tucker(tensor, ranks=fold_rank, **TUCKER_KWARGS)

            print(f"    Fold {fold_num}/{total_folds} — decomposition: {decomp_time:.2f}s")

            for frac in LABEL_FRACTIONS:
                mask = make_labeled_mask(n_train, train_labels, frac, seed)
                if mask is None:
                    continue

                lr_f1, lr_t = eval_logistic_regression(embeddings, train_labels, mask, test_labels, n_train)
                results[frac]["lr"].append(lr_f1)
                results[frac]["lr_time"].append(lr_t)
                results[frac]["decomp_time"].append(decomp_time)

                if RUN_LABEL_SPREADING:
                    ls_f1, ls_t = eval_label_spreading(embeddings, train_labels, mask, test_labels, n_train)
                    results[frac]["labelspread"].append(ls_f1)
                    results[frac]["ls_time"].append(ls_t)

    header = f"{'Label %':<10}{'LR F1':<10}{'LR time(s)':<12}{'Decomp time(s)'}"
    if RUN_LABEL_SPREADING:
        header = f"{'Label %':<10}{'LR F1':<10}{'LabelSpread F1':<16}{'LR time(s)':<12}{'LS time(s)':<12}{'Decomp time(s)'}"
    print(f"\n{header}")

    for frac in LABEL_FRACTIONS:
        r = results[frac]
        if RUN_LABEL_SPREADING:
            print(f"{frac*100:<10.1f}{np.mean(r['lr']):<10.3f}{np.mean(r['labelspread']):<16.3f}"
                  f"{np.mean(r['lr_time']):<12.4f}{np.mean(r['ls_time']):<12.4f}{np.mean(r['decomp_time']):.2f}")
        else:
            print(f"{frac*100:<10.1f}{np.mean(r['lr']):<10.3f}{np.mean(r['lr_time']):<12.4f}{np.mean(r['decomp_time']):.2f}")

    return results


DATASETS = [
    ("DS2", "ds2_wambosec_clean.json"),
    ("DS3", "ds3_slabs_experiment.json"),
    ("DS4", "ds4_geekyrakshit_experiment.json"),
    ("Combined", "combined_clean.json"),
]

if __name__ == "__main__":
    out_dir = ROOT / "results" / "run_02"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, filename in DATASETS:
        dataset_path = ROOT / "datasets" / "prompt_injection" / filename
        results = run_dataset(dataset_path, name)

        out_path = out_dir / f"classify_{name.lower()}.json"
        with open(out_path, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        print(f"Saved {name} results to {out_path}")

    print("\nAll datasets complete.")