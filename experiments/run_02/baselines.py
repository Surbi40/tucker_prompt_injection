import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import json
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from src.utils.config import load_config, get_project_root, set_seed

config = load_config()
set_seed(config["random_seed"])
ROOT = get_project_root()

VOCAB_SIZE = 500
LABEL_FRACTIONS = [0.20, 0.10, 0.05, 0.02, 0.01, 0.005]
SEEDS = [42, 123, 7]
N_FOLDS = 5


def eval_baseline(model_cls, X_train_full, train_labels, mask, X_test, test_labels):
    X_train = X_train_full[mask]
    y_train = train_labels[mask]

    start = time.perf_counter()
    if model_cls is SVC:
        clf = model_cls(class_weight="balanced")
    else:
        clf = model_cls(class_weight="balanced", random_state=config["random_seed"])
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
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


def run_dataset(dataset_path, dataset_name, vocab_size=VOCAB_SIZE):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = np.array([row["text"] for row in data])
    labels = np.array([row["label"] for row in data])

    print(f"\n{'='*70}\n{dataset_name} baselines (n={len(texts)})\n{'='*70}")

    results = {frac: {"svm": [], "rf": [], "svm_time": [], "rf_time": []} for frac in LABEL_FRACTIONS}

    fold_num = 0
    total_folds = len(SEEDS) * N_FOLDS

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(texts, labels):
            fold_num += 1
            train_texts, test_texts = texts[train_idx], texts[test_idx]
            train_labels, test_labels = labels[train_idx], labels[test_idx]
            n_train = len(train_texts)

            vectorizer = TfidfVectorizer(max_features=vocab_size)
            X_train_full = vectorizer.fit_transform(train_texts)
            X_test = vectorizer.transform(test_texts)

            for frac in LABEL_FRACTIONS:
                mask = make_labeled_mask(n_train, train_labels, frac, seed)
                if mask is None:
                    continue

                svm_f1, svm_t = eval_baseline(SVC, X_train_full, train_labels, mask, X_test, test_labels)
                rf_f1, rf_t = eval_baseline(RandomForestClassifier, X_train_full, train_labels, mask, X_test, test_labels)

                results[frac]["svm"].append(svm_f1)
                results[frac]["rf"].append(rf_f1)
                results[frac]["svm_time"].append(svm_t)
                results[frac]["rf_time"].append(rf_t)

            if fold_num % 5 == 0:
                print(f"    Fold {fold_num}/{total_folds} done")

    print(f"\n{'Label %':<10}{'SVM F1':<10}{'RF F1':<10}{'SVM time(s)':<14}{'RF time(s)'}")
    for frac in LABEL_FRACTIONS:
        r = results[frac]
        print(f"{frac*100:<10.1f}{np.mean(r['svm']):<10.3f}{np.mean(r['rf']):<10.3f}"
              f"{np.mean(r['svm_time']):<14.4f}{np.mean(r['rf_time']):.4f}")

    return results


DATASETS = [
    ("DS1", "ds1_deepset_clean.json"),
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

        out_path = out_dir / f"baselines_{name.lower()}.json"
        with open(out_path, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        print(f"Saved {name} baselines to {out_path}")

    print("\nAll baseline runs complete.")