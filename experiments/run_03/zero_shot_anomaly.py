import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

from src.preprocessing.tensor_builder import build_vocab, build_tensor
from src.decomposition.tucker_decomposer import (
    run_tucker,
    score_by_centroid_distance,
    zero_label_threshold,
    gmm_threshold,
    kde_threshold,
)
from src.baselines.embedding_drift import embedding_drift_score
from src.utils.config import load_config, get_project_root, set_seed

config = load_config()
set_seed(config["random_seed"])
ROOT = get_project_root()

WINDOW = 10
RANK = 30
VOCAB_SIZE = 500
TUCKER_KWARGS = dict(init="random", tol=1e-4, n_iter_max=25)
SEED = 42


def oracle_best_f1(scores, labels):
    """
    Sweeps thresholds using TEST LABELS to find the best achievable F1.
    This is an upper-bound reference only — NOT a deployable number, since a
    real zero-shot system has no test labels to peek at. Kept for comparison
    against the genuinely zero-label methods below.
    """
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = np.nanargmax(f1s)
    return f1s[best_idx], thresholds[min(best_idx, len(thresholds) - 1)]


def evaluate_with_threshold_methods(train_emb, scores, test_labels, label_prefix):
    """
    Scores a set of anomaly scores against three zero-label threshold
    methods (percentile, GMM, KDE) plus the oracle upper bound. Returns a
    dict of {method_name: {"threshold":..., "f1":...}}.
    """
    results = {}

    thresh_pct = zero_label_threshold(train_emb, percentile=95)
    preds_pct = (scores > thresh_pct).astype(int)
    results["percentile_95"] = {"threshold": float(thresh_pct), "f1": float(f1_score(test_labels, preds_pct))}

    thresh_gmm = gmm_threshold(train_emb, n_std=2.5)
    preds_gmm = (scores > thresh_gmm).astype(int)
    results["gmm_2.5std"] = {"threshold": float(thresh_gmm), "f1": float(f1_score(test_labels, preds_gmm))}

    thresh_kde = kde_threshold(train_emb, quantile=0.95)
    preds_kde = (scores > thresh_kde).astype(int)
    results["kde_95"] = {"threshold": float(thresh_kde), "f1": float(f1_score(test_labels, preds_kde))}

    oracle_f1, oracle_thresh = oracle_best_f1(scores, test_labels)
    results["oracle_upper_bound"] = {"threshold": float(oracle_thresh), "f1": float(oracle_f1)}

    print(f"\n  {label_prefix} threshold comparison:")
    for name, r in results.items():
        print(f"    {name:<20} threshold={r['threshold']:.4f}  F1={r['f1']:.3f}")

    return results


def run_dataset(dataset_path, dataset_name):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = np.array([row["text"] for row in data])
    labels = np.array([row["label"] for row in data])

    # Simulate "cold start": split off a benign-only pool to build structure from,
    # evaluate anomaly scores on a held-out mixed test set. NO labels used to fit
    # the tensor, centroid, or threshold — labels only used afterward to score.
    benign_idx = np.where(labels == 0)[0]
    benign_pool, benign_holdout = train_test_split(benign_idx, test_size=0.3, random_state=SEED)

    test_idx = np.concatenate([benign_holdout, np.where(labels == 1)[0]])
    test_texts = texts[test_idx]
    test_labels = labels[test_idx]

    train_texts = texts[benign_pool]

    print(f"\n{'='*70}\n{dataset_name} — zero-shot structural anomaly detection\n"
          f"Benign-only fit set: {len(train_texts)} | Test set (mixed): {len(test_texts)}\n{'='*70}")

    # --- Tucker structural score ---
    vocab = build_vocab(list(train_texts), vocab_size=VOCAB_SIZE, min_freq=2)
    all_texts = list(train_texts) + list(test_texts)
    tensor = build_tensor(all_texts, vocab, window=WINDOW)
    fold_rank = [min(RANK, tensor.shape[0]), min(RANK, tensor.shape[1]), min(RANK, tensor.shape[2] - 1)]
    embeddings, core, factors, decomp_time = run_tucker(tensor, ranks=fold_rank, **TUCKER_KWARGS)

    train_emb = embeddings[:len(train_texts)]
    test_emb = embeddings[len(train_texts):]
    tucker_scores = score_by_centroid_distance(train_emb, test_emb)
    tucker_auc = roc_auc_score(test_labels, tucker_scores)

    print(f"\n  Tucker ROC-AUC: {tucker_auc:.3f}  (decomp time: {decomp_time:.2f}s)")
    tucker_thresholds = evaluate_with_threshold_methods(train_emb, tucker_scores, test_labels, "Tucker")

    # --- ZEDD-style embedding drift baseline ---
    # Needs its own benign-only embedding space to threshold against, computed
    # via the same sentence embedder used to score.
    from src.baselines.embedding_drift import get_embedder
    embedder = get_embedder()
    drift_train_emb = embedder.encode(list(train_texts))

    drift_scores = embedding_drift_score(train_texts, test_texts)
    drift_auc = roc_auc_score(test_labels, drift_scores)

    print(f"\n  Embedding drift ROC-AUC: {drift_auc:.3f}")
    drift_thresholds = evaluate_with_threshold_methods(drift_train_emb, drift_scores, test_labels, "Embedding drift")

    print(f"\n{'='*70}\nSUMMARY — {dataset_name}\n{'='*70}")
    print(f"{'Method':<25}{'ROC-AUC':<12}{'Percentile F1':<16}{'GMM F1':<12}{'KDE F1':<12}{'Oracle F1'}")
    print(f"{'Tucker (structural)':<25}{tucker_auc:<12.3f}"
          f"{tucker_thresholds['percentile_95']['f1']:<16.3f}"
          f"{tucker_thresholds['gmm_2.5std']['f1']:<12.3f}"
          f"{tucker_thresholds['kde_95']['f1']:<12.3f}"
          f"{tucker_thresholds['oracle_upper_bound']['f1']:.3f}")
    print(f"{'Embedding drift (ZEDD)':<25}{drift_auc:<12.3f}"
          f"{drift_thresholds['percentile_95']['f1']:<16.3f}"
          f"{drift_thresholds['gmm_2.5std']['f1']:<12.3f}"
          f"{drift_thresholds['kde_95']['f1']:<12.3f}"
          f"{drift_thresholds['oracle_upper_bound']['f1']:.3f}")

    return {
        "tucker": {"auc": float(tucker_auc), "decomp_time": float(decomp_time), "thresholds": tucker_thresholds},
        "embedding_drift": {"auc": float(drift_auc), "thresholds": drift_thresholds},
    }


DATASETS = [
    ("DS1", "ds1_deepset_clean.json"),
    ("DS2", "ds2_wambosec_clean.json"),
    ("DS3", "ds3_slabs_experiment.json"),
    ("DS4", "ds4_geekyrakshit_experiment.json"),
    ("Combined", "combined_clean.json"),
]

if __name__ == "__main__":
    out_dir = ROOT / "results" / "run_03"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for name, filename in DATASETS:
        dataset_path = ROOT / "datasets" / "prompt_injection" / filename
        results = run_dataset(dataset_path, name)
        all_results[name] = results

    with open(out_dir / "zero_shot_anomaly_results_v2.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved to {out_dir / 'zero_shot_anomaly_results_v2.json'}")