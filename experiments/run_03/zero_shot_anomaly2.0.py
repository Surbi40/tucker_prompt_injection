import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
import json
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, average_precision_score
from src.preprocessing.tensor_builder import build_vocab, build_tensor, prompt_cooccurrence_matrix
from src.decomposition.tucker_decomposer import (
    run_tucker, zero_label_threshold, gmm_threshold, kde_threshold,
)
from src.decomposition.tucker_projection import project_new_prompt
from src.baselines.embedding_drift import embedding_drift_score, get_embedder
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
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = np.nanargmax(f1s)
    return f1s[best_idx], thresholds[min(best_idx, len(thresholds) - 1)]


def evaluate_with_threshold_methods(train_emb, scores, test_labels, label_prefix):
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


def diagnose_score_distributions(train_emb, test_emb, test_labels, centroid):
    """
    Print self-distance (benign fit set) vs. test-score distributions,
    split by true label, to diagnose the oracle/real-threshold gap.
    """
    self_dists = np.linalg.norm(train_emb - centroid, axis=1)
    test_dists = np.linalg.norm(test_emb - centroid, axis=1)
    benign_test = test_dists[test_labels == 0]
    inject_test = test_dists[test_labels == 1]

    print(f"\n  Distribution diagnostic:")
    print(f"    Benign fit-set self-distances: mean={self_dists.mean():.4f} std={self_dists.std():.4f} "
          f"p95={np.percentile(self_dists, 95):.4f}")
    print(f"    Test benign distances:         mean={benign_test.mean():.4f} std={benign_test.std():.4f}")
    print(f"    Test injection distances:      mean={inject_test.mean():.4f} std={inject_test.std():.4f}")
    print(f"    (If test-benign mean >> fit-set self-distance mean, the benign fit set "
          f"isn't representative of the held-out benign distribution.)")


def run_ds1_induced():
    with open(ROOT / "datasets" / "prompt_injection" / "ds1_deepset_clean.json", encoding="utf-8") as f:
        data = json.load(f)
    texts = np.array([row["text"] for row in data])
    labels = np.array([row["label"] for row in data])

    benign_idx = np.where(labels == 0)[0]
    benign_pool, benign_holdout = train_test_split(benign_idx, test_size=0.3, random_state=SEED)
    test_idx = np.concatenate([benign_holdout, np.where(labels == 1)[0]])
    test_texts = texts[test_idx]
    test_labels = labels[test_idx]
    train_texts = texts[benign_pool]

    print(f"DS1 — zero-shot structural anomaly detection (TRUE INDUCTION)")
    print(f"Benign-only fit set: {len(train_texts)} | Test set (mixed): {len(test_texts)}\n")

    # --- Fit ONCE on benign-only data ---
    vocab = build_vocab(list(train_texts), vocab_size=VOCAB_SIZE, min_freq=2)
    train_tensor = build_tensor(list(train_texts), vocab, window=WINDOW)
    fold_rank = [min(RANK, train_tensor.shape[0]), min(RANK, train_tensor.shape[1]), min(RANK, train_tensor.shape[2] - 1)]

    fit_start = time.perf_counter()
    train_emb, core, factors, decomp_time = run_tucker(train_tensor, ranks=fold_rank, **TUCKER_KWARGS)
    fit_time = time.perf_counter() - fit_start
    A, B, C = factors
    centroid = train_emb.mean(axis=0)
    print(f"  Benign-only fit time: {fit_time:.2f}s (this happens ONCE, not per query)")

    # --- Project each test prompt individually — no re-decomposition ---
    proj_start = time.perf_counter()
    test_emb = np.zeros((len(test_texts), train_emb.shape[1]))
    for i, text in enumerate(test_texts):
        cooc = prompt_cooccurrence_matrix(text, vocab, window=WINDOW).toarray()
        test_emb[i] = project_new_prompt(cooc, core, A, B)
    proj_time = time.perf_counter() - proj_start
    per_prompt_ms = (proj_time / len(test_texts)) * 1000
    print(f"  Projection time: {proj_time:.3f}s total for {len(test_texts)} prompts "
          f"({per_prompt_ms:.2f}ms/prompt — this IS the deployable per-query latency)")

    tucker_scores = np.linalg.norm(test_emb - centroid, axis=1)
    tucker_auc = roc_auc_score(test_labels, tucker_scores)
    tucker_prauc = average_precision_score(test_labels, tucker_scores)
    print(f"\n  Tucker ROC-AUC: {tucker_auc:.3f}   PR-AUC: {tucker_prauc:.3f}")

    diagnose_score_distributions(train_emb, test_emb, test_labels, centroid)
    tucker_thresholds = evaluate_with_threshold_methods(train_emb, tucker_scores, test_labels, "Tucker (induced)")

    # --- ZEDD-style baseline (already inductive by construction — pretrained encoder) ---
    embedder = get_embedder()
    drift_train_emb = embedder.encode(list(train_texts))
    drift_scores = embedding_drift_score(train_texts, test_texts)
    drift_auc = roc_auc_score(test_labels, drift_scores)
    drift_prauc = average_precision_score(test_labels, drift_scores)
    print(f"\n  Embedding drift ROC-AUC: {drift_auc:.3f}   PR-AUC: {drift_prauc:.3f}")
    drift_thresholds = evaluate_with_threshold_methods(drift_train_emb, drift_scores, test_labels, "Embedding drift")

    print(f"\n{'='*70}\nSUMMARY — DS1 (induced)\n{'='*70}")
    print(f"{'Method':<25}{'ROC-AUC':<10}{'PR-AUC':<10}{'Pctile F1':<12}{'GMM F1':<10}{'KDE F1':<10}{'Oracle F1'}")
    print(f"{'Tucker (structural)':<25}{tucker_auc:<10.3f}{tucker_prauc:<10.3f}"
          f"{tucker_thresholds['percentile_95']['f1']:<12.3f}"
          f"{tucker_thresholds['gmm_2.5std']['f1']:<10.3f}"
          f"{tucker_thresholds['kde_95']['f1']:<10.3f}"
          f"{tucker_thresholds['oracle_upper_bound']['f1']:.3f}")
    print(f"{'Embedding drift (ZEDD)':<25}{drift_auc:<10.3f}{drift_prauc:<10.3f}"
          f"{drift_thresholds['percentile_95']['f1']:<12.3f}"
          f"{drift_thresholds['gmm_2.5std']['f1']:<10.3f}"
          f"{drift_thresholds['kde_95']['f1']:<10.3f}"
          f"{drift_thresholds['oracle_upper_bound']['f1']:.3f}")

    return {
        "tucker": {"auc": float(tucker_auc), "pr_auc": float(tucker_prauc), "fit_time": float(fit_time),
                   "per_prompt_ms": float(per_prompt_ms), "thresholds": tucker_thresholds},
        "embedding_drift": {"auc": float(drift_auc), "pr_auc": float(drift_prauc), "thresholds": drift_thresholds},
    }


if __name__ == "__main__":
    out_dir = ROOT / "results" / "run_03"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = run_ds1_induced()
    with open(out_dir / "zero_shot_ds1_induced.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir / 'zero_shot_ds1_induced.json'}")