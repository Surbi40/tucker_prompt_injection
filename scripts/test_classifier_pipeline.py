import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from src.utils.config import get_project_root
from src.preprocessing.tensor_builder import build_vocab, build_tensor
from src.decomposition.tucker_decomposer import run_tucker
from src.classification.sampling import stratified_labeled_mask
from src.classification.baselines import fit_tfidf, run_svm_baseline, run_rf_baseline
from src.classification.logreg_lasso import run_logreg_lasso_baseline, self_training_logreg_lasso

ROOT = get_project_root()
out_dir = ROOT / "datasets" / "prompt_injection"

with open(out_dir / "ds1_deepset_clean.json", encoding="utf-8") as f:
    data = json.load(f)
texts = [d["text"] for d in data]
labels = np.array([d["label"] for d in data])

word2idx = build_vocab(texts, vocab_size=500, min_freq=2)
tensor = build_tensor(texts, word2idx, window=5, normalize=False)
embeddings, _, _, t_tucker = run_tucker(tensor, ranks=(20, 20, 20), init="random", tol=1e-4, n_iter_max=25)
print(f"Tucker: {embeddings.shape}, {t_tucker:.2f}s")

idx_train, idx_test = train_test_split(np.arange(len(labels)), test_size=0.3, stratify=labels, random_state=42)
emb_train, labels_train = embeddings[idx_train], labels[idx_train]
texts_train = [texts[i] for i in idx_train]

N_SEEDS = 10  # bump to 30-50 once this runs cleanly, matching the paper's resampling protocol
PCTS = [0.20, 0.10, 0.05, 0.03, 0.02, 0.01, 0.005]

print(f"\nRunning {N_SEEDS} seeds per percentage point...\n")
print(f"{'pct':>6} {'LR-L1_sup':>18} {'LR-L1_self':>18} {'SVM':>18} {'RF':>18}")

for pct in PCTS:
    f1_lr_sup_runs, f1_lr_self_runs, f1_svm_runs, f1_rf_runs = [], [], [], []
    nonzero_counts = []

    for seed in range(N_SEEDS):
        labeled_mask = stratified_labeled_mask(labels_train, pct, seed=seed)
        unlabeled_idx = ~labeled_mask
        y_unlabeled = labels_train[unlabeled_idx]

        # skip degenerate seeds where a class is entirely absent from the unlabeled eval set
        if len(np.unique(y_unlabeled)) < 2:
            continue

        # LR-L1 supervised (scaled Tucker embeddings)
        preds_sup, _, clf_sup = run_logreg_lasso_baseline(
            emb_train[labeled_mask], labels_train[labeled_mask], emb_train[unlabeled_idx]
        )
        f1_lr_sup_runs.append(f1_score(y_unlabeled, preds_sup))
        nonzero_counts.append(np.sum(clf_sup.coef_ != 0))

        # LR-L1 self-training (semi-supervised, scaled Tucker embeddings)
        try:
            preds_self_all, _, _ = self_training_logreg_lasso(emb_train, labels_train, labeled_mask, threshold=0.75)
            f1_lr_self_runs.append(f1_score(y_unlabeled, preds_self_all[unlabeled_idx]))
        except Exception:
            pass  # skip failed seeds rather than crash the whole sweep

        # SVM (TF-IDF) baseline
        texts_labeled = [texts_train[i] for i in np.where(labeled_mask)[0]]
        texts_unlab = [texts_train[i] for i in np.where(unlabeled_idx)[0]]
        X_tr, X_te, _ = fit_tfidf(texts_labeled, texts_unlab)
        preds_svm, _ = run_svm_baseline(X_tr, labels_train[labeled_mask], X_te)
        f1_svm_runs.append(f1_score(y_unlabeled, preds_svm))

        # RF (TF-IDF) baseline
        preds_rf, _ = run_rf_baseline(X_tr, labels_train[labeled_mask], X_te)
        f1_rf_runs.append(f1_score(y_unlabeled, preds_rf))

    def fmt(runs):
        if not runs:
            return "   n/a (all seeds degenerate)"
        return f"{np.mean(runs):.3f}±{np.std(runs):.3f} (n={len(runs)})"

    n_lab_example = stratified_labeled_mask(labels_train, pct, seed=0).sum()
    print(f"{pct:>6.3f} {fmt(f1_lr_sup_runs):>18} {fmt(f1_lr_self_runs):>18} "
          f"{fmt(f1_svm_runs):>18} {fmt(f1_rf_runs):>18}")
    print(f"       (n_labeled≈{n_lab_example}, avg nonzero coefs={np.mean(nonzero_counts):.1f}/{emb_train.shape[1]})")