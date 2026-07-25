import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from src.utils.config import get_project_root
from src.preprocessing.tensor_builder import build_vocab, build_tensor
from src.decomposition.tucker_decomposer import run_tucker
from src.classification.sampling import stratified_labeled_mask
from src.classification.label_propagation import knn_label_propagation
from src.classification.logreg_lasso import run_logreg_lasso_baseline
from src.classification.baselines import fit_tfidf, run_svm_baseline as run_svm_tfidf, run_rf_baseline as run_rf_tfidf

ROOT = get_project_root()
out_dir = ROOT / "datasets" / "prompt_injection"

with open(out_dir / "ds2_wambosec_clean.json", encoding="utf-8") as f:
    data = json.load(f)
texts = [d["text"] for d in data]
labels = np.array([d["label"] for d in data])
print(f"DS2: {len(texts)} prompts, benign={sum(labels==0)}, injection={sum(labels==1)}")

word2idx = build_vocab(texts, vocab_size=300, min_freq=2)  # vocab=300, matching earlier DS2 memory constraints
tensor = build_tensor(texts, word2idx, window=5, normalize=False)
print(f"Tensor: {tensor.shape}, {tensor.nbytes/1e6:.0f} MB")

embeddings, _, _, t_tucker = run_tucker(tensor, ranks=(20, 20, 20), init="random", tol=1e-4, n_iter_max=25)
print(f"Tucker: {embeddings.shape}, {t_tucker:.2f}s")

idx_train, idx_test = train_test_split(np.arange(len(labels)), test_size=0.3, stratify=labels, random_state=42)
emb_train, labels_train = embeddings[idx_train], labels[idx_train]
texts_train = [texts[i] for i in idx_train]


def svm_on_tucker(emb_lab, y_lab, emb_unlab):
    scaler = StandardScaler().fit(emb_lab)
    clf = LinearSVC(class_weight="balanced", random_state=42, max_iter=10000)
    clf.fit(scaler.transform(emb_lab), y_lab)
    return clf.predict(scaler.transform(emb_unlab))


def rf_on_tucker(emb_lab, y_lab, emb_unlab):
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    clf.fit(emb_lab, y_lab)
    return clf.predict(emb_unlab)


N_SEEDS = 10
PCTS = [0.20, 0.10, 0.05, 0.03, 0.02, 0.01, 0.005]

print(f"\n{'pct':>6} {'Tucker+KNN':>14} {'Tucker+SVM':>14} {'Tucker+RF':>14} "
      f"{'Tucker+LR-L1':>14} {'TFIDF+SVM':>14} {'TFIDF+RF':>14}")

for pct in PCTS:
    results = {k: [] for k in ["knn", "tsvm", "trf", "lr", "tfidf_svm", "tfidf_rf"]}
    for seed in range(N_SEEDS):
        labeled_mask = stratified_labeled_mask(labels_train, pct, seed=seed)
        unlabeled_idx = ~labeled_mask
        y_unlabeled = labels_train[unlabeled_idx]
        if len(np.unique(y_unlabeled)) < 2:
            continue

        preds_knn, _, _ = knn_label_propagation(emb_train, labels_train, labeled_mask, n_neighbors=7, alpha=0.2)
        results["knn"].append(f1_score(y_unlabeled, preds_knn[unlabeled_idx]))

        preds_tsvm = svm_on_tucker(emb_train[labeled_mask], labels_train[labeled_mask], emb_train[unlabeled_idx])
        results["tsvm"].append(f1_score(y_unlabeled, preds_tsvm))

        preds_trf = rf_on_tucker(emb_train[labeled_mask], labels_train[labeled_mask], emb_train[unlabeled_idx])
        results["trf"].append(f1_score(y_unlabeled, preds_trf))

        preds_lr, _, _ = run_logreg_lasso_baseline(emb_train[labeled_mask], labels_train[labeled_mask], emb_train[unlabeled_idx])
        results["lr"].append(f1_score(y_unlabeled, preds_lr))

        texts_labeled = [texts_train[i] for i in np.where(labeled_mask)[0]]
        texts_unlab = [texts_train[i] for i in np.where(unlabeled_idx)[0]]
        X_tr, X_te, _ = fit_tfidf(texts_labeled, texts_unlab)

        preds_tfidf_svm, _ = run_svm_tfidf(X_tr, labels_train[labeled_mask], X_te)
        results["tfidf_svm"].append(f1_score(y_unlabeled, preds_tfidf_svm))

        preds_tfidf_rf, _ = run_rf_tfidf(X_tr, labels_train[labeled_mask], X_te)
        results["tfidf_rf"].append(f1_score(y_unlabeled, preds_tfidf_rf))

    def fmt(k):
        r = results[k]
        return f"{np.mean(r):.3f}±{np.std(r):.3f}" if r else "n/a"

    print(f"{pct:>6.3f} {fmt('knn'):>14} {fmt('tsvm'):>14} {fmt('trf'):>14} "
          f"{fmt('lr'):>14} {fmt('tfidf_svm'):>14} {fmt('tfidf_rf'):>14}")