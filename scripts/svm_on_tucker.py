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
from src.classification.logreg_lasso import run_logreg_lasso_baseline
from src.classification.baselines import fit_tfidf, run_svm_baseline as run_svm_tfidf

ROOT = get_project_root()
out_dir = ROOT / "datasets" / "prompt_injection"

with open(out_dir / "ds1_deepset_clean.json", encoding="utf-8") as f:
    data = json.load(f)
texts = [d["text"] for d in data]
labels = np.array([d["label"] for d in data])

word2idx = build_vocab(texts, vocab_size=500, min_freq=2)
tensor = build_tensor(texts, word2idx, window=5, normalize=False)
embeddings, _, _, _ = run_tucker(tensor, ranks=(20, 20, 20), init="random", tol=1e-4, n_iter_max=25)

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
    clf.fit(emb_lab, y_lab)  # RF doesn't need scaling
    return clf.predict(emb_unlab)

N_SEEDS = 10
print(f"{'pct':>6} {'Tucker+SVM':>16} {'Tucker+RF':>16} {'Tucker+LR-L1':>16} {'TFIDF+SVM':>16}")
for pct in [0.20, 0.10, 0.05, 0.03, 0.02, 0.01, 0.005]:
    f1_tsvm, f1_trf, f1_lr, f1_tfidf_svm = [], [], [], []
    for seed in range(N_SEEDS):
        labeled_mask = stratified_labeled_mask(labels_train, pct, seed=seed)
        unlabeled_idx = ~labeled_mask
        y_unlabeled = labels_train[unlabeled_idx]
        if len(np.unique(y_unlabeled)) < 2:
            continue

        preds_tsvm = svm_on_tucker(emb_train[labeled_mask], labels_train[labeled_mask], emb_train[unlabeled_idx])
        f1_tsvm.append(f1_score(y_unlabeled, preds_tsvm))

        preds_trf = rf_on_tucker(emb_train[labeled_mask], labels_train[labeled_mask], emb_train[unlabeled_idx])
        f1_trf.append(f1_score(y_unlabeled, preds_trf))

        preds_lr, _, _ = run_logreg_lasso_baseline(emb_train[labeled_mask], labels_train[labeled_mask], emb_train[unlabeled_idx])
        f1_lr.append(f1_score(y_unlabeled, preds_lr))

        texts_labeled = [texts_train[i] for i in np.where(labeled_mask)[0]]
        texts_unlab = [texts_train[i] for i in np.where(unlabeled_idx)[0]]
        X_tr, X_te, _ = fit_tfidf(texts_labeled, texts_unlab)
        preds_tfidf_svm, _ = run_svm_tfidf(X_tr, labels_train[labeled_mask], X_te)
        f1_tfidf_svm.append(f1_score(y_unlabeled, preds_tfidf_svm))

    def fmt(r):
        return f"{np.mean(r):.3f}±{np.std(r):.3f}" if r else "n/a"
    print(f"{pct:>6.3f} {fmt(f1_tsvm):>16} {fmt(f1_trf):>16} {fmt(f1_lr):>16} {fmt(f1_tfidf_svm):>16}")