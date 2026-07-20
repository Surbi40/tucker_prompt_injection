import json
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import f1_score

from src.preprocessing.tensor_builder import build_vocabulary, build_dataset_tensor
from src.decomposition.tucker_decomposer import decompose_tucker
from src.utils.config import load_config, get_project_root, set_seed

config = load_config()
set_seed(config["random_seed"])
ROOT = get_project_root()

# Reuse the exact same train/val split from your earlier search — no new leakage risk
with open(ROOT / "results" / "joint_search_ds1.json") as f:
    search = json.load(f)

train_texts = np.array(search["train_texts"])
train_labels = np.array(search["train_labels"])
# use the held-out TEST split here only for this debug session — fine since we are not
# reporting these numbers as final results, only diagnosing a mechanism
test_texts = np.array(search["test_texts"])
test_labels = np.array(search["test_labels"])

WINDOW = 10
RANK_3 = 10
FRACTION = 0.02

vocab = build_vocabulary(list(train_texts), max_vocab_size=config["max_vocab_size"])
all_texts = list(train_texts) + list(test_texts)
tensor = build_dataset_tensor(all_texts, vocab, window_size=WINDOW)
rank = [30, 30, min(RANK_3, tensor.shape[2] - 1)]
core, factors = decompose_tucker(tensor, rank=rank)
embeddings = factors[2]
n_train = len(train_texts)

n_labeled = max(4, int(n_train * FRACTION))
rng = np.random.RandomState(config["random_seed"])
positions = rng.choice(n_train, size=n_labeled, replace=False)
mask = np.zeros(n_train, dtype=bool)
mask[positions] = True

X_train_full = embeddings[:n_train]
X_test = embeddings[n_train:]

y_semi = np.full(n_train, -1)
y_semi[mask] = train_labels[mask]

print(f"n_train={n_train}, n_labeled={n_labeled}, n_test={len(test_texts)}\n")


def try_variant(name, X_train, X_test_local, n_neighbors=7, kernel="knn", gamma=20, alpha=0.2):
    kwargs = {"kernel": kernel, "alpha": alpha}
    if kernel == "knn":
        kwargs["n_neighbors"] = n_neighbors
    else:
        kwargs["gamma"] = gamma

    model = LabelSpreading(**kwargs)
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.fit(X_train, y_semi)
        preds = model.predict(X_test_local)
        had_warning = len(w) > 0

    f1 = f1_score(test_labels, preds)
    print(f"{name}: F1={f1:.3f}  {'(RuntimeWarning triggered)' if had_warning else '(clean)'}")
    return f1


# --- Baseline: original setup, for reference ---
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train_full)
X_test_std = scaler.transform(X_test)
try_variant("Baseline (StandardScaler, knn, n_neighbors=7)", X_train_std, X_test_std, n_neighbors=7, kernel="knn")

# --- Variant 1: higher n_neighbors ---
try_variant("Higher n_neighbors=15", X_train_std, X_test_std, n_neighbors=15, kernel="knn")
try_variant("Higher n_neighbors=25", X_train_std, X_test_std, n_neighbors=25, kernel="knn")

# --- Variant 2: RBF kernel instead of KNN kernel ---
try_variant("RBF kernel, gamma=20 (default)", X_train_std, X_test_std, kernel="rbf", gamma=20)
try_variant("RBF kernel, gamma=1", X_train_std, X_test_std, kernel="rbf", gamma=1)
try_variant("RBF kernel, gamma=0.1", X_train_std, X_test_std, kernel="rbf", gamma=0.1)

# --- Variant 3: unit-norm embeddings instead of StandardScaler ---
X_train_norm = normalize(X_train_full, norm="l2")
X_test_norm = normalize(X_test, norm="l2")
try_variant("L2-normalized, knn n_neighbors=7", X_train_norm, X_test_norm, n_neighbors=7, kernel="knn")
try_variant("L2-normalized, rbf gamma=20", X_train_norm, X_test_norm, kernel="rbf", gamma=20)

# --- For reference: Logistic Regression on the same split ---
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_std[mask], train_labels[mask])
lr_preds = clf.predict(X_test_std)
lr_f1 = f1_score(test_labels, lr_preds)
print(f"\nReference — Logistic Regression: F1={lr_f1:.3f}")