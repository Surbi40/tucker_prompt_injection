import json
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.preprocessing.tensor_builder import build_vocabulary, build_dataset_tensor
from src.decomposition.tucker_decomposer import decompose_tucker
from src.utils.config import load_config, get_project_root

config = load_config()
ROOT = get_project_root()

LABEL_FRACTIONS = [0.20, 0.10, 0.05, 0.02, 0.01, 0.005]
SEEDS = [42, 123, 7]

def evaluate_tucker(train_texts, train_labels_subset, subset_mask, test_texts, test_labels, vocab):
    all_texts = list(train_texts) + list(test_texts)
    tensor = build_dataset_tensor(all_texts, vocab, window_size=config["window_size"])
    rank = [30, 30, min(15, tensor.shape[2] - 1)]
    core, factors = decompose_tucker(tensor, rank=rank)
    embeddings = factors[2]

    X_train_full = embeddings[:len(train_texts)]
    X_test = embeddings[len(train_texts):]

    # Only use the labeled subset for training the classifier
    X_train_labeled = X_train_full[subset_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_labeled)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_s, train_labels_subset)
    preds = clf.predict(X_test_s)
    return f1_score(test_labels, preds)

def evaluate_tfidf_baseline(model_cls, train_texts, train_labels_subset, subset_mask, test_texts, test_labels):
    vectorizer = TfidfVectorizer(max_features=config["max_vocab_size"])
    X_train_full = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    X_train_labeled = X_train_full[subset_mask]

    clf = model_cls(class_weight="balanced") if model_cls is SVC else model_cls(class_weight="balanced", random_state=config["random_seed"])
    clf.fit(X_train_labeled, train_labels_subset)
    preds = clf.predict(X_test)
    return f1_score(test_labels, preds)

def run_sweep(dataset_path, subsample_size=None):
    with open(dataset_path) as f:
        data = json.load(f)
    texts = np.array([row["text"] for row in data])
    labels = np.array([row["label"] for row in data])

    if subsample_size and len(texts) > subsample_size:
        texts, _, labels, _ = train_test_split(
            texts, labels, train_size=subsample_size,
            stratify=labels, random_state=config["random_seed"]
        )

    results = {frac: {"tucker": [], "svm": [], "rf": []} for frac in LABEL_FRACTIONS}

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=config["n_folds"], shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(texts, labels):
            train_texts, test_texts = texts[train_idx], texts[test_idx]
            train_labels, test_labels = labels[train_idx], labels[test_idx]
            vocab = build_vocabulary(list(train_texts), max_vocab_size=config["max_vocab_size"])

            for frac in LABEL_FRACTIONS:
                n_labeled = max(4, int(len(train_texts) * frac))  # at least 4 labeled samples
                rng = np.random.RandomState(seed)
                subset_positions = rng.choice(len(train_texts), size=n_labeled, replace=False)
                subset_mask = np.zeros(len(train_texts), dtype=bool)
                subset_mask[subset_positions] = True
                train_labels_subset = train_labels[subset_mask]

                if len(set(train_labels_subset)) < 2:
                    continue  # skip if subset has only one class

                t_f1 = evaluate_tucker(train_texts, train_labels_subset, subset_mask, test_texts, test_labels, vocab)
                svm_f1 = evaluate_tfidf_baseline(SVC, train_texts, train_labels_subset, subset_mask, test_texts, test_labels)
                rf_f1 = evaluate_tfidf_baseline(RandomForestClassifier, train_texts, train_labels_subset, subset_mask, test_texts, test_labels)

                results[frac]["tucker"].append(t_f1)
                results[frac]["svm"].append(svm_f1)
                results[frac]["rf"].append(rf_f1)

    print(f"\n{'Label %':<10}{'Tucker':<12}{'SVM':<12}{'RF':<12}")
    for frac in LABEL_FRACTIONS:
        t = np.mean(results[frac]["tucker"])
        s = np.mean(results[frac]["svm"])
        r = np.mean(results[frac]["rf"])
        print(f"{frac*100:<10.1f}{t:<12.3f}{s:<12.3f}{r:<12.3f}")

    return results

print("=== DS1 (deepset) low-label sweep ===")
ds1_sweep = run_sweep(ROOT / "datasets" / "prompt_injection" / "ds1_deepset_clean.json")

out_dir = ROOT / "results"
with open(out_dir / "low_label_sweep_ds1.json", "w") as f:
    json.dump({str(k): v for k, v in ds1_sweep.items()}, f, indent=2)