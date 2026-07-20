import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import f1_score

from src.preprocessing.tensor_builder import build_vocabulary, build_dataset_tensor
from src.decomposition.tucker_decomposer import decompose_tucker
from src.utils.config import load_config, get_project_root, set_seed

config = load_config()
set_seed(config["random_seed"])
ROOT = get_project_root()

WINDOW_SIZES = [3, 5, 10]
THIRD_MODE_RANKS = [10, 15, 20]
FIXED_VOCAB_RANK = 30
LOW_LABEL_FRACTION = 0.02  # search under a realistic low-label condition, not full labels


def eval_logistic_regression(embeddings, train_labels, subset_mask, val_labels, n_train):
    X_train_full = embeddings[:n_train]
    X_val = embeddings[n_train:]
    X_train_labeled = X_train_full[subset_mask]
    y_labeled = train_labels[subset_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_labeled)
    X_val_s = scaler.transform(X_val)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_s, y_labeled)
    preds = clf.predict(X_val_s)
    return f1_score(val_labels, preds)


def eval_label_spreading(embeddings, train_labels, subset_mask, val_labels, n_train):
    X_train_full = embeddings[:n_train]
    X_val = embeddings[n_train:]

    y_semi = np.full(n_train, -1)
    y_semi[subset_mask] = train_labels[subset_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_full)
    X_val_s = scaler.transform(X_val)

    n_labeled = subset_mask.sum()
    n_neighbors = min(7, max(2, n_labeled - 1))

    model = LabelSpreading(kernel="knn", n_neighbors=n_neighbors)
    model.fit(X_train_s, y_semi)
    preds = model.predict(X_val_s)
    return f1_score(val_labels, preds)


def make_labeled_subset(n_train, labels_train, fraction, seed):
    n_labeled = max(4, int(n_train * fraction))
    rng = np.random.RandomState(seed)
    positions = rng.choice(n_train, size=n_labeled, replace=False)
    mask = np.zeros(n_train, dtype=bool)
    mask[positions] = True
    if len(set(labels_train[mask])) < 2:
        return None
    return mask


def run_search(dataset_path, dataset_name):
    with open(dataset_path) as f:
        data = json.load(f)
    texts = np.array([row["text"] for row in data])
    labels = np.array([row["label"] for row in data])

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.4, stratify=labels, random_state=config["random_seed"]
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=config["random_seed"]
    )

    print(f"\n=== {dataset_name} joint search ===")
    print(f"Train: {len(train_texts)}  Val: {len(val_texts)}  Test (held out): {len(test_texts)}")

    results = []
    for window_size in WINDOW_SIZES:
        for third_rank in THIRD_MODE_RANKS:
            vocab = build_vocabulary(list(train_texts), max_vocab_size=config["max_vocab_size"])
            all_texts = list(train_texts) + list(val_texts)
            tensor = build_dataset_tensor(all_texts, vocab, window_size=window_size)
            rank = [FIXED_VOCAB_RANK, FIXED_VOCAB_RANK, min(third_rank, tensor.shape[2] - 1)]
            core, factors = decompose_tucker(tensor, rank=rank)
            embeddings = factors[2]
            n_train = len(train_texts)

            subset_mask = make_labeled_subset(n_train, train_labels, LOW_LABEL_FRACTION, config["random_seed"])
            if subset_mask is None:
                continue

            lr_f1 = eval_logistic_regression(embeddings, train_labels, subset_mask, val_labels, n_train)
            ls_f1 = eval_label_spreading(embeddings, train_labels, subset_mask, val_labels, n_train)

            results.append({
                "window_size": window_size, "third_rank": third_rank,
                "lr_val_f1": lr_f1, "labelspread_val_f1": ls_f1,
            })
            print(f"  window={window_size}, rank_3={third_rank}: LR={lr_f1:.3f}  LabelSpread={ls_f1:.3f}")

    best_lr = max(results, key=lambda r: r["lr_val_f1"])
    best_ls = max(results, key=lambda r: r["labelspread_val_f1"])
    overall_best = best_lr if best_lr["lr_val_f1"] >= best_ls["labelspread_val_f1"] else best_ls
    overall_best_classifier = "lr" if overall_best is best_lr else "labelspread"

    print(f"\nBest LR config:          window={best_lr['window_size']}, rank_3={best_lr['third_rank']}, F1={best_lr['lr_val_f1']:.3f}")
    print(f"Best LabelSpread config: window={best_ls['window_size']}, rank_3={best_ls['third_rank']}, F1={best_ls['labelspread_val_f1']:.3f}")
    print(f"Overall winner: {overall_best_classifier.upper()}, window={overall_best['window_size']}, rank_3={overall_best['third_rank']}")

    return {
        "dataset": dataset_name,
        "search_results": results,
        "best_lr_config": best_lr,
        "best_labelspread_config": best_ls,
        "overall_best_classifier": overall_best_classifier,
        "overall_best_config": {"window_size": overall_best["window_size"], "third_rank": overall_best["third_rank"]},
        "train_texts": list(train_texts), "train_labels": [int(l) for l in train_labels],
        "test_texts": list(test_texts), "test_labels": [int(l) for l in test_labels],
    }


ds1_search = run_search(ROOT / "datasets" / "prompt_injection" / "ds1_deepset_clean.json", "DS1")

out_dir = ROOT / "results"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "joint_search_ds1.json", "w") as f:
    json.dump(ds1_search, f, indent=2)

print("\nSaved to results/joint_search_ds1.json")