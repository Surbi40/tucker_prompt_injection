import json
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from src.utils.config import load_config, get_project_root

config = load_config()
ROOT = get_project_root()

def run_baselines(dataset_path, subsample_size=None, n_repeats=3, seeds=(42, 123, 7)):
    with open(dataset_path) as f:
        data = json.load(f)
    texts = np.array([row["text"] for row in data])
    labels = np.array([row["label"] for row in data])

    if subsample_size and len(texts) > subsample_size:
        texts, _, labels, _ = train_test_split(
            texts, labels, train_size=subsample_size,
            stratify=labels, random_state=config["random_seed"]
        )

    results = {"svm": [], "rf": []}

    for seed in seeds:
        skf = StratifiedKFold(n_splits=config["n_folds"], shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(texts, labels):
            train_texts, test_texts = texts[train_idx], texts[test_idx]
            train_labels, test_labels = labels[train_idx], labels[test_idx]

            vectorizer = TfidfVectorizer(max_features=config["max_vocab_size"])
            X_train = vectorizer.fit_transform(train_texts)
            X_test = vectorizer.transform(test_texts)

            svm = SVC(class_weight="balanced")
            svm.fit(X_train, train_labels)
            results["svm"].append(f1_score(test_labels, svm.predict(X_test)))

            rf = RandomForestClassifier(class_weight="balanced", random_state=config["random_seed"])
            rf.fit(X_train, train_labels)
            results["rf"].append(f1_score(test_labels, rf.predict(X_test)))

    for method in results:
        f1s = results[method]
        print(f"{method.upper()}: mean={np.mean(f1s):.3f}  std={np.std(f1s):.3f}")

    return results

print("=== DS1 (deepset) ===")
ds1_results = run_baselines(ROOT / "datasets" / "prompt_injection" / "ds1_deepset_clean.json")

print("\n=== DS2 (wambosec, subsampled) ===")
ds2_results = run_baselines(
    ROOT / "datasets" / "prompt_injection" / "ds2_wambosec_clean.json",
    subsample_size=1200
)

out_dir = ROOT / "results"
with open(out_dir / "baselines_results.json", "w") as f:
    json.dump({"ds1": ds1_results, "ds2": ds2_results}, f, indent=2)