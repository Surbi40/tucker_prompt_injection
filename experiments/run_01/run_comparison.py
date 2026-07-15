import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

from src.preprocessing.tensor_builder import build_vocabulary, build_dataset_tensor
from src.decomposition.tucker_decomposer import decompose_tucker
from src.decomposition.cp_decomposer import decompose_cp
from src.utils.config import load_config, get_project_root, set_seed

config = load_config()
set_seed(config["random_seed"])
ROOT = get_project_root()

with open(ROOT / "datasets" / "prompt_injection" / "ds1_deepset_clean.json") as f:
    data = json.load(f)

texts = np.array([row["text"] for row in data])
labels = np.array([row["label"] for row in data])

tucker_rank = [30, 30, 15]
cp_rank = 15  

skf = StratifiedKFold(n_splits=config["n_folds"], shuffle=True, random_state=config["random_seed"])

results = {"tucker": [], "cp": []}

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
    train_texts, test_texts = texts[train_idx], texts[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]

    vocab = build_vocabulary(list(train_texts), max_vocab_size=config["max_vocab_size"])
    all_texts = list(train_texts) + list(test_texts)
    tensor = build_dataset_tensor(all_texts, vocab, window_size=config["window_size"])

    def evaluate(embeddings, method_name):
        X_train = embeddings[:len(train_texts)]
        X_test = embeddings[len(train_texts):]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_train_s, train_labels)
        preds = clf.predict(X_test_s)
        f1 = f1_score(test_labels, preds)
        acc = accuracy_score(test_labels, preds)
        print(f"  [{method_name}] Fold {fold_idx}: F1={f1:.3f}  Accuracy={acc:.3f}")
        return {"fold": fold_idx, "f1": f1, "accuracy": acc}

    # Tucker
    fold_tucker_rank = [tucker_rank[0], tucker_rank[1], min(tucker_rank[2], tensor.shape[2] - 1)]
    core, factors = decompose_tucker(tensor, rank=fold_tucker_rank)
    results["tucker"].append(evaluate(factors[2], "Tucker"))

    # CP
    fold_cp_rank = min(cp_rank, tensor.shape[2] - 1)
    weights, cp_factors = decompose_cp(tensor, rank=fold_cp_rank)
    results["cp"].append(evaluate(cp_factors[2], "CP"))

print("\n--- Summary ---")
for method in ["tucker", "cp"]:
    f1s = [r["f1"] for r in results[method]]
    print(f"{method.upper()}: Mean F1 = {np.mean(f1s):.3f} (+/- {np.std(f1s):.3f})")

out_dir = ROOT / "results"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "tucker_vs_cp_ds1.json", "w") as f:
    json.dump(results, f, indent=2)