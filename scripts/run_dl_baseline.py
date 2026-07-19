import json
import numpy as np
from transformers import pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report
from src.utils.config import get_project_root

ROOT = get_project_root()

print("Loading pretrained prompt injection detector (no training needed)...")
detector = pipeline(
    "text-classification",
    model="protectai/deberta-v3-base-prompt-injection-v2"
)

def evaluate_zero_shot(dataset_path, dataset_name, subsample_size=None, seed=42):
    with open(dataset_path) as f:
        data = json.load(f)

    texts = [row["text"] for row in data]
    labels = [row["label"] for row in data]

    if subsample_size and len(texts) > subsample_size:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(texts), size=subsample_size, replace=False)
        texts = [texts[i] for i in idx]
        labels = [labels[i] for i in idx]

    print(f"\n=== {dataset_name} (n={len(texts)}) ===")

    preds = []
    for i, text in enumerate(texts):
        result = detector(text[:2000], truncation=True)[0]  # guard against extremely long inputs
        pred_label = 1 if result["label"].upper() in ("INJECTION", "MALICIOUS", "LABEL_1") else 0
        preds.append(pred_label)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(texts)}")

    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)

    print(f"\n{dataset_name} zero-shot DL baseline:")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  F1: {f1:.3f}")
    print(classification_report(labels, preds))

    return {"dataset": dataset_name, "n": len(texts), "accuracy": acc, "f1": f1}


results = {}

results["ds1"] = evaluate_zero_shot(
    ROOT / "datasets" / "prompt_injection" / "ds1_deepset_clean.json",
    "DS1 (deepset)"
)

results["ds2"] = evaluate_zero_shot(
    ROOT / "datasets" / "prompt_injection" / "ds2_wambosec_clean.json",
    "DS2 (wambosec)",
    subsample_size=1200
)

out_dir = ROOT / "results"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "dl_baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved to results/dl_baseline_results.json")