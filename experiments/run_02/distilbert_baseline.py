import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


import json
import time
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

from src.utils.config import load_config, get_project_root, set_seed

config = load_config()
set_seed(config["random_seed"])
ROOT = get_project_root()

MODEL_NAME = "distilbert-base-uncased"
LABEL_FRACTIONS = [0.20, 0.10, 0.05, 0.02, 0.01, 0.005]
SEEDS = [42, 123, 7]
N_FOLDS = 5
device = "cuda" if torch.cuda.is_available() else "cpu"


def make_labeled_mask(n_train, labels_train, fraction, seed):
    n_labeled = max(4, int(n_train * fraction))
    rng = np.random.RandomState(seed)
    positions = rng.choice(n_train, size=n_labeled, replace=False)
    mask = np.zeros(n_train, dtype=bool)
    mask[positions] = True
    if len(set(labels_train[mask])) < 2:
        return None
    return mask
def fine_tune_and_eval(train_texts, train_labels, test_texts, test_labels, tokenizer):
    train_ds = Dataset.from_dict({"text": list(train_texts), "label": list(train_labels)})
    test_ds = Dataset.from_dict({"text": list(test_texts), "label": list(test_labels)})

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    train_ds = train_ds.map(tok, batched=True)
    test_ds = test_ds.map(tok, batched=True)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

    args = TrainingArguments(
        output_dir="/tmp/distilbert_lowlabel",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=16,
        logging_strategy="no",
        report_to="none",
        save_strategy="no",
    )

    start = time.perf_counter()
    trainer = Trainer(model=model, args=args, train_dataset=train_ds)
    trainer.train()
    fit_time = time.perf_counter() - start

    preds = trainer.predict(test_ds).predictions.argmax(axis=1)
    f1 = f1_score(test_labels, preds)

    del model, trainer
    torch.cuda.empty_cache()
    return f1, fit_time


def run_dataset(dataset_path, dataset_name):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = np.array([row["text"] for row in data])
    labels = np.array([row["label"] for row in data])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"\n{'='*70}\n{dataset_name} — DistilBERT low-label fine-tuning\n{'='*70}")

    results = {frac: {"distilbert": [], "distilbert_time": []} for frac in LABEL_FRACTIONS}

    fold_num = 0
    total_folds = len(SEEDS) * N_FOLDS

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(texts, labels):
            fold_num += 1
            train_texts, test_texts = texts[train_idx], texts[test_idx]
            train_labels, test_labels = labels[train_idx], labels[test_idx]
            n_train = len(train_texts)

            for frac in LABEL_FRACTIONS:
                mask = make_labeled_mask(n_train, train_labels, frac, seed)
                if mask is None:
                    continue

                f1, fit_time = fine_tune_and_eval(
                    train_texts[mask], train_labels[mask], test_texts, test_labels, tokenizer
                )
                results[frac]["distilbert"].append(f1)
                results[frac]["distilbert_time"].append(fit_time)

            print(f"  Fold {fold_num}/{total_folds} done")

    print(f"\n{'Label %':<10}{'DistilBERT F1':<16}{'Fine-tune time(s)'}")
    for frac in LABEL_FRACTIONS:
        r = results[frac]
        print(f"{frac*100:<10.1f}{np.mean(r['distilbert']):<16.3f}{np.mean(r['distilbert_time']):.2f}")

    return results

if __name__ == "__main__":
    out_dir = ROOT / "results" / "run_02"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds1_results = run_dataset(ROOT / "datasets" / "prompt_injection" / "ds1_deepset_clean.json", "DS1")
    with open(out_dir / "distilbert_ds1.json", "w") as f:
        json.dump({str(k): v for k, v in ds1_results.items()}, f, indent=2)