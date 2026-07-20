import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import json
import time
import numpy as np
import re
from langdetect import detect, LangDetectException

from src.utils.config import get_project_root
from src.preprocessing.tensor_builder import build_vocab, build_tensor
from src.decomposition.tucker_decomposer import run_tucker, reconstruction_error, lexical_centrality_ranking

ROOT = get_project_root()

# --- Load dataset (DS1 for smoke test) ---
with open(ROOT / "datasets" / "prompt_injection" / "ds1_deepset_clean.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [d["text"] for d in data]
labels = [d["label"] for d in data]
labels_arr = np.array(labels)

# --- Language detection check (using langdetect) ---
def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return True  # too short/ambiguous, keep it

non_english_flags = [not is_english(t) for t in texts]
non_english_arr = np.array(non_english_flags)

print(f"Injection rate among non-English-looking prompts: {labels_arr[non_english_arr].mean():.2f} (n={non_english_arr.sum()})")
print(f"Injection rate among English prompts: {labels_arr[~non_english_arr].mean():.2f} (n={(~non_english_arr).sum()})")

# --- Optional: filter to English-only subset ---
texts_english = [t for t, flag in zip(texts, non_english_flags) if not flag]
labels_english = [l for l, flag in zip(labels, non_english_flags) if not flag]

# Toggle here: set use_english_only = True to run on filtered subset
use_english_only = True
if use_english_only:
    texts, labels = texts_english, labels_english
    print(f"Filtered to English-only subset: {len(texts)} prompts")

# --- Build vocab + tensor ---
word2idx = build_vocab(texts, vocab_size=500, min_freq=2)
print(f"Vocab size (incl <UNK>): {len(word2idx)}")

t0 = time.perf_counter()
tensor = build_tensor(texts, word2idx, window=5)
print(f"Tensor shape: {tensor.shape}, build time: {time.perf_counter()-t0:.1f}s, "
      f"memory: {tensor.nbytes/1e6:.1f} MB")

# --- Fast-config anomaly check ---
embeddings, core, factors, elapsed = run_tucker(
    tensor, ranks=(10, 10, 10), init="random", tol=1e-4, n_iter_max=25
)
print(f"Fast-config decomposition time: {elapsed:.2f}s")

ranked, dists = lexical_centrality_ranking(embeddings, texts=texts, top_k=10)
n_injection = sum(labels[r["index"]] for r in ranked)
print(f"{n_injection}/10 top-ranked prompts are true injection examples "
      f"(base rate: {sum(labels)/len(labels):.2f})")

# --- Timing grid experiment ---
import itertools
print("\n--- Timing grid ---")
for rank_val, tol, n_iter, init in itertools.product(
    [10, 30], [1e-6, 1e-4], [100, 25], ["svd", "random"]
):
    embeddings, core, factors, elapsed = run_tucker(
        tensor, ranks=(rank_val, rank_val, rank_val),
        init=init, tol=tol, n_iter_max=n_iter
    )
    err = reconstruction_error(tensor, core, factors)
    print(f"rank={rank_val:3d} tol={tol:.0e} n_iter={n_iter:3d} init={init:7s}: "
          f"{elapsed:6.1f}s  error={err:.4f}")

# --- Tucker decomposition (baseline run) ---
embeddings, core, factors, elapsed = run_tucker(tensor, ranks=(10, 10, 10))
print(f"Tucker decomposition time: {elapsed:.2f}s")
print(f"Embeddings shape: {embeddings.shape}")

# --- Reconstruction error ---
err = reconstruction_error(tensor, core, factors)
print(f"Relative reconstruction error: {err:.4f}")

# --- Ranking by lexical centrality ---
ranked, dists = lexical_centrality_ranking(embeddings, texts=texts, top_k=10)
print("\nTop 10 most centroid-distant prompts:")
n_injection = 0
for r in ranked:
    true_label = labels[r["index"]]
    n_injection += true_label
    print(f"  dist={r['distance']:.3f}  label={true_label}  text={r['text'][:70]!r}")

print(f"\n{n_injection}/10 top-ranked prompts are true injection examples "
      f"(base rate: {sum(labels)/len(labels):.2f})")
