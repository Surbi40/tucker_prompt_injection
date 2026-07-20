import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
from src.utils.config import get_project_root
from src.preprocessing.tensor_builder import (
    build_vocab,
    build_tensor,
    prompt_cooccurrence_matrix,
    tokenize,
    compute_norm_constant,
)
from src.decomposition.tucker_decomposer import run_tucker, lexical_centrality_ranking

ROOT = get_project_root()
out_dir = ROOT / "datasets" / "prompt_injection"

datasets = {
    "DS1": (out_dir / "ds1_deepset_clean.json", 500),
    "DS3": (out_dir / "ds3_slabs_experiment.json", 500),
}

for name, (path, vocab_size) in datasets.items():
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    texts = [d["text"] for d in data]
    labels = np.array([d["label"] for d in data])
    lengths = np.array([len(t) for t in texts])

    word2idx = build_vocab(texts, vocab_size=vocab_size, min_freq=2)

    # --- Part A: per-prompt magnitude check ---
    order_by_len = np.argsort(lengths)
    short_idx = order_by_len[5]   # one short prompt
    long_idx = order_by_len[-5]   # one long prompt
    print("\n--- Per-prompt magnitude check ---")
    for i, tag in [(short_idx, "SHORT"), (long_idx, "LONG")]:
        text = texts[i]
        n_tok = len(tokenize(text))
        mat_raw = prompt_cooccurrence_matrix(text, word2idx, window=5, normalize=False)
        mat_norm = prompt_cooccurrence_matrix(text, word2idx, window=5, normalize=True)
        print(f"  [{tag}] len={len(text):4d} chars, {n_tok:3d} tokens, label={labels[i]} | "
              f"raw sum={mat_raw.sum():.2f}  norm sum={mat_norm.sum():.4f}")

    # --- Part B: full pipeline ---
    # RAW counts (default for main pipeline)
    tensor_raw = build_tensor(texts, word2idx, window=5, normalize=False, norm_constant=None)
    embeddings, core, factors, elapsed = run_tucker(
        tensor_raw, ranks=(10, 10, 10), init="random", tol=1e-4, n_iter_max=25
    )
    centroid = embeddings.mean(axis=0)
    all_dists = np.linalg.norm(embeddings - centroid, axis=1)
    length_corr = np.corrcoef(all_dists, lengths)[0, 1]
    ranked, dists = lexical_centrality_ranking(embeddings, texts=texts, top_k=10)
    n_injection = sum(labels[r["index"]] for r in ranked)
    base_rate = labels.mean()
    print(f"\n[RAW] Tucker time={elapsed:.2f}s")
    print(f"  Top-10 centroid-distant: {n_injection}/10 injections (base rate: {base_rate:.2f})")
    print(f"  Correlation(centroid-distance, prompt length): r={length_corr:.3f}")

    # PER-PROMPT normalization (kept for ablation evidence only)
    tensor_norm = build_tensor(texts, word2idx, window=5, normalize=True, norm_constant=None)
    embeddings, core, factors, elapsed = run_tucker(
        tensor_norm, ranks=(10, 10, 10), init="random", tol=1e-4, n_iter_max=25
    )
    centroid = embeddings.mean(axis=0)
    all_dists = np.linalg.norm(embeddings - centroid, axis=1)
    length_corr = np.corrcoef(all_dists, lengths)[0, 1]
    ranked, dists = lexical_centrality_ranking(embeddings, texts=texts, top_k=10)
    n_injection = sum(labels[r["index"]] for r in ranked)
    base_rate = labels.mean()
    print(f"\n[PER-PROMPT] Tucker time={elapsed:.2f}s")
    print(f"  Top-10 centroid-distant: {n_injection}/10 injections (base rate: {base_rate:.2f})")
    print(f"  Correlation(centroid-distance, prompt length): r={length_corr:.3f}")

    # FIXED-CONSTANT normalization (included for completeness)
    norm_const = compute_norm_constant(texts, window=5, method="median")
    tensor_fixed = build_tensor(texts, word2idx, window=5, normalize=False, norm_constant=norm_const)
    embeddings, core, factors, elapsed = run_tucker(
        tensor_fixed, ranks=(10, 10, 10), init="random", tol=1e-4, n_iter_max=25
    )
    centroid = embeddings.mean(axis=0)
    all_dists = np.linalg.norm(embeddings - centroid, axis=1)
    length_corr = np.corrcoef(all_dists, lengths)[0, 1]
    ranked, dists = lexical_centrality_ranking(embeddings, texts=texts, top_k=10)
    n_injection = sum(labels[r["index"]] for r in ranked)
    base_rate = labels.mean()
    print(f"\n[FIXED-CONSTANT] Tucker time={elapsed:.2f}s (const={norm_const:.1f})")
    print(f"  Top-10 centroid-distant: {n_injection}/10 injections (base rate: {base_rate:.2f})")
    print(f"  Correlation(centroid-distance, prompt length): r={length_corr:.3f}")
