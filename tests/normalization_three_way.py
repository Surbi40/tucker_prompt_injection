import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
from src.utils.config import get_project_root
from src.preprocessing.tensor_builder import build_vocab, build_tensor, compute_norm_constant
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
    const = compute_norm_constant(texts, method="median")
    print(f"Corpus median n_pairs (fixed norm constant): {const:.1f}")

    configs = [
        ("RAW", dict(normalize=False, norm_constant=None)),
        ("PER-PROMPT", dict(normalize=True, norm_constant=None)),
        ("FIXED-CONSTANT", dict(normalize=False, norm_constant=const)),
    ]

    for tag, kwargs in configs:
        tensor = build_tensor(texts, word2idx, window=5, **kwargs)
        embeddings, core, factors, elapsed = run_tucker(
            tensor, ranks=(10, 10, 10), init="random", tol=1e-4, n_iter_max=25
        )
        centroid = embeddings.mean(axis=0)
        all_dists = np.linalg.norm(embeddings - centroid, axis=1)
        length_corr = np.corrcoef(all_dists, lengths)[0, 1]

        ranked, dists = lexical_centrality_ranking(embeddings, texts=texts, top_k=10)
        n_injection = sum(labels[r["index"]] for r in ranked)
        base_rate = labels.mean()
        avg_len_top10 = np.mean([lengths[r["index"]] for r in ranked])

        print(f"\n[{tag}] time={elapsed:.2f}s")
        if tag == "FIXED-CONSTANT":
            print(f"  Using fixed norm constant: {const:.1f}")
        print(f"  Top-10 centroid-distant: {n_injection}/10 (base rate: {base_rate:.2f})")
        print(f"  Correlation(distance, length): r={length_corr:.3f}")
        print(f"  Avg length of top-10 anomalies: {avg_len_top10:.1f} chars")
