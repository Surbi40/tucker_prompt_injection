import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
from collections import Counter

from src.utils.config import get_project_root
from src.preprocessing.tensor_builder import build_vocab, build_tensor, tokenize
from src.decomposition.tucker_decomposer import run_tucker, lexical_centrality_ranking

ROOT = get_project_root()
out_dir = ROOT / "datasets" / "prompt_injection"

# Define datasets and vocab sizes (tune vocab per dataset for memory)
datasets = {
    "DS1": (out_dir / "ds1_deepset_clean.json", 500),
    "DS2": (out_dir / "ds2_wambosec_clean.json", 300),   # smaller vocab for memory
    "DS3": (out_dir / "ds3_slabs_experiment.json", 500),
    "DS4": (out_dir / "ds4_geekyrakshit_experiment.json", 500),
    "Combined": (out_dir / "combined_clean.json", 500),
}

for name, (path, vocab_size) in datasets.items():
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]
    sources = [d.get("source") for d in data]  # only present in Combined

    # Tokenizer coverage check
    empty_after_tok = sum(1 for t in texts if len(tokenize(t)) < 2)
    print(f"Rows: {len(texts)}  |  Empty/near-empty after tokenization: {empty_after_tok} "
          f"({empty_after_tok/len(texts)*100:.1f}%)")

    # Build vocab + tensor
    word2idx = build_vocab(texts, vocab_size=vocab_size, min_freq=2)
    print(f"Vocab size (incl <UNK>): {len(word2idx)}")

    t0 = time.perf_counter()
    tensor = build_tensor(texts, word2idx, window=5)
    print(f"Tensor shape: {tensor.shape}, build: {time.perf_counter()-t0:.1f}s, "
          f"memory: {tensor.nbytes/1e6:.1f} MB")

    # Tucker decomposition (fast config)
    embeddings, core, factors, elapsed = run_tucker(
        tensor, ranks=(10, 10, 10), init="random", tol=1e-4, n_iter_max=25
    )
    print(f"Tucker time: {elapsed:.2f}s")

    # Anomaly ranking
    ranked, dists = lexical_centrality_ranking(embeddings, texts=texts, top_k=10)
    n_injection = sum(labels[r["index"]] for r in ranked)
    base_rate = sum(labels) / len(labels)
    print(f"Top-10 centroid-distant: {n_injection}/10 are true injection "
          f"(base rate: {base_rate:.2f})")

    # Diagnostic: show labels + text for top-10 anomalies
    print(f"\n{name} top-10 centroid-distant prompts:")
    for r in ranked:
        lbl = labels[r["index"]]
        src = sources[r["index"]] if sources[0] is not None else ""
        print(f"  dist={r['distance']:.3f} label={lbl} {src} text={r['text'][:80]!r}")

    # Source confound check for Combined dataset
    if name == "Combined" and sources[0] is not None:
        top_sources = [sources[r["index"]] for r in ranked]
        print(f"Source breakdown of top-10 distant prompts: {Counter(top_sources)}")
