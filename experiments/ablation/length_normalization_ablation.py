# scripts/raw_tensor_experiment.py
import sys, json
from pathlib import Path

# Go up to project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils.config import get_project_root
from src.preprocessing.tensor_builder import build_vocab, build_tensor
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
    labels = [d["label"] for d in data]

    # Build vocab
    word2idx = build_vocab(texts, vocab_size=vocab_size, min_freq=2)

    # Always use raw counts (normalize=False, norm_constant=None)
    tensor = build_tensor(texts, word2idx, window=5, normalize=False, norm_constant=None)

    # Tucker decomposition
    embeddings, core, factors, elapsed = run_tucker(
        tensor, ranks=(10, 10, 10), init="random", tol=1e-4, n_iter_max=25
    )

    # Ranking anomalies
    ranked, dists = lexical_centrality_ranking(embeddings, texts=texts, top_k=10)
    n_injection = sum(labels[r["index"]] for r in ranked)
    base_rate = sum(labels) / len(labels)

    print(f"Tucker time={elapsed:.2f}s")
    print(f"Top-10 centroid-distant: {n_injection}/10 injections "
          f"(base rate: {base_rate:.2f})")
