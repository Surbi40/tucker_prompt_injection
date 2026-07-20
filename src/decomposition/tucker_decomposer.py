import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
import time

tl.set_backend("numpy")


def run_tucker(tensor, ranks=(10, 10, 10), init="svd", tol=1e-6, n_iter_max=100, random_state=42):
    start = time.perf_counter()
    core, factors = tucker(
        tensor, rank=list(ranks), init=init, tol=tol,
        n_iter_max=n_iter_max, random_state=random_state
    )
    elapsed = time.perf_counter() - start
    embeddings = factors[2]
    return embeddings, core, factors, elapsed



def reconstruction_error(tensor, core, factors):
    """
    Relative Frobenius norm reconstruction error.
    Useful sanity check: should be small if ranks are adequate.
    """
    reconstructed = tl.tucker_to_tensor((core, factors))
    return np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)


def lexical_centrality_ranking(embeddings, texts=None, top_k=10):
    """
    Rank prompts by distance from the embedding centroid.
    Larger distance = more 'different' from the bulk of prompts = flagged
    as more likely adversarial.
    """
    centroid = embeddings.mean(axis=0)
    dists = np.linalg.norm(embeddings - centroid, axis=1)
    order = np.argsort(-dists)  # descending

    ranked = []
    for rank, idx in enumerate(order[:top_k]):
        entry = {"index": int(idx), "distance": float(dists[idx])}
        if texts is not None:
            entry["text"] = texts[idx]
        ranked.append(entry)
    return ranked, dists
