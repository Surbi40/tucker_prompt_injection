# src/baselines/embedding_drift.py
import numpy as np
from sentence_transformers import SentenceTransformer

_embedder = None

def get_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(model_name)
    return _embedder


def embedding_drift_score(benign_texts, query_texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    ZEDD-style zero-shot baseline: semantic drift from a benign-only
    embedding centroid, using a pretrained sentence embedder (no fine-tuning,
    no labels, no model-internals access).
    """
    embedder = get_embedder(model_name)
    benign_vecs = embedder.encode(list(benign_texts))
    query_vecs = embedder.encode(list(query_texts))

    centroid = benign_vecs.mean(axis=0)
    dists = np.linalg.norm(query_vecs - centroid, axis=1)
    return dists