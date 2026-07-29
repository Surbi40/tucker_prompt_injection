import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
import time

tl.set_backend("numpy")


def fit_tucker(train_tensor, ranks=(10, 10, 10), init="svd", tol=1e-6, n_iter_max=100, random_state=42):
    """
    Fit Tucker decomposition ONCE on the benign training tensor. Returns a
    dict containing everything needed to score new, individual prompts
    later without ever recomputing the decomposition:

        core, factors     -> the fitted Tucker model
        embeddings        -> factors[2], one embedding per training prompt
        projection_basis  -> precomputed linear map from a raw (N,N) prompt
                              matrix to mode-3 (prompt-space) coordinates,
                              used by project_prompt() below

    factors[2] is only a valid "prompt embedding" matrix if mode index 2 of
    train_tensor really is the prompt/document axis (i.e. tensor shape is
    (N, N, M) with M = number of prompts, as in tensor_builder.build_tensor).
    If you change the tensor layout, update which factor you treat as the
    prompt-mode factor accordingly.
    """
    start = time.perf_counter()
    core, factors = tucker(
        train_tensor, rank=list(ranks), init=init, tol=tol,
        n_iter_max=n_iter_max, random_state=random_state
    )
    elapsed = time.perf_counter() - start

    embeddings = factors[2]
    projection_basis = _build_projection_basis(core, factors)
    projection_matrix = np.linalg.pinv(projection_basis)
    centroid = embeddings.mean(axis = 0)

    return {
        "core": core,
        "factors": factors,
        "embeddings": embeddings,
        "projection_basis": projection_basis,
        "projection_matrix": projection_matrix,
        "centroid": centroid,
        "fit_time_sec": elapsed,
    }


def _build_projection_basis(core, factors):
    """
    Precompute an exact linear map M (shape: N*N x r3) such that, for a new
    prompt matrix P (shape N,N), the mode-3 (prompt-space) coordinates z
    satisfying   flatten(P) ~= M @ z   can be solved by least squares.

    Rather than assuming tensorly's specific mode-3 unfolding/Kronecker
    ordering convention (easy to get subtly wrong), we derive M empirically:
    reconstruct the tensor for each unit basis vector e_k in the r3-dim
    prompt space (with the frozen A=factors[0], B=factors[1]) and record
    the resulting (N,N) slice. Because Tucker reconstruction is linear in
    the mode-3 coordinate, this gives an exact basis regardless of internal
    unfolding order. r3 is small (e.g. 10), so this is cheap and only done
    once, at fit time.
    """
    A, B = factors[0], factors[1]
    assert core.shape[2] == factors[2].shape[1]
    r3 = core.shape[2]
    N = A.shape[0]
    M = np.zeros((N * N, r3), dtype=np.float32)
    for k in range(r3):
        e_k = np.zeros((1, r3), dtype=np.float32)
        e_k[0, k] = 1.0
        recon_slice = tl.tucker_to_tensor((core, [A, B, e_k]))[:, :, 0]
        M[:, k] = recon_slice.flatten()
    return M


def project_prompt(prompt_matrix, model):
    """
    Project one unseen prompt into the frozen Tucker latent space.
    """

    assert prompt_matrix.ndim == 2, "Prompt matrix must be 2D."
    assert prompt_matrix.shape[0] == prompt_matrix.shape[1], \
        "Prompt matrix must be square."

    p_flat = np.asarray(prompt_matrix, dtype=np.float32).ravel()

    z = model["projection_matrix"] @ p_flat

    return z


def run_tucker(
    tensor,
    ranks=(10, 10, 10),
    init="svd",
    tol=1e-6,
    n_iter_max=100,
    random_state=42,
):
    """
    Backward-compatible wrapper around fit_tucker().

    This function performs a one-time Tucker decomposition on the training
    tensor and returns the complete fitted model dictionary. New code should
    prefer calling fit_tucker() directly.
    """
    return fit_tucker(
        train_tensor=tensor,
        ranks=ranks,
        init=init,
        tol=tol,
        n_iter_max=n_iter_max,
        random_state=random_state,
    )

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


def score_by_centroid_distance(model, query_embeddings):
    """
    Compute anomaly scores for one or more query embeddings using the
    centroid of the benign training embeddings.

    Parameters
    ----------
    model : dict
        Dictionary returned by fit_tucker(). Must contain:
            - "centroid": ndarray of shape (r3,)
    query_embeddings : ndarray
        Query embedding(s) produced by project_prompt().
        Shape can be:
            (r3,)          -> single prompt
            (n_queries,r3) -> multiple prompts

    Returns
    -------
    ndarray
        Euclidean distance(s) from the benign centroid.
        Larger distance = more anomalous.
    """

    centroid = model["centroid"]

    # Allow passing either one embedding or many
    query_embeddings = np.atleast_2d(query_embeddings)

    dists = np.linalg.norm(query_embeddings - centroid, axis=1)

    return dists

if __name__ == "__main__":
    # Minimal smoke test: fit on random benign-like tensor, then project a
    # brand-new single prompt slice without ever touching the training
    # tensor again.
    rng = np.random.default_rng(0)
    N, M = 50, 30
    train_tensor = rng.random((N, N, M)).astype(np.float32)
    train_tensor = (train_tensor + train_tensor.transpose(1, 0, 2)) / 2  # symmetric slices

    model = fit_tucker(train_tensor, ranks=(10, 10, 10))
    print("Train embeddings shape:", model["embeddings"].shape)
    err = reconstruction_error(train_tensor, model["core"], model["factors"])
    print(f"Reconstruction error: {err:.4f}")

    new_prompt_matrix = rng.random((N, N)).astype(np.float32)
    new_prompt_matrix = (new_prompt_matrix + new_prompt_matrix.T) / 2
    z = project_prompt(new_prompt_matrix, model)
    print("Projected new-prompt embedding:", z)

    centroid_dist = score_by_centroid_distance(model, z)
    print("Centroid distance for new prompt:", centroid_dist)