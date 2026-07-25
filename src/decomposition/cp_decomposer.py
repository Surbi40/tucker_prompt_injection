import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
import time

tl.set_backend("numpy")


def run_cp(tensor, rank=10, init="random", tol=1e-4, n_iter_max=25, random_state=42):
    """
    CANDECOMP/PARAFAC decomposition — the original CoCoTen paper's method.
    Unlike Tucker, CP takes a single rank (not one per mode) and produces
    three factor matrices of matching rank R, no core tensor.

    Returns:
        embeddings: (M, R) — mode-3 factor matrix, same role as Tucker's C.
        weights: CP component weights.
        factors: [A, B, C] factor matrices.
        elapsed: decomposition runtime in seconds.
    """
    start = time.perf_counter()
    weights, factors = parafac(
        tensor, rank=rank, init=init, tol=tol, n_iter_max=n_iter_max, random_state=random_state
    )
    elapsed = time.perf_counter() - start

    embeddings = factors[2]  # (M, R) prompt embeddings, same convention as Tucker
    return embeddings, weights, factors, elapsed


def cp_reconstruction_error(tensor, weights, factors):
    reconstructed = tl.cp_to_tensor((weights, factors))
    return np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)