import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

tl.set_backend("numpy")

def decompose_cp(tensor, rank):
    """rank : single, integer-CP uses the same rank across all nodes.
    Returns (weights,factors) via ALS"""

    weights, factors = parafac(tensor, rank = rank, init="svd",random_state=42)
    return weights, factors


def cp_reconstruction_error(tensor, weights, factors):
    reconstructed = tl.cp_to_tensor((weights, factors))
    return np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)