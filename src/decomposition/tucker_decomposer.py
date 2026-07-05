import numpy as np 
import tensorly as tl
from tensorly.decomposition import tucker

tl.set_backend("numpy")


def decompose_tensor(tensor, rank):
    """
        rank: list, e.g. [10, 10, 3] — one rank per mode.
    Returns (core, factors) via HOOI (unsupervised).

    """
    core, factors =tucker(tensor, rank = rank, init="svd",random_state=42)
    return core, factors

def extract_tucker_features(core, factors):
    """ flattens core tensor+ all factor matrices into one feature vector"""
    parts =[core.flatten()]+[f.flatten() for f in factors]
    return np.concatenate(parts)

def reconstruction_error(tensor, core, factors):
    """sanity check metric: relative frobenius norm reconstruction error"""
    reconstructed = tl. tucker_to_tensor((core, factors))
    return np.linalg.norm(tensor - reconstructed)/ np.linalg.norm(tensor)