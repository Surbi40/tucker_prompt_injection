import numpy as np
import tensorly as tl


def project_new_prompt(cooc_matrix, core, A, B):
    """
    Project a single new prompt's (N, N) co-occurrence matrix into the
    embedding space of an already-fitted, benign-only Tucker model,
    WITHOUT re-decomposing the tensor. This is the actual zero-shot
    inference step a deployed system would use.
    """
    R1, R2, R3 = core.shape
    reduced = A.T @ cooc_matrix @ B  # (R1, R2)

    core_unfold = tl.unfold(core, mode=2)  # (R3, R1*R2)
    reduced_flat = reduced.reshape(-1, order="F")  # match tensorly's column-major unfold convention

    c_new, residuals, rank, sv = np.linalg.lstsq(core_unfold.T, reduced_flat, rcond=None)
    return c_new