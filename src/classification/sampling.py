# src/classification/sampling.py
import numpy as np


def stratified_labeled_mask(labels, pct, seed=42, min_per_class=1):
    """
    Select a labeled subset at the given percentage, guaranteed at least
    min_per_class examples per class — critical at 0.5%-1% labeling rates
    where naive random sampling can produce zero minority-class examples.
    """
    rng = np.random.RandomState(seed)
    labels = np.array(labels)
    mask = np.zeros(len(labels), dtype=bool)
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        n_select = max(min_per_class, int(round(len(cls_idx) * pct)))
        n_select = min(n_select, len(cls_idx))
        selected = rng.choice(cls_idx, size=n_select, replace=False)
        mask[selected] = True
    return mask