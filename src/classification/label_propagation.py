import numpy as np
from sklearn.semi_supervised import LabelSpreading
import time


def knn_label_propagation(embeddings, labels, labeled_mask, n_neighbors=7,
                           kernel="knn", alpha=0.2, max_iter=1000, gamma=20):
    y_input = np.full(len(labels), -1, dtype=int)
    y_input[labeled_mask] = np.array(labels)[labeled_mask]

    if kernel == "knn":
        model = LabelSpreading(kernel=kernel, n_neighbors=n_neighbors, alpha=alpha, max_iter=max_iter)
    else:
        model = LabelSpreading(kernel=kernel, gamma=gamma, alpha=alpha, max_iter=max_iter)

    start = time.perf_counter()
    model.fit(embeddings, y_input)
    elapsed = time.perf_counter() - start
    return model.transduction_, elapsed, model