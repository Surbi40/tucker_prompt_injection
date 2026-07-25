import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier


def run_logreg_lasso_baseline(embeddings_train, labels_train, embeddings_test,
                               C=1.0, class_weight="balanced"):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(embeddings_train)
    X_test_scaled = scaler.transform(embeddings_test)

    start = time.perf_counter()
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                              class_weight=class_weight, max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, labels_train)
    preds = clf.predict(X_test_scaled)
    elapsed = time.perf_counter() - start
    return preds, elapsed, clf


def self_training_logreg_lasso(embeddings_all, labels, labeled_mask,
                                C=1.0, threshold=0.75, class_weight="balanced", max_iter=20):
    scaler = StandardScaler()
    # fit scaler on labeled points only, to avoid leaking unlabeled-set statistics
    scaler.fit(embeddings_all[labeled_mask])
    embeddings_scaled = scaler.transform(embeddings_all)

    y_input = np.full(len(labels), -1, dtype=int)
    y_input[labeled_mask] = np.array(labels)[labeled_mask]

    base_clf = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                   class_weight=class_weight, max_iter=1000, random_state=42)
    self_training_clf = SelfTrainingClassifier(base_clf, threshold=threshold, max_iter=max_iter)

    start = time.perf_counter()
    self_training_clf.fit(embeddings_scaled, y_input)
    elapsed = time.perf_counter() - start

    preds_all = self_training_clf.predict(embeddings_scaled)
    return preds_all, elapsed, self_training_clf