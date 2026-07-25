import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf(texts_train, texts_test, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(texts_train)
    X_test = vectorizer.transform(texts_test)
    return X_train, X_test, vectorizer


def run_svm_baseline(X_train, y_train, X_test, class_weight="balanced"):
    start = time.perf_counter()
    clf = SVC(kernel="linear", class_weight=class_weight, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return preds, time.perf_counter() - start


def run_rf_baseline(X_train, y_train, X_test, class_weight="balanced", n_estimators=100):
    start = time.perf_counter()
    clf = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return preds, time.perf_counter() - start