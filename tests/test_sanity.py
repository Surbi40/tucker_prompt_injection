"""
tests/test_sanity.py

Run: 
    $env:PYTHONPATH="."
    python tests/test_sanity.py

All 7 tests must pass before generating the real 200-sample dataset.
"""

import sys
import numpy as np

from src.preprocessing.tensor_builder import (
    tokenize,
    build_vocab,          # <-- updated to match your tensor_builder.py
    cooccurrence_tensor,
)
from src.decomposition.tucker_decomposer import (
    decompose_tucker,
    extract_tucker_features,
    reconstruction_error,
)
from src.utils.config import load_config

config = load_config()

PASS = "PASS"
FAIL = "FAIL"
results = []


def report(test_name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((test_name, status))
    print(f"[{status}] {test_name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------
# Test 1: Tensor shape is exactly (V, V, window_size)
# ---------------------------------------------------------------------
def test_tensor_shape():
    text = "the cat sat on the mat"
    vocab = build_vocab([text], max_vocab_size=20)
    window_size = 3
    tensor = cooccurrence_tensor(text, vocab, window_size=window_size)

    V = len(vocab)
    expected_shape = (V, V, window_size)
    report(
        "1. Tensor shape correctness",
        tensor.shape == expected_shape,
        f"expected {expected_shape}, got {tensor.shape}",
    )


# ---------------------------------------------------------------------
# Test 2: Known word pair at known distance produces expected count
# ---------------------------------------------------------------------
def test_known_pair_count():
    text = "the cat sat on the mat"
    vocab = build_vocab([text], max_vocab_size=20)
    tensor = cooccurrence_tensor(text, vocab, window_size=3)

    # "cat" is 1 position before "sat" -> tensor[cat, sat, 0] should be 1
    cat_id = vocab["cat"]
    sat_id = vocab["sat"]
    count = tensor[cat_id, sat_id, 0]

    report(
        "2. Known word-pair distance count",
        count == 1,
        f"expected tensor[cat,sat,0]=1, got {count}",
    )


# ---------------------------------------------------------------------
# Test 3: Directionality — reverse pair at same offset should NOT match
# ---------------------------------------------------------------------
def test_directionality():
    text = "the cat sat on the mat"
    vocab = build_vocab([text], max_vocab_size=20)
    tensor = cooccurrence_tensor(text, vocab, window_size=3)

    cat_id = vocab["cat"]
    sat_id = vocab["sat"]
    reverse_count = tensor[sat_id, cat_id, 0]

    report(
        "3. Directional (non-symmetric) counting",
        reverse_count == 0,
        f"expected tensor[sat,cat,0]=0 (directional), got {reverse_count}",
    )


# ---------------------------------------------------------------------
# Test 4: Tucker reconstruction error is small on a synthetic tensor
# ---------------------------------------------------------------------
def test_tucker_reconstruction_error():
    np.random.seed(42)
    synthetic_tensor = np.random.rand(10, 10, 5).astype(np.float32)
    rank = config["tucker_rank"]  # e.g. [10, 10, 3]

    core, factors = decompose_tucker(synthetic_tensor, rank=rank)
    error = reconstruction_error(synthetic_tensor, core, factors)

    report(
        "4. Tucker reconstruction error is bounded",
        error < 0.5,
        f"reconstruction error = {error:.4f}",
    )


# ---------------------------------------------------------------------
# Test 5: Feature vector length matches documented formula
# ---------------------------------------------------------------------
def test_feature_vector_length():
    V = 20
    rank = [10, 10, 3]
    np.random.seed(0)
    synthetic_tensor = np.random.rand(V, V, 5).astype(np.float32)

    core, factors = decompose_tucker(synthetic_tensor, rank=rank)
    feature = extract_tucker_features(core, factors)

    R1, R2, R3 = rank
    expected_len = (R1 * R2 * R3) + (V * R1) + (V * R2) + (5 * R3)

    report(
        "5. Feature vector length matches formula",
        len(feature) == expected_len,
        f"expected {expected_len}, got {len(feature)}",
    )


# ---------------------------------------------------------------------
# Test 6: Vocabulary built ONLY from training split (no leakage)
# ---------------------------------------------------------------------
def test_no_vocabulary_leakage():
    train_texts = ["the cat sat on the mat"]
    test_only_word = "elephant"
    test_texts = [f"an {test_only_word} walked by"]

    vocab = build_vocab(train_texts, max_vocab_size=50)

    report(
        "6. No vocabulary leakage from test split",
        test_only_word not in vocab,
        f"'{test_only_word}' should NOT be in train-only vocab, "
        f"found={test_only_word in vocab}",
    )


# ---------------------------------------------------------------------
# Test 7: Full pipeline runs end-to-end on dummy data without crashing
# ---------------------------------------------------------------------
def test_end_to_end_pipeline():
    try:
        texts = [
            "the cat sat on the mat",
            "a dog ran in the park",
            "the mat was on the floor",
        ]
        vocab = build_vocab(texts, max_vocab_size=config["max_vocab_size"])

        tensors = [
            cooccurrence_tensor(t, vocab, window_size=config["window_size"])
            for t in texts
        ]

        features = []
        for t in tensors:
            core, factors = decompose_tucker(t, rank=config["tucker_rank"])
            feat = extract_tucker_features(core, factors)
            features.append(feat)

        all_same_length = len(set(len(f) for f in features)) == 1

        report(
            "7. End-to-end pipeline runs without crashing",
            all_same_length,
            f"produced {len(features)} feature vectors, "
            f"consistent length={all_same_length}",
        )
    except Exception as e:
        report("7. End-to-end pipeline runs without crashing", False, f"raised {type(e).__name__}: {e}")


# ---------------------------------------------------------------------
def main():
    print("Running sanity tests...\n")

    test_tensor_shape()
    test_known_pair_count()
    test_directionality()
    test_tucker_reconstruction_error()
    test_feature_vector_length()
    test_no_vocabulary_leakage()
    test_end_to_end_pipeline()

    print("\n--- Summary ---")
    passed = sum(1 for _, s in results if s == PASS)
    total = len(results)
    print(f"{passed}/{total} tests passed.")

    if passed < total:
        print("\nDo NOT proceed to dataset generation until all tests pass.")
        sys.exit(1)
    else:
        print("\nAll sanity checks passed. Safe to proceed to Step 4 (dataset generation).")


if __name__ == "__main__":
    main()
