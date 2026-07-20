import numpy as np
import re
from collections import Counter
from scipy import sparse


def tokenize(text, lowercase=True):
    """Tokenize text into alphanumeric tokens."""
    if lowercase:
        text = text.lower()
    return re.findall(r"[a-zA-Z0-9']+", text)


def build_vocab(texts, vocab_size=500, min_freq=2, use_term_freq=False):
    """
    Build a shared vocabulary across all prompts.
    - vocab_size caps the size (controls tensor dimensions).
    - min_freq filters rare words.
    - use_term_freq=False -> document frequency (unique words per prompt).
    - use_term_freq=True -> raw term frequency (counts all tokens).
    """
    counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        if use_term_freq:
            counter.update(tokens)
        else:
            counter.update(set(tokens))  # doc frequency

    vocab_items = [(w, c) for w, c in counter.items() if c >= min_freq]
    vocab_items.sort(key=lambda x: -x[1])
    vocab_items = vocab_items[:vocab_size]

    word2idx = {w: i for i, (w, _) in enumerate(vocab_items)}
    word2idx["<UNK>"] = len(word2idx)  # fallback for OOV tokens
    return word2idx


def compute_norm_constant(texts, window=5, method="median"):
    """
    Corpus-wide normalization constant based on n_pairs (co-occurrence pair
    count) across all prompts. Used for fixed-constant normalization.
    """
    pair_counts = []
    for text in texts:
        tokens = tokenize(text)
        n_tok = len(tokens)
        if n_tok < 2:
            pair_counts.append(0)
            continue
        # mirrors the windowing logic in prompt_cooccurrence_matrix
        n_pairs = sum(min(window - 1, n_tok - 1 - i) for i in range(n_tok))
        pair_counts.append(n_pairs)
    pair_counts = np.array(pair_counts)
    if method == "median":
        const = np.median(pair_counts[pair_counts > 0])
    elif method == "mean":
        const = np.mean(pair_counts[pair_counts > 0])
    else:
        raise ValueError(f"unknown method {method}")
    return max(1.0, float(const))


def prompt_cooccurrence_matrix(
    text,
    word2idx,
    window=5,
    sparse_output=True,
    drop_oov=False,
    normalize=False,
    norm_constant=None,
):
    """
    Build a co-occurrence matrix for a single prompt.
    - normalize=True -> per-prompt normalization (divide by n_pairs).
    - norm_constant != None -> fixed-constant normalization (same divisor for all prompts).
    """
    N = len(word2idx)
    tokens = tokenize(text)

    if drop_oov:
        idxs = [word2idx[t] for t in tokens if t in word2idx and t != "<UNK>"]
    else:
        idxs = [word2idx.get(t, word2idx["<UNK>"]) for t in tokens]

    if len(idxs) < 2:
        return sparse.csr_matrix((N, N), dtype=np.float32) if sparse_output else np.zeros((N, N), dtype=np.float32)

    rows, cols, vals = [], [], []
    n_pairs = 0
    for i in range(len(idxs)):
        for j in range(i + 1, min(i + window, len(idxs))):
            a, b = idxs[i], idxs[j]
            rows += [a, b]
            cols += [b, a]
            vals += [1.0, 1.0]
            n_pairs += 1

    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N), dtype=np.float32)
    mat.sum_duplicates()

    if norm_constant is not None:
        mat = mat / norm_constant       # fixed-constant normalization
    elif normalize and n_pairs > 0:
        mat = mat / n_pairs             # per-prompt normalization

    return mat if sparse_output else mat.toarray()


def build_tensor(texts, word2idx, window=5, drop_oov=False, normalize=False, norm_constant=None):
    """
    Build the full dataset tensor: shape (N, N, M).
    Each slice [:,:,m] is the co-occurrence matrix for prompt m.
    """
    N = len(word2idx)
    M = len(texts)
    tensor = np.zeros((N, N, M), dtype=np.float32)

    for m, text in enumerate(texts):
        mat = prompt_cooccurrence_matrix(
            text,
            word2idx,
            window=window,
            sparse_output=True,
            drop_oov=drop_oov,
            normalize=normalize,
            norm_constant=norm_constant,
        )
        tensor[:, :, m] = mat.toarray()

    return tensor


if __name__ == "__main__":
    import json

    # Quick smoke test on DS1
    with open("datasets/prompt_injection/ds1_deepset_clean.json", "r", encoding="utf-8") as f:
        ds1 = json.load(f)

    texts = [ex["text"] for ex in ds1]
    labels = [ex["label"] for ex in ds1]

    # Build vocab + tensor
    word2idx = build_vocab(texts, vocab_size=500, min_freq=2)
    print("Vocab size (including <UNK>):", len(word2idx))

    tensor_raw = build_tensor(texts, word2idx, window=5, normalize=False)
    print("Tensor shape (raw counts):", tensor_raw.shape)

    tensor_per_prompt = build_tensor(texts, word2idx, window=5, normalize=True)
    print("Tensor shape (per-prompt normalized):", tensor_per_prompt.shape)

    norm_const = compute_norm_constant(texts, window=5, method="median")
    tensor_fixed = build_tensor(texts, word2idx, window=5, norm_constant=norm_const)
    print(f"Tensor shape (fixed-constant normalized, const={norm_const:.1f}):", tensor_fixed.shape)
