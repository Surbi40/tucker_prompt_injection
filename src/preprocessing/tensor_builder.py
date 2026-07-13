import numpy as np
from collections import Counter


def tokenize(text):
    return text.lower().split()


def build_vocabulary(texts, max_vocab_size=500):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    most_common = counter.most_common(max_vocab_size - 1)
    vocab = {word: idx for idx, (word, _) in enumerate(most_common)}
    vocab["<UNK>"] = len(vocab)
    return vocab


def prompt_cooccurrence_matrix(text, vocab, window_size=5):
    """
    Builds a single (V, V) co-occurrence matrix for ONE prompt,
    collapsing across all window distances (unlike your old per-sample
    3-mode tensor). This matches CoCoTen's matrix-per-prompt design.
    """
    tokens = tokenize(text)
    V = len(vocab)
    matrix = np.zeros((V, V), dtype=np.float32)

    token_ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    n = len(token_ids)

    for i in range(n):
        word_i = token_ids[i]
        for k in range(1, window_size + 1):
            j = i + k
            if j >= n:
                break
            word_j = token_ids[j]
            matrix[word_i, word_j] += 1

    return matrix


def build_dataset_tensor(texts, vocab, window_size=5):
    """
    Builds the FULL dataset tensor: shape (V, V, M) where M = number of prompts.
    This is the CoCoTen-style construction — one tensor for the whole dataset,
    not one tensor per sample.
    """
    matrices = [prompt_cooccurrence_matrix(t, vocab, window_size) for t in texts]
    tensor = np.stack(matrices, axis=2)  # stack along 3rd mode -> (V, V, M)
    return tensor