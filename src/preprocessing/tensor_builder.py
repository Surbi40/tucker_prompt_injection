import numpy as np 
from collections import Counter

def tokenize(text):
    return text.lower().split()

def build_vocab(texts, max_vocab_size =500):
     """
    Build vocabulary from a list of texts.
    IMPORTANT: caller must pass TRAIN-SPLIT-ONLY texts when used inside
    cross-validation, to avoid vocabulary leakage from test data.
    """
     counter = Counter()
     for text in texts:
          counter.update(tokenize(text))

     most_common = counter.most_common(max_vocab_size - 1)  #reserve one token for <UNK>
     vocab = {word : idx for idx, (word, _) in enumerate(most_common)}
     vocab["<UNK>"] = len(vocab)
     return vocab

def cooccurrence_tensor(text, vocab , window_size = 5):
     """
    Builds a (V, V, window_size) tensor for ONE text sample.
    Entry (i, j, k) = count of word_i appearing k+1 positions before word_j.
    Directional: only counts forward co-occurrence (i occurs before j).
    """
     tokens = tokenize(text)
     V = len(vocab)
     tensor = np.zeros((V,V, window_size),dtype=np.float32)
     token_ids= [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
     n= len(token_ids)

     for i in range(n):
          word_i = token_ids[i]
          for k in range(1, window_size + 1):
               j = i + k
               if j >= n:
                    break
               word_j = token_ids[j]
               tensor[word_i, word_j, k-1] += 1
     return tensor 

def build_dataset_tensors(texts, vocab, window_size = 5):
     """build tensors for a list of texts using a shared vocabulary"""
     return [cooccurrence_tensor(t, vocab, window_size) for t in texts]

     
     