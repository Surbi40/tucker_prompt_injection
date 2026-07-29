import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import numpy as np
from src.utils.config import get_project_root
from src.preprocessing.tensor_builder import build_vocab, build_tensor, prompt_cooccurrence_matrix
from src.decomposition.tucker_decomposer import run_tucker
from src.decomposition.tucker_projection import project_new_prompt

ROOT = get_project_root()
with open(ROOT / "datasets" / "prompt_injection" / "ds1_deepset_clean.json", encoding="utf-8") as f:
    data = json.load(f)
texts = [d["text"] for d in data]
labels = np.array([d["label"] for d in data])
benign_texts = [t for t, l in zip(texts, labels) if l == 0]

vocab = build_vocab(benign_texts, vocab_size=500, min_freq=2)
tensor = build_tensor(benign_texts, vocab, window=10)
embeddings, core, factors, _ = run_tucker(tensor, ranks=(30, 30, 30), init="random", tol=1e-4, n_iter_max=25)
A, B, C = factors

# Pick 5 held-in benign prompts, project them, compare to their actual row in C
errors = []
for idx in [0, 10, 50, 100, 150]:
    cooc = prompt_cooccurrence_matrix(benign_texts[idx], vocab, window=10).toarray()
    c_projected = project_new_prompt(cooc, core, A, B)
    c_actual = C[idx]
    rel_error = np.linalg.norm(c_projected - c_actual) / (np.linalg.norm(c_actual) + 1e-10)
    errors.append(rel_error)
    print(f"idx={idx}: relative error = {rel_error:.4f}")

print(f"\nMean relative error: {np.mean(errors):.4f}")
print("PASS" if np.mean(errors) < 0.05 else "FAIL — check unfold ordering / lstsq setup")