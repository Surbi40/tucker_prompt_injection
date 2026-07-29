# experiments/run_04/train_and_freeze.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from src.preprocessing.tensor_builder import (
    build_vocab,
    build_tensor,
    compute_norm_constant,
)

from src.decomposition.tucker_decomposer import (
    fit_tucker,
    reconstruction_error,
)

from src.utils.config import (
    load_config,
    get_project_root,
    set_seed,
)

config = load_config()
set_seed(config["random_seed"])
ROOT = get_project_root()

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

WINDOW = 10
VOCAB_SIZE = 500
RANK = 30
SEED = 42

TUCKER_KWARGS = {
    "init": "random",
    "tol": 1e-4,
    "n_iter_max": 25,
}


def main():

    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)

    with open(
        ROOT / "datasets" / "prompt_injection" / "ds1_deepset_clean.json",
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    texts = np.array([x["text"] for x in data])
    labels = np.array([x["label"] for x in data])

    print(f"Total prompts : {len(texts)}")

    # --------------------------------------------------------------
    # Benign-only training split
    # --------------------------------------------------------------

    benign_idx = np.where(labels == 0)[0]

    train_idx, _ = train_test_split(
        benign_idx,
        test_size=0.20,
        shuffle=True,
        random_state=SEED,
    )

    train_texts = texts[train_idx]

    print(f"Benign training prompts : {len(train_texts)}")

    # --------------------------------------------------------------
    # Frozen vocabulary
    # --------------------------------------------------------------

    vocab = build_vocab(
        list(train_texts),
        vocab_size=VOCAB_SIZE,
        min_freq=2,
    )

    print(f"Vocabulary size : {len(vocab)}")

    # --------------------------------------------------------------
    # Frozen normalization constant
    # --------------------------------------------------------------

    norm_const = compute_norm_constant(
        list(train_texts),
        window=WINDOW,
        method="median",
    )

    print(f"Normalization constant : {norm_const:.3f}")

    # --------------------------------------------------------------
    # Build training tensor
    # --------------------------------------------------------------

    print("\nBuilding training tensor...")

    train_tensor = build_tensor(
        list(train_texts),
        vocab,
        window=WINDOW,
        norm_constant=norm_const,
    )

    print("Training tensor shape:", train_tensor.shape)

    # --------------------------------------------------------------
    # Tucker decomposition
    # --------------------------------------------------------------

    fold_rank = [
        min(RANK, train_tensor.shape[0]),
        min(RANK, train_tensor.shape[1]),
        min(RANK, train_tensor.shape[2] - 1),
    ]

    print("\nRunning Tucker decomposition...")

    model = fit_tucker(
        train_tensor,
        ranks=fold_rank,
        **TUCKER_KWARGS,
    )

    err = reconstruction_error(
        train_tensor,
        model["core"],
        model["factors"],
    )

    print(f"Fit time              : {model['fit_time_sec']:.2f} sec")
    print(f"Reconstruction error  : {err:.6f}")

    # --------------------------------------------------------------
    # Save frozen model
    # --------------------------------------------------------------

    frozen_model = {
        "core": model["core"],
        "factors": model["factors"],
        "embeddings": model["embeddings"],
        "projection_basis": model["projection_basis"],
        "projection_matrix": model["projection_matrix"],
        "centroid": model["centroid"],
        "vocab": vocab,
        "norm_const": norm_const,
        "window": WINDOW,
    }

    out_dir = ROOT / "results" / "run_04"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_path = out_dir / "frozen_model.pkl"

    with open(save_path, "wb") as f:
        pickle.dump(frozen_model, f)

    print("\n" + "=" * 60)
    print("Frozen model saved successfully.")
    print(save_path)
    print("=" * 60)

    print("\nSaved contents:")
    print("  ✓ Tucker core")
    print("  ✓ Tucker factors")
    print("  ✓ Benign embeddings")
    print("  ✓ Projection basis")
    print("  ✓ Projection matrix")
    print("  ✓ Benign centroid")
    print("  ✓ Frozen vocabulary")
    print("  ✓ Frozen normalization constant")
    print("\nNo training tensor is stored.")


if __name__ == "__main__":
    main()