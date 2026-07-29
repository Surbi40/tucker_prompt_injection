import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
import json
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from src.preprocessing.tensor_builder import (
    build_vocab, build_tensor, build_prompt_slice, compute_norm_constant,
)
from src.decomposition.tucker_decomposer import (
    fit_tucker, project_prompt, score_by_centroid_distance, reconstruction_error,
)
from src.utils.config import load_config, get_project_root, set_seed

config = load_config()
set_seed(config["random_seed"])
ROOT = get_project_root()

WINDOW = 10
RANK = 30
VOCAB_SIZE = 500
SEED = 42
TUCKER_KWARGS = dict(init="random", tol=1e-4, n_iter_max=25)


def main():
    """
    Sanity check for the inductive Tucker pipeline only: fit on benign-only
    data, project a held-out mixed test set, score, report ROC-AUC/PR-AUC.
    Also includes a rank sweep to check whether reconstruction error and
    detection AUC move together or diverge (they measure different things:
    compression fidelity vs. class-separation quality).

    NO calibration/thresholding happens here — see
    multiseed_calibration_framework.py for the leakage-free, bootstrap-
    conservative, multiseed calibrated result.
    """
    with open(ROOT / "datasets" / "prompt_injection" / "ds1_deepset_clean.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = np.array([row["text"] for row in data])
    labels = np.array([row["label"] for row in data])

    benign_idx = np.where(labels == 0)[0]
    benign_pool, benign_holdout = train_test_split(benign_idx, test_size=0.3, shuffle= True, random_state=SEED)
    test_idx = np.concatenate([benign_holdout, np.where(labels == 1)[0]])
    test_texts = texts[test_idx]
    test_labels = labels[test_idx]
    train_texts = texts[benign_pool]
    print(f"Benign-only fit set: {len(train_texts)} | Held-out test set: {len(test_texts)}")

    
    vocab = build_vocab(list(train_texts), vocab_size=VOCAB_SIZE, min_freq=2)
    norm_const = compute_norm_constant(list(train_texts), window=WINDOW, method="median")
    train_tensor = build_tensor(list(train_texts), vocab, window=WINDOW, norm_constant=norm_const)
    fold_rank = [
        min(RANK, train_tensor.shape[0]),
        min(RANK, train_tensor.shape[1]),
        min(RANK, train_tensor.shape[2] - 1),
    ]

    model = fit_tucker(train_tensor, ranks=fold_rank, **TUCKER_KWARGS)
    assert model["embeddings"].shape[0] == len(train_texts)
    assert model["centroid"].shape[0] == model["embeddings"].shape[1]
    fit_time = model["fit_time_sec"]
    print(f"Fit complete in {fit_time:.2f}s. Vocab, norm_const, and model are now FROZEN.")

    print("\nProjection basis verification (held-in points should show ~0 relative error):")
    for idx in [0, len(train_texts)//4, len(train_texts)//2]:
        slice_matrix = build_prompt_slice(train_texts[idx], vocab, window=WINDOW, norm_constant=norm_const)
        projected = project_prompt(slice_matrix, model)
        actual = model["embeddings"][idx]
        rel_error = np.linalg.norm(projected - actual) / (np.linalg.norm(actual) + 1e-10)
        print(f"  idx={idx}: relative error = {rel_error:.6f}")

    recon_err = reconstruction_error(train_tensor, model["core"], model["factors"])
    print(f"\nReconstruction error (benign fit tensor, rank={RANK}): {recon_err:.4f}")

  
    print("\nRank sweep — reconstruction error vs. detection AUC:")
    rank_sweep_results = []
    for test_rank in [10, 20, 30, 50, 80]:
        fold_rank_r = [
            min(test_rank, train_tensor.shape[0]),
            min(test_rank, train_tensor.shape[1]),
            min(test_rank, train_tensor.shape[2] - 1),
        ]
        model_r = fit_tucker(train_tensor, ranks=fold_rank_r, **TUCKER_KWARGS)
        err_r = reconstruction_error(train_tensor, model_r["core"], model_r["factors"])
        test_emb_r = np.array([
            project_prompt(build_prompt_slice(t, vocab, window=WINDOW, norm_constant=norm_const), model_r)
            for t in test_texts
        ])
        scores_r =  score_by_centroid_distance(model_r, test_emb_r)
        auc_r = roc_auc_score(test_labels, scores_r)
        prauc_r = average_precision_score(test_labels, scores_r)
        rank_sweep_results.append({
            "rank": test_rank,
            "reconstruction_error": float(err_r),
            "auc": float(auc_r),
            "pr_auc": float(prauc_r),
        })
        print(f"  rank={test_rank:3d}: reconstruction_error={err_r:.4f}  AUC={auc_r:.3f}  PR-AUC={prauc_r:.3f}")

    project_start = time.perf_counter()
    test_embeddings = np.array([
        project_prompt(
            build_prompt_slice(t, vocab, window=WINDOW, norm_constant=norm_const),
            model,
        )
        
        for t in test_texts
    ])
    assert test_embeddings.shape == (
    len(test_texts),
    model["embeddings"].shape[1],
)
    project_time = time.perf_counter() - project_start
    print(f"\nProjected {len(test_texts)} test prompts in {project_time:.3f}s "
          f"({project_time / len(test_texts) * 1000:.2f}ms/prompt) — no re-decomposition.")

    scores = score_by_centroid_distance(model, test_embeddings)
    auc = roc_auc_score(test_labels, scores)
    prauc = average_precision_score(test_labels, scores)
    print(f"\nInductive Tucker ROC-AUC (rank={RANK}): {auc:.3f}")
    print(f"Inductive Tucker PR-AUC  (rank={RANK}): {prauc:.3f}")

    out = {
        "note": "Scoring-only sanity check — no calibration/thresholds computed here. "
            "See multiseed_calibration_framework.py for the leakage-free, "
            "bootstrap-conservative, multiseed calibrated result.",

    "primary_rank": RANK,
    "embedding_dimension": int(model["embeddings"].shape[1]),
    "num_train_embeddings": int(model["embeddings"].shape[0]),

    "reconstruction_error": float(recon_err),
    "rank_sweep": rank_sweep_results,

    "auc": float(auc),
    "pr_auc": float(prauc),

    "fit_time_sec": float(fit_time),
    "project_time_sec": float(project_time),
    "project_time_per_prompt_ms": float(project_time / len(test_texts) * 1000),
    "window": WINDOW,
"vocab_size": len(vocab),
"normalization_constant": float(norm_const),
"fit_prompts": len(train_texts),
"test_prompts": len(test_texts),

    }
    out_dir = ROOT / "results" / "run_04"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "inductive_ds1_scoring_only.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_dir / 'inductive_ds1_scoring_only.json'}")


if __name__ == "__main__":
    main()