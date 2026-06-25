# Tucker Prompt Injection Detection in RAG Systems

This repository contains the research code, datasets, experiment scripts, and paper drafts for our study on detecting adversarial inputs in Retrieval-Augmented Generation (RAG) systems using Tucker decomposition.

## Project Overview

Prompt injection is a major security risk in RAG systems because malicious instructions can appear either in the user query or inside retrieved context. These attacks can alter model behavior, leak system information, or cause the model to ignore safety instructions.

This project investigates whether Tucker decomposition applied to a custom co-occurrence tensor can capture latent multilinear structure that helps distinguish benign inputs from adversarial ones. We focus on RAG-style security settings where malicious content may be hidden in retrieved passages.

## Research Objectives

- Build a clean dataset for benign QA, prompt injection, and RAG-style context injection.
- Construct a co-occurrence tensor representation for RAG inputs.
- Apply Tucker decomposition to extract latent multilinear structure.
- Compare Tucker-based features with classical baselines and prior tensor-based approaches.
- Evaluate detection performance on both direct and indirect prompt injection settings.

## Novelty

Our key novelty is the use of a **custom co-occurrence tensor** for adversarial input detection in RAG systems. Since no off-the-shelf library directly provides the exact tensor construction needed for this setting, we implement the tensor-building step from scratch to preserve the intended multilinear structure.

We then apply **Tucker decomposition**, optimized with **HOOI**, to recover latent multiway interactions from the tensor. TensorLy is used as the decomposition backend, while our contribution focuses on the task-specific tensor construction and its application to RAG security.

## Repository Structure

- `data/` - raw, processed, and split datasets.
- `datasets/` - dataset-specific files, metadata, and prepared samples.
- `src/` - reusable source code for preprocessing, tensor construction, decomposition, modeling, and evaluation.
- `baselines/` - baseline methods such as regex, TF-IDF + Logistic Regression, Random Forest, SVM, and transformer baselines.
- `sota/` - state-of-the-art papers, summaries, and comparison notes.
- `experiments/` - experiment runs, ablations, and result logs.
- `paper/` - LaTeX, Word, and PDF versions of the manuscript.
- `drafts/` - abstract drafts, section drafts, and revisions.
- `figures/` - plots, diagrams, and architecture figures.
- `tables/` - dataset summaries, metric tables, and comparison tables.
- `results/` - final predictions, metrics, and evaluation outputs.
- `references/` - bibliography files and reading notes.
- `scripts/` - helper scripts for data preparation, training, and evaluation.
- `tests/` - unit tests and sanity checks.
- `configs/` - configuration files for datasets, models, and experiments.
- `logs/` - training and evaluation logs.
- `docs/` - project notes, methodology notes, and planning documents.
- `models/` - saved models, vectorizers, and tensor artifacts.

## Data Sources

Planned datasets include:
- TriviaQA for benign QA and RAG-style query evaluation.
- Prompt injection datasets for direct adversarial input detection.
- RAG-context injection data for indirect prompt injection experiments.

## Method Summary

The main pipeline is:

1. Build a co-occurrence tensor from query and retrieved context.
2. Apply Tucker decomposition using HOOI.
3. Extract latent factors and core tensor features.
4. Train a downstream classifier for benign vs adversarial detection.
5. Evaluate on direct and indirect prompt injection benchmarks.

TensorLy is used for Tucker decomposition, but the custom tensor construction is implemented in this repository.

## Planned Baselines

- Regex-based detection.
- TF-IDF + Logistic Regression.
- Random Forest.
- Support Vector Machine.
- Transformer-based classifier.
- Prior tensor-based adversarial detection methods.


## Usage

Recommended workflow:

1. Place raw datasets in `data/raw/`.
2. Run preprocessing scripts from `scripts/` or `src/preprocessing/`.
3. Generate co-occurrence tensors using the tensor construction module.
4. Run Tucker decomposition and feature extraction.
5. Train baselines and evaluate results.
6. Save outputs in `results/`, `figures/`, and `tables/`.

## Planned Experiments

- Direct prompt injection detection.
- Indirect prompt injection detection in retrieved context.
- Comparison between Tucker features and classical baselines.
- Ablation studies on tensor rank and classifier choice.
- Robustness testing under RAG-style attack scenarios.

## Expected Contributions

- A custom co-occurrence tensor representation for RAG security.
- Tucker decomposition-based adversarial detection pipeline.
- A reproducible benchmark for prompt injection in RAG systems.
- Comparative analysis against classical and tensor-based baselines.

## Notes

- This repository is for academic research only.
- Any dataset used will follow its applicable license and usage terms.
- Final results, figures, and paper drafts will be updated as the project progresses.


## Important SOTA References Link
https://genai.owasp.org/llmrisk/llm01-prompt-injection/
https://galileo.ai/blog/ai-prompt-injection-attacks-detection-and-prevention
https://www.nature.com/articles/s41598-026-43883-0
https://www.microsoft.com/en-us/msrc/blog/2025/07/how-microsoft-defends-against-indirect-prompt-injection-attacks
https://www.preprints.org/manuscript/202510.0418/v1
## Authors

- Surbi Karki


## Citation

If you use this work in your research, please cite the final paper once published.