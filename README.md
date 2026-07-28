# Zero‑Shot Structural Detection of Prompt Injection via Tensor Decomposition

This repository contains the research code, datasets, experiment scripts, and paper drafts for our study on detecting adversarial inputs in Retrieval-Augmented Generation (RAG) systems using Tucker decomposition.

## Problem Statement

Large Language Models (LLMs) used in Retrieval‑Augmented Generation (RAG) applications are vulnerable to prompt injection attacks, where adversarial instructions hidden in queries, documents, or retrieved context bypass safety rules and cause harmful or unintended outputs.
A deployment team faces a critical challenge: they have no labeled examples of prompt injections from their own traffic. Without labeled attack data, fine‑tuning or supervised defenses are infeasible.

## Research Gap
Existing defenses often fail under label scarcity:

-Gradient‑based fine‑tuning → requires labeled attack data.
-Model‑specific patches → brittle and collapse under unseen attacks.
-Generic pretrained classifiers → transfer poorly to new domains.
None provide a lightweight, generalizable, zero‑shot detection method that works without retraining, labeled attack data, or deep access to model internals.

##Proposed Solution
We introduce a structural, zero‑shot anomaly detection framework based on co‑occurrence tensor decomposition (Tucker Decomposition).
Unlike semantic drift methods (e.g., ZEDD) that measure meaning shifts, our approach models the structural regularities of benign prompts and flags anomalies when injected prompts deviate from this learned distribution.
**Key Features**
-Zero‑shot / zero‑label → trained only on benign prompts, we don't need examples of attacks or special labels.
-Structure‑aware → detects anomalies in word/token patterns, not just meaning. Flag the unusual structure.
-Model‑agnostic → works with any LLM without retraining.
-Benign‑only calibration → thresholds set using normal prompts only where attack data isn't available.
-Robust to obfuscation → catches encoding tricks, formatting anomalies.
-Lightweight & efficient → no GPUs or heavy training required.
-Domain‑adaptive → learns the normal style of prompts in your environment.


## Novelty

Our contribution is structural, **zero‑shot anomaly detection** using co‑occurrence tensor decomposition:
-Custom co‑occurrence tensor: We build a special data structure that shows how words and context appear together. This isn’t available in standard libraries, so we designed it ourselves.
-Tucker decomposition : We use a mathematical technique to break down the tensor and uncover hidden patterns that represent normal, safe prompts.
-Structural vs semantic focus: Unlike ZEDD, which looks at changes in meaning, our method looks at unusual shapes and patterns in text — like odd token sequences, strange formatting, or hidden encodings.
-Zero‑shot calibration: We only train on safe prompts and set thresholds using them. This matches real‑world situations where attack examples aren’t available.
-Lightweight & efficient: Our method doesn’t need GPUs or heavy retraining. It’s fast, practical, and easy to deploy.

## Expected Results

-Zero‑shot detection (no retraining).
-High accuracy.
-Low false positives .
-Generalizable across multiple LLMs (Llama, Qwen, Mistral).

## Experimental Design
-Prong A: Direct Injection — zero‑shot detection of adversarial queries.

-Prong B: Indirect Injection — detection of malicious content hidden in retrieved context.

## Benchmarks and Datasets

**Internal Benchmark**
DS1 (deepset/prompt‑injections, Brooks & Turner 2025)
660 prompts (399 benign, 261 injected).
Diverse attack types: jailbreak, system leak, task override, encoding tricks.

**External Benchmark**
LLMail‑Inject (2026)
51k+ aligned pairs (clean vs injected).
5 attack categories across multiple models (Llama, Qwen, Mistral, SBERT).
Community‑recognized dataset used by ZEDD.

**Additional Benchmarks**
-Semalith Evaluation Suite (2026) → supervised SOTA alignment.
-AdvBench / JailbreakBench (2025–2026) → adversarial robustness testing.

## Baselines to Test Against
1. Semantic Drift Reference
ZEDD (2025) → prior zero‑shot anomaly detection baseline.

2.Structural Baselines
Char‑TF‑IDF + SVM (Brooks & Turner 2025)
ByT5 (Google 2021)
CANINE (Google 2021)
ILSE (2026) → current SOTA structural encoder.

3.Supervised Classifiers
Semalith v1.4 (2026) → supervised SOTA.
Llama‑Guard‑3‑8B (2025) → large guardrail model.
Fine‑tuned DeBERTa/RoBERTa (2025–2026) → mid‑tier supervised baselines.

## Contributions

-A reproducible benchmark for prompt injection in RAG systems.
-Tucker decomposition‑based zero‑shot anomaly detection pipeline.
-Comparative analysis against semantic (ZEDD), structural (ILSE, ByT5, CANINE), and supervised (Semalith,  TF‑IDF+SVM) baselines.
-Benchmarked on DS1 (internal) and LLMail‑Inject (external) for reproducibility and scalability.



## Authors

- Surbi Karki


## Citation

If you use this work in your research, please cite the final paper once published.
