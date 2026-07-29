# Zero-Shot Structural Detection of Prompt Injection via Tensor Decomposition

This repository contains the research code, datasets, experiment scripts, and paper drafts for our study on detecting adversarial inputs in Retrieval-Augmented Generation (RAG) systems using Tucker decomposition.

---

## Problem Statement

Large Language Models (LLMs) used in Retrieval-Augmented Generation (RAG) applications are vulnerable to prompt injection attacks, where adversarial instructions hidden in queries, documents, or retrieved context bypass safety rules and cause harmful or unintended outputs.

A deployment team faces a critical challenge: they have no labeled examples of prompt injections from their own traffic. Without labeled attack data, fine-tuning or supervised defenses are infeasible.

---

## Research Gap

Existing defenses often fail under label scarcity:

- Gradient-based fine-tuning → requires labeled attack data.
- Model-specific patches → brittle and collapse under unseen attacks.
- Generic pretrained classifiers → transfer poorly to new domains.

None provide a lightweight, generalizable, zero-shot detection method that works without retraining, labeled attack data, or deep access to model internals.

---

## Proposed Solution

We introduce a structural, zero-shot anomaly detection framework based on co-occurrence tensor decomposition (Tucker Decomposition).

Unlike semantic drift methods (e.g., ZEDD) that measure meaning shifts, our approach models the structural regularities of benign prompts and flags anomalies when injected prompts deviate from this learned distribution.

### **Key Features**

- Zero-shot / zero-label → trained only on benign prompts, we don't need examples of attacks or special labels.
- Structure-aware → detects anomalies in word/token patterns, not just meaning. Flag the unusual structure.
- Model-agnostic → works with any LLM without retraining.
- Benign-only calibration → thresholds set using normal prompts only where attack data isn't available.
- Robust to obfuscation → catches encoding tricks, formatting anomalies.
- Lightweight & efficient → no GPUs or heavy training required.
- Domain-adaptive → learns the normal style of prompts in your environment.

---

## Novelty

Our contribution is structural, **zero-shot anomaly detection** using co-occurrence tensor decomposition:

- Custom co-occurrence tensor: We build a special data structure that shows how words and context appear together. This isn’t available in standard libraries, so we designed it ourselves.
- Tucker decomposition : We use a mathematical technique to break down the tensor and uncover hidden patterns that represent normal, safe prompts.
- Structural vs semantic focus: Unlike ZEDD, which looks at changes in meaning, our method looks at unusual shapes and patterns in text — like odd token sequences, strange formatting, or hidden encodings.
- Zero-shot calibration: We only train on safe prompts and set thresholds using them. This matches real-world situations where attack examples aren’t available.
- Lightweight & efficient: Our method doesn’t need GPUs or heavy retraining. It’s fast, practical, and easy to deploy.

---

## Expected Results

- Zero-shot detection (no retraining).
- High accuracy.
- Low false positives.
- Generalizable across multiple LLMs (Llama, Qwen, Mistral).

---

## Experimental Design

- Prong A: Direct Injection — zero-shot detection of adversarial queries.
- Prong B: Indirect Injection — detection of malicious content hidden in retrieved context.

---

## Benchmarks and Datasets

### **Internal Benchmark**

**DS1 (deepset/prompt-injections, Brooks & Turner 2025)**

- 660 prompts (399 benign, 261 injected).
- Diverse attack types: jailbreak, system leak, task override, encoding tricks.

### **External Benchmark**

**LLMail-Inject (2026)**

- 51k+ aligned pairs (clean vs injected).
- 5 attack categories across multiple models (Llama, Qwen, Mistral, SBERT).
- Community-recognized dataset used by ZEDD.

### **Additional Benchmarks**

- Semalith Evaluation Suite (2026) → supervised SOTA alignment.
- AdvBench / JailbreakBench (2025–2026) → adversarial robustness testing.

---

## Baselines to Test Against

### 1. Semantic Drift Reference

**ZEDD (2025)** → prior zero-shot anomaly detection baseline.

### 2. Structural Baselines

- Char-TF-IDF + SVM (Brooks & Turner 2025)
- ByT5 (Google 2021)
- CANINE (Google 2021)
- ILSE (2026) → current SOTA structural encoder.

### 3. Supervised Classifiers

- Semalith v1.4 (2026) → supervised SOTA.
- Llama-Guard-3-8B (2025) → large guardrail model.
- Fine-tuned DeBERTa/RoBERTa (2025–2026) → mid-tier supervised baselines.

---
---

## Evaluation Metrics

### Core Classification Metrics

- **ROC-AUC (Area Under the Receiver Operating Characteristic Curve)**  
  Standard metric for anomaly detection. Measures how well the detector separates benign and injected prompts across different decision thresholds.

- **F1 Score (Macro or Weighted)**  
  Balances precision and recall, providing a reliable measure of overall detection performance, especially under class imbalance.

- **Precision & Recall**
  - **Precision** → Measures how many flagged prompts are truly adversarial.
  - **Recall** → Measures how many adversarial prompts are successfully detected.

- **Accuracy**  
  Provides overall classification performance but is reported alongside F1 Score and ROC-AUC due to potential class imbalance.

---

### Robustness & Deployment Metrics

- **False Positive Rate (FPR)**  
  Measures how often benign prompts are incorrectly flagged as malicious. A low FPR is critical for practical deployment, with a target of **<3%**, consistent with ZEDD.

- **False Negative Rate (FNR)**  
  Measures how many prompt injection attacks remain undetected.

- **Detection Latency / Efficiency**  
  Measures the average inference time per prompt, demonstrating the practicality of the proposed lightweight detection framework.

---

### Additional Evaluation

- **PR-AUC (Area Under the Precision-Recall Curve)**  
  Complements ROC-AUC and provides a more informative evaluation when adversarial prompts are relatively rare.

- **Calibration Curves / Threshold Analysis**  
  Evaluates the effectiveness of benign-only calibration methods (e.g., percentile thresholds, Gaussian Mixture Models, and Kernel Density Estimation) for anomaly detection.

- **Cross-Model Generalization**  
  Reports performance separately across multiple LLM families (e.g., Llama, Qwen, and Mistral) to evaluate model-agnostic portability and robustness.

---

## Contributions

- A reproducible benchmark for prompt injection in RAG systems.
- Tucker decomposition-based zero-shot anomaly detection pipeline.
- Comparative analysis against semantic (ZEDD), structural (ILSE, ByT5, CANINE), and supervised (Semalith, TF-IDF+SVM) baselines.
- Benchmarked on DS1 (internal) and LLMail-Inject (external) for reproducibility and scalability.

---
---

## References

### Prompt Injection & Zero-Shot Detection

- **ZEDD (Zero-Shot Embedding Drift Detection, 2025)**  
  *Zero-Shot Embedding Drift Detection for Prompt Injection Attacks*  
  Introduced semantic drift detection as a lightweight, model-agnostic baseline. Essential zero-shot reference for this work.  
  Paper: https://arxiv.org/abs/2504.11049

- **Brooks & Turner (2025)**  
  *An Empirical Evaluation of Prompt Injection Detection and Refusal-Usefulness Tradeoffs Using the deepset/prompt-injections Dataset*  
  Established char-level TF-IDF + SVM as a supervised baseline. Introduced the DS1 dataset (399 benign, 263 injected prompts). Key reference for structural baselines.  
  Paper: https://arxiv.org/abs/2504.11168

---

### Structural Detection Baselines

- **ByT5 (Google, 2021)**  
  *ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Text Transformers*  
  Byte-level encoder robust to noisy and obfuscated text.  
  Paper: https://arxiv.org/abs/2105.13626

- **CANINE (Google, 2021)**  
  *CANINE: Pre-trained Character-Level Models for NLP*  
  Character-level encoder that eliminates tokenization and captures fine-grained structural irregularities.  
  Paper: https://arxiv.org/abs/2103.06874

- **ILSE (Inter-Layer Structural Encoders, 2026)**  
  *Improving LLM Predictions via Inter-Layer Structural Encoders* (arXiv:2603.22665)  
  Current state-of-the-art structural encoder. Aggregates signals across transformer layers using Cayley-Encoder graphs and reports substantial improvements across multiple tasks.  
  Paper: https://arxiv.org/abs/2603.22665

---

### Supervised Classifiers

- **Semalith v1.4 (2026)**  
  *Semalith: Multi-Axis Safety Classification for LLMs*  
  Built on DeBERTa-v3-base (184M). Current supervised SOTA for prompt injection detection and outperforms Llama-Guard-3-8B across multiple benchmarks.  
  Paper: https://arxiv.org/abs/2506.15656

- **Llama-Guard-3-8B (Meta AI, 2025)**  
  *Llama-Guard: LLM Safety Classifiers*  
  Large guardrail model providing strong general safety classification while being comparatively weaker on prompt injection detection.  
  Paper: https://arxiv.org/abs/2411.10414

---

### Tensor Decomposition Foundations & Applications

- **Tucker (1966)**  
  *Some Mathematical Notes on Three-Mode Factor Analysis*  
  Original paper introducing Tucker decomposition.

- **Harshman (1970)**  
  *PARAFAC: Parallel Factor Analysis*  
  Introduced Canonical Polyadic (CP/PARAFAC) tensor decomposition.

- **Kolda & Bader (2009)**  
  *Tensor Decompositions and Applications*  
  Definitive survey covering tensor decomposition methods and applications.  
  Paper: https://epubs.siam.org/doi/10.1137/07070111X

- **Anandkumar et al. (2014)**  
  *Tensor Methods for Learning Latent Variable Models*  
  Demonstrates tensor decomposition techniques for latent variable learning in machine learning and NLP.  
  Paper: https://jmlr.org/papers/v15/anandkumar14b.html

- **Papalexakis, Faloutsos & Sidiropoulos (2016)**  
  *Tensors for Data Mining and Machine Learning: Theory, Algorithms, and Applications*  
  Comprehensive survey covering tensor methods for anomaly detection, representation learning, and data mining.  
  Paper: https://dl.acm.org/doi/10.1145/2911996

- **Sun et al. (2006)**  
  *Beyond Streams and Graphs: Dynamic Tensor Analysis*  
  Early work applying tensor analysis for anomaly detection in dynamic data.  
  Paper: https://dl.acm.org/doi/10.1145/1150402.1150425

- **CoCoTen (2025)**  
  *CoCoTen: Contextual Co-occurrence Tensor Representation for Language Modeling*  
  Introduces contextual co-occurrence tensor representations for modeling higher-order token relationships. Relevant to tensor construction methodologies used in this project.  
  Paper: https://arxiv.org/abs/2508.02997

---

### Benchmark Datasets

- **DS1 – deepset/prompt-injections (2025)**  
  Introduced by Brooks & Turner. Widely used benchmark for prompt injection detection.  
  Dataset: https://huggingface.co/datasets/deepset/prompt-injections

- **LLMail-Inject (2026)**  
  Modern benchmark for prompt injection detection in agentic email workflows. Evaluates detectors under realistic deployment conditions.
  Dataset: https://huggingface.co/datasets/microsoft/llmail-inject-challenge?utm_source=chatgpt.com

- **Semalith Evaluation Suite (2026)**  
  Multi-axis evaluation benchmark including prompt injection, harmful content, and compliance tasks.
  Dataset: https://huggingface.co/datasets/Zytra/finproof-bench

- **AdvBench**  
  Adversarial benchmark for evaluating LLM robustness and jailbreak resistance.  
  Paper: https://arxiv.org/abs/2307.15043

- **JailbreakBench**  
  Benchmark for evaluating jailbreak attacks and safety defenses in LLMs.  
  Project: https://jailbreakbench.github.io/

---


## Authors

- Surbi Karki

---

## Citation

If you use this work in your research, please cite the final paper once published.
```
