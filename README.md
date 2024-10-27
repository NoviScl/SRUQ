# Semantic Resonance Uncertainty Quantification: Calibrating LLM Confidence through Multi-Path Reasoning

This repo implements an AI-generated research idea. The original idea can be found in Appendix Q of the paper [Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers](https://arxiv.org/pdf/2409.04109). This serves as a demonstration of how our generated ideas could be implemented and executed as full research projects.

## Usage 

```
pip install -r requirements.txt
bash run.sh 
```

## Results 

Model: GPT-4o-mini

Dataset: GSM8K 

Without CoT (using character-level Jaccard similarity and PageRank centrality for SRUQ):

| Method | Accuracy | Brier Score | Expected Calibration Error | Cost |
|--------|----------|-------------|----------------------------|------|
| Logprob | 31.54 | 0.2983 | 0.3324 | 0.12 |
| Ensemble | 31.39 | 0.3299 | 0.3985 | 0.62 |
| SRUQ | | | | |

With CoT:

| Method | Accuracy | Brier Score | Expected Calibration Error |
|--------|----------|-------------|----------------------------|
| Logprob | | | |
| Ensemble | | | |
| SRUQ  | | | |


Dataset: HotpotQA

Without CoT:

| Method | Accuracy | Brier Score | Expected Calibration Error |
|--------|----------|-------------|----------------------------|
| Logprob | | | |
| Ensemble | | | |
| SRUQ | | | |

With CoT:

| Method | Accuracy | Brier Score | Expected Calibration Error |
|--------|----------|-------------|----------------------------|
| Logprob | | | |
| Ensemble | | | |
| SRUQ | | | |


## Ablation Study

Model: GPT-4o-mini

Dataset: GSM8K 

With CoT:

The impact of different centrality measures on the performance of SRUQ.

| Method | Accuracy | Brier Score | Expected Calibration Error |
|--------|----------|-------------|----------------------------|
| Degree Centrality | | | |
| Eigenvector Centrality | | | |
| PageRank Centrality | | | |


The impact of different similarity metrics on the performance of SRUQ.

| Method | Accuracy | Brier Score | Expected Calibration Error |
|--------|----------|-------------|----------------------------|
| Character-Level Jaccard | | | |
| Word-Level Jaccard | | | |
| Sentence Embedding Cosine | | | |
| LLM Judge | | | |