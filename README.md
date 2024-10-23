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

Without CoT:

| Method | Accuracy | Brier Score | Expected Calibration Error |
|--------|----------|-------------|----------------------------|
| Baseline | | | |
| Ensemble Confidence | | | |
| SRUQ Confidence | | | |

With CoT:

| Method | Accuracy | Brier Score | Expected Calibration Error |
|--------|----------|-------------|----------------------------|
| Baseline | | | |
| Ensemble | | | |
| SRUQ  | | | |


Dataset: HotpotQA

Without CoT:

| Method | Accuracy | Brier Score | Expected Calibration Error |
|--------|----------|-------------|----------------------------|
| Baseline | | | |
| Ensemble | | | |
| SRUQ | | | |

With CoT:

| Method | Accuracy | Brier Score | Expected Calibration Error |
|--------|----------|-------------|----------------------------|
| Baseline | | | |
| Ensemble | | | |
| SRUQ | | | |
