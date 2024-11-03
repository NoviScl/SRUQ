# Semantic Resonance Uncertainty Quantification: Calibrating LLM Confidence through Multi-Path Reasoning

This repo implements an AI-generated research idea. The original idea can be found in Appendix Q of the paper [Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers](https://arxiv.org/pdf/2409.04109). This serves as a demonstration of how our generated ideas could be implemented and executed as full research projects.

## Usage 

```
pip install -r requirements.txt
bash run.sh 
```

## Original Idea

See this document for the idea input along with my proposed modifications: [idea doc](https://docs.google.com/document/d/1SzdD6CPYpT_4yQaCeT1xFLA56PrLr2W8EwFNmTlnsX4/edit?usp=sharing). 

## Main Results 

Few-shot CoT prompting on GSM8K with GPT-4o-mini:

| Method | Accuracy | Brier Score | Expected Calibration Error | AUC | Cost |
|--------|----------|-------------|----------------------------|------|------|
| Logprob | 90.1 | 0.090 | 0.060 | 93.8 | 0.48 | 
| Ensemble | 92.9 | 0.054 | 0.041 | 95.7 | 2.39 |
| SRUQ (Proposed) | 92.7| 0.056 | 0.038 | 97.2 | 2.38 |   

Few-shot CoT prompting on StrategyQA with GPT-4o-mini:

| Method | Accuracy | Brier Score | Expected Calibration Error | AUC | Cost |
|--------|----------|-------------|----------------------------|------|------|
| Logprob | 83.5 | 0.154 | 0.127 | 89.3 | 0.54 |
| Ensemble | 83.7 | 0.142 | 0.126 | 86.8 | 2.72 |
| SRUQ  | 84.1 | 0.142 | 0.128 | 87.1 | 2.72 |

## Full Report 

Read the [full report](https://www.overleaf.com/read/jswkzmpmhwfn#fd1147). 