# Building Damage Assessment - Cross-Domain Generalization

Topic: Evaluating and improving cross-domain generalization capabilities using multi-resolution satellite imagery for building damage assessment.

## 1. Overview
This repository contains the source code and experimental results for the thesis.
Main objectives:
- Train a baseline model on the standard xBD dataset.
- Measure the Domain Shift Gap when the model evaluates on different geographical regions or lower resolution satellite imagery (ida-BD, xBD-S12).
- Experiment on a Vietnam Case Study (using Copernicus EMS data for floods) to evaluate real-world applicability in Vietnam.
- Apply preprocessing and adaptation methods (Domain Adaptation, Fine-tuning, Super-Resolution) to improve performance.

## 2. Current Experimental Results

| Dataset | Evaluation Type | F1 Loc | F1 Dmg | xView2 Score |
|---|---|---|---|---|
| xBD (Test split) | In-distribution | 0.8509 | 0.7376 | 0.7716 |
| ida-BD | Zero-shot (Baseline) | 0.6853 | 0.1717 | 0.3258 |
| ida-BD | Zero-shot (FDA Inference) | 0.6647 | 0.1598 | 0.3112 |
| ida-BD | Zero-shot (FDA Aug) | 0.7424 | 0.2506 | 0.3982 |
| ida-BD (77 imgs) | Few-Shot 5% (Linear Probing) | 0.6904 | 0.1695 | 0.3258 |
| ida-BD (77 imgs) | Few-Shot 10% (Linear Probing) | 0.6904 | 0.1707 | 0.3266 |
| xBD-S12 (100 imgs) | Zero-shot (cv2.resize) | 0.0000 | 0.0000 | 0.0000 |
| xBD-S12 (100 imgs) | Zero-shot (LapSRN x8 Super-Res) | 0.0000 | 0.0000 | 0.0000 |
| Vietnam Case Study | Zero-shot | (Not evaluated) | (Not evaluated) | (Not evaluated) |