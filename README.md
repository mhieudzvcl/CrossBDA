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
| ida-BD (77 imgs) | Hybrid FDA+FT 10% (Encoder Frozen) | 0.7527 | 0.2042 | 0.3687 |
| ida-BD (77 imgs) | Phase 3.4 Pseudo-Labeling (Self-Training) | 0.2805 | 0.0009 | 0.0848 |
| xBD-S12 (100 imgs) | Zero-shot (cv2.resize) | 0.0000 | 0.0000 | 0.0000 |
| xBD-S12 (100 imgs) | Zero-shot (LapSRN x8 Super-Res) | 0.0000 | 0.0000 | 0.0000 |
| Vietnam Case Study | Zero-shot | (Not evaluated) | (Not evaluated) | (Not evaluated) |

## 3. Environment Setup

Activate the Conda environment:
```bash
conda activate xbd_env
```

Install dependencies if not already installed:
```bash
pip install -r requirements.txt
```

## 4. Running Experiments

Evaluate on xBD Test set:
```bash
python src/evaluate.py --config configs/baseline.yaml --checkpoint experiments/baseline_resnet34/checkpoints/best_model.pth --split test
```

Zero-shot evaluation on ida-BD:
```bash
python src/eval_ida.py --checkpoint experiments/baseline_resnet34/checkpoints/best_model.pth
```

Train model from scratch:
```bash
python src/train.py --config configs/baseline.yaml
```

Monitor training progress:
```bash
tensorboard --logdir experiments/baseline_resnet34/logs
```

## 5. Data Structure
- Note: The `data/` directory is not pushed to GitHub and will be synced via Google Drive.
- `data/xBD/`: Original dataset, containing train/tier3/test splits.
- `data/ida-BD/`: Hurricane Ida dataset.
- `data/Vietnam-Floods/`: Satellite imagery from Copernicus EMS (to be added in Phase 3).

## 6. Web Application (Demo)
The model is integrated with a Web UI for visual classification results.
Start the backend server:
```bash
cd webapp/backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Access `http://localhost:8000` to view the demo.