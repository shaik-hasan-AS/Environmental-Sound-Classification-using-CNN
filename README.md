# MR-AFCNN: Multi-Resolution Audio-Frequency Convolutional Neural Network

> **An efficient, novel CNN architecture for Environmental Sound Classification**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

MR-AFCNN is a lightweight yet powerful CNN designed specifically for **environmental sound classification** on mel-spectrogram inputs. Unlike generic image-based CNNs applied to audio, MR-AFCNN introduces several **audio-specific structural innovations** that exploit the unique properties of time-frequency representations.

### Key Novelties

| Innovation | Description |
|---|---|
| **3-Channel Delta Features** | Input is a 3-channel tensor: static mel-spectrogram + Δ (velocity) + ΔΔ (acceleration), encoding temporal dynamics explicitly |
| **Multi-Scale Depthwise Convolutions** | Parallel `3×3`, `1×5`, and `5×1` depthwise kernels capture short events, long events, and wide-band events simultaneously |
| **Dual Attention (SE + Frequency-Spatial)** | Combines Squeeze-Excitation (channel attention) with a novel Frequency-Axis Spatial Attention that pools over time but retains per-frequency-bin weights |
| **Supervised Contrastive Loss (SupCon)** | Multi-head loss with Cross-Entropy + SupCon + Fused classification, forcing tight latent-space clustering of similar sounds |
| **Stochastic Weight Averaging (SWA)** | Automatically averages model weights in the final 25% of training for flatter minima and better generalization |
| **Test-Time Augmentation (TTA)** | Averages predictions over 5 augmented views at inference for a free accuracy boost |

### Architecture Summary

```
Input: [3, 128, 128] (mel + delta + delta-delta)
  │
  ├── Stem: Conv2d(3→32) + BN + SiLU, stride=2
  │
  ├── Stage 1: 2× MRAFCNNBlock (32→64),   stride=2
  ├── Stage 2: 2× MRAFCNNBlock (64→128),  stride=2
  ├── Stage 3: 3× MRAFCNNBlock (128→256), stride=2
  ├── Stage 4: 2× MRAFCNNBlock (256→384), stride=2
  │
  ├── AdaptiveAvgPool2d → flatten
  │
  ├── Head 1: CE classifier (384→50)
  ├── Head 2: SupCon projector (384→128, L2-normalized)
  └── Head 3: Fused classifier (384+256→50)

Total Parameters: ~3.0M
```

---

## Results

### ESC-50 (Environmental Sound Classification, 50 classes)

| Model | Params | Accuracy | Type |
|---|---|---|---|
| Human Baseline | — | 81.30% | — |
| Piczak CNN | — | 64.50% | CNN |
| TF-Attention CNN | — | 84.40% | CNN + Attention |
| **MR-AFCNN (Ours)** | **3.5M** | **83.25%** | **Efficient CNN (novel)** |
| AST | 87M | 95.60% | Transformer |

---

## Project Structure

```
ESC_SIH/
├── Model.py          # MR-AFCNN architecture (MultiScaleDW, DualAttention, SupCon)
├── dataset.py        # Feature extraction (3-ch delta), augmentations, data loaders
├── Train.py          # Training pipeline with SWA, Mixup, cosine LR
├── Evaluate.py       # Evaluation with TTA, confusion matrices, Grad-CAM
├── Requirements.txt  # Python dependencies
├── setup_and_run.sh  # One-click setup script
├── eval_outputs/     # Generated plots, CSVs, and analysis reports
├── checkpoints/      # Saved model weights
└── ESC-50/           # Dataset (not included, download separately)
```

---

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r Requirements.txt
```

### 2. Download Dataset

Download [ESC-50](https://github.com/karolpiczak/ESC-50) and extract to `./ESC-50/`.

### 3. Train

```bash
# Train on ESC-50, Fold 1, 300 epochs
python train.py --dataset esc50 --root ./ESC-50 --epochs 300 --cache --lr 3e-4 --batch_size 32 --num_workers 0 --folds 1

# Train all 5 folds (full cross-validation)
python train.py --dataset esc50 --root ./ESC-50 --epochs 300 --cache
```

### 4. Evaluate

```bash
# Standard evaluation with full reports
python evaluate.py --dataset esc50 --root ./ESC-50 --ckpt ./checkpoints/esc50_fold1_best.pt --fold 1 --full

# With Test-Time Augmentation (recommended)
python evaluate.py --dataset esc50 --root ./ESC-50 --ckpt ./checkpoints/esc50_fold1_best.pt --fold 1 --full --tta
```

---

## Training Features

- **Mixup Augmentation** (α=0.4) at batch level with soft label cross-entropy
- **SpecAugment** (frequency + time masking)
- **Cosine Annealing with Warm Restarts** (T₀=50, T_mult=2)
- **Linear Warmup** (10 epochs)
- **Gradient Clipping** (max norm = 5.0)
- **Early Stopping** (patience = 30 epochs)
- **Mixed Precision Training** (AMP) on CUDA
- **SWA** activated at 75% of total epochs

---

## Citation

If you use MR-AFCNN in your research, please cite:

```bibtex
@article{mrafcnn2026,
  title={MR-AFCNN: An Efficient Multi-Resolution Audio-Frequency CNN for Environmental Sound Classification},
  author={Shaik Hasan A S, Prajin S},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License.
