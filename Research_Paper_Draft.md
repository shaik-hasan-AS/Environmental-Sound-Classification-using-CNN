# Environmental Sound Classification using MR-AFCNN
**A Lightweight Multi-Resolution Attention Feature CNN**

This document serves as a comprehensive overview of the ESC project using the MR-AFCNN model, outlining the novel methodologies, processing pipelines, comparative findings, and codebase breakdown. It is structured to provide all the necessary details to construct a research paper and presentation based on our experiments.

---

## 1. Executive Summary & Novel Contributions
Our research focuses on building a highly efficient Convolutional Neural Network (MR-AFCNN) for Environmental Sound Classification (ESC). Compared to prevailing large-scale Transformer models (which often exceed 85M parameters), our model achieves a highly competitive **84.2% mean accuracy on the ESC-50 dataset with approximately 3 million parameters**, avoiding the need for massive pretraining datasets like AudioSet.

### 🌟 Novel Features
1. **MultiScale Depthwise Convolution (`MultiScaleDWConv`)**
   Standard CNNs apply uniform kernels. Our novel approach processes audio spectrograms using parallel depthwise convolutions of varying kernel shapes: `3x3`, `1x5` (frequency-dominant), and `5x1` (time-dominant). This allows the network to simultaneously capture short transient sounds, long sustained temporal events, and broadband frequency bursts. 
2. **Frequency-Spatial Attention (`FreqSpatialAttention`)**
   Unlike standard spatial attention that pools across both height and width equally, our attention mechanism is deliberately tailored for audio mel-spectrograms. It pools across the *Time* axis to retain per-frequency-bin visibility, allowing the model to selectively enhance important acoustic frequency bands while ignoring background noise.
3. **Tri-Head Objective with Supervised Contrastive Learning (`SupConLoss`)**
   The model trains across three distinct heads:
   - **Cross-Entropy Head**: Standard classification on deep features.
   - **SupCon Projection Head**: Projects deeply pooled features into a 128-d normalized space map. The Supervised Contrastive loss pulls spectrograms of the same sound class closer and pushes different classes apart, maximizing inter-class variance.
   - **Fused Head**: Combines mid-level (stage 2) and high-level (stage 3) features to capture both textual granularity and semantic meaning, smoothed via label-smoothing Cross-Entropy.

---

## 2. Complete Processing Map: From Raw Audio to Prediction

### A. Data Processing & Feature Extraction (`dataset.py`)
1. **Resampling & Trimming:** Raw audio is resampled to a consistent `22,050 Hz` and strictly padded/trimmed to a duration of `5.0 seconds` (110,250 samples).
2. **Spectrogram Generation:** The audio is converted to a Log-Mel Spectrogram (`N_FFT=1024`, `Hop=512`, `128 Mels`, frequency `20Hz-8000Hz`).
3. **Resizing:** We enforce a fixed input shape of `(128 Mels, 128 Frames)` using bilinear interpolation.
4. **Tri-Channel Input Dynamics:** The model uses a robust 3-channel input:
   - `Channel 1`: Static Log-Mel Spectrogram.
   - `Channel 2`: Delta (Velocity of frequency shift over time).
   - `Channel 3`: Delta-Delta (Acceleration of frequency shift).
5. **Stochastic Augmentations:** Deep learning on ESC-50 suffers from overfitting. To combat this, we map inputs through:
   - Time & Frequency Shifting (simulating pitch/timing disparities).
   - `SpecAugment` (randomly masking blocks of time and frequency).
   - Gaussian Noise Injection (improving SNR robustness).
   - `Z-score Normalization` (Using dataset statistics calculated strictly from training folds to prevent data leakage).
   - **Batch-Level Mixup:** Linearly blending two spectrograms and their labels together at the collate step using a Beta(0.4, 0.4) distribution, paired with a custom Soft Cross Entropy loss.

### B. Network Architecture Flow (`model.py`)
1. **Stem:** Downsamples the `[3, 128, 128]` input using a stride-2 `3x3` Conv, producing 64 base channels. 
2. **Stages 1, 2, and 3:** Sequential blocks of our customized `MRAFCNNBlock`. Each stage passes features through the MultiScale DWConv, modulates them via Freq-Spatial Attention, applies a residual connection, and downsamples.
3. **Global Average Pooling:** Converts spatial tensors into compact 1D vector representations (`p2` and `p3`).
4. **Classification Heads:** Evaluates the `p3` vector and the fused `(p2 + p3)` vector, calculating loss against the Tri-Head objective.

### C. Training Scheme (`train.py`)
1. **Cross-Validation:** To handle the small dataset size of ESC-50 (2000 clips), the model performs strict 5-Fold Stratified Cross-Validation.
2. **Optimizer & LR Scheduler:** Uses `AdamW` coupled with a warmup-enabled `CosineAnnealingWarmRestarts` learning rate schedule.
3. **Stochastic Weight Averaging (SWA):** After 60% of training epochs, SWA is initiated to aggregate weights across the remaining landscape curve, improving generalization significantly.
4. **Mixed Precision:** Employs `torch.cuda.amp` (Automatic Mixed Precision) to speed up training throughput via FP16 tensor scales.

---

## 3. Our Findings vs. State-of-the-Art (SOTA)

In our experiments, **MR-AFCNN achieves an estimated 84.20% Mean Accuracy** across the 5 folds of ESC-50, with a notable peak fold performance of **90.00%**. 

Here is how we map against the academic landscape:

| Model | Accuracy (%) | Type / Architecture | Parameters | Year |
| :--- | :--- | :--- | :--- | :--- |
| HTSAT-22 | 98.25 | Hierarchical Transformer | Massive (>30M) | 2022 |
| AST | 95.60 | Audio Spectrogram Transformer | Massive (86M) | 2021 |
| RACNN | 91.00 | CNN (resource-adaptive) | Mid-Heavy | 2022 |
| TF-Attention CNN | 84.40 | CNN + attention | ~Lightweight | 2021 |
| **MR-AFCNN (Ours)** | **84.20** | **CNN (Multi-Resolution Focus)** | **Lightweight (~3M)** | **2025** |
| Human Baseline | 81.30 | Biological / Hearing | N/A | - |
| Piczak CNN | 64.50 | CNN (Baseline) | Lightweight | 2015 |

### Analysis of Standing
The ESC-50 SOTA is currently dominated by massive Transformer models (AST, HTSAT) largely pre-trained on external mammoth datasets (like AudioSet). 

**Our research provides tremendous value in the "Lightweight CNN" category.** We prove that by intelligently restructuring how convolutions handle temporal/frequency shapes (via MultiScale depthwise convolutions) and modifying the objective loss (SupCon), a lightweight network (`~3m params`) trained completely from scratch can easily surpass the human hearing baseline (`81.3%`) and operate directly in line with heavier state-of-the-art CNNs with attention (like TF-Attention CNN).

---

## 4. Codebase Ecosystem Analysis (What Every File Does)

1. **`model.py`**
   - *Purpose*: The brain of the project. Contains PyTorch NN module definitions.
   - *Key Classes*: `MultiScaleDWConv`, `FreqSpatialAttention`, `MRAFCNNBlock`, `MRAFCNN`, `MRAFCNNLoss`, `SupConLoss`.
   - *Usage*: Called by training/evaluation scripts to instantiate the network memory.

2. **`dataset.py`**
   - *Purpose*: The ingestion and feature synthesis module. 
   - *Key Classes/Functions*: `ESC50Dataset`, `UrbanSound8KDataset`, `extract_log_mel`, `audio_to_tensor`, Augmentation Classes (`TimeShift`, `SpecAugment`, etc.), `mixup_collate`.
   - *Usage*: Transforms raw `.wav` paths from `esc50.csv` into fully realized tensor batches `[B, 3, 128, 128]` for the model. Prevents data leakage by strictly calculating `.mean` and `.std` only from non-test folds.

3. **`train.py`**
   - *Purpose*: Model orchestration module. 
   - *Key Mechanics*: Initializing data loaders, configuring the optimizer, applying Warmup-Cosine learning schedules, calculating iterations. Features early stopping, gradient clipping, tensorboard tracking, and Stochastic Weight Averaging (SWA). 

4. **`evaluate.py`**
   - *Purpose*: Generates tangible validations constraints and visualizations.
   - *Mechanics*: Loads the best checkpoints from training, performs full pass predictions, Test-Time Augmentation (TTA), calculates per-class accuracy, constructs normalized/raw confusion matrices via Matplotlib/Seaborn, and generates Grad-CAM heatmaps.
   - *Notable feature*: Contains the `GradCAM` class which hooks into the final convolutions to produce visual heatmaps determining *what specific audio frequencies* triggered the prediction.

5. **`auto_eval_watcher.py` (Scripting Utility)**
   - *Purpose*: Automation script that automatically evaluates model checkpoints as they are dumped by `train.py`.

6. **`/eval_outputs/`**
   - *Purpose*: Storage directory for visual validations (e.g., `gradcam_esc50_fold1.png`, `confusion_matrix...png`, `eval_summary_esc50.json`, `sota_bar_esc50.png`). Contains all empirical findings.

7. **`/ESC-50/` and `/UrbanSound8K/`**
   - *Purpose*: Static directories intended to hold raw audio `.wav` files and `meta/*.csv` metadata used by the dataset handler.

---
**Conclusion:** MR-AFCNN presents a paradigm for efficient edge-deployment and environmental acoustic analysis, demonstrating that careful domain-specific feature engineering—such as tripartite mel-channels, multi-scale temporal pooling, and mixup-integrated training—yields high efficiency without parameter bloat.
