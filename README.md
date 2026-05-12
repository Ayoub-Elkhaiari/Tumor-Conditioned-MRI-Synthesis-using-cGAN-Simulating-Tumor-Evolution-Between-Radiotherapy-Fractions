# Tumor-Conditioned MRI Synthesis using Conditional GAN: Simulating Tumor Evolution Between Radiotherapy Fractions

> A conditional GAN that synthesizes realistic brain MRI slices conditioned on evolved tumor
> segmentation masks, simulating tumor state changes between radiotherapy treatment fractions.
> Achieves **SSIM=0.9933**, **PSNR=39.18dB**, **MAE=0.0032** on the test set.

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Tumor Evolution Operators](#tumor-evolution-operators)
- [Architecture](#architecture)
- [Training](#training)
- [Results](#results)
- [Qualitative Results](#qualitative-results)
- [Difference Maps](#difference-maps)
- [Training Curves](#training-curves)
- [Metrics Table](#metrics-table)
- [Setup](#setup)
- [Relevance to Radiotherapy Digital Twins](#relevance-to-radiotherapy-digital-twins)

---

## Overview

This project implements a **tumor-conditioned image synthesis pipeline** using a conditional GAN
(Pix2Pix). Given a real MRI slice at timepoint T and a modified segmentation mask representing
the tumor state at T+1, the model generates a realistic synthetic MRI at T+1 that reflects the
new tumor state.

This directly mirrors the digital twin loop in adaptive radiotherapy planning:

```
segment → evolve mask → synthesize image → plan next fraction → repeat
```

---

## Motivation

In radiotherapy planning, clinicians need to predict how a tumor will look after each treatment
fraction to adapt the dose delivery. Building a **digital twin** of the patient requires a model
that can generate realistic medical images conditioned on a predicted tumor state. This project
demonstrates that a conditional GAN can learn this mapping directly from data, producing
high-fidelity synthetic MRI that reflects biologically meaningful tumor changes.

The methodology is directly transferable to:
- Prostate MRI 
- Pseudo-CT synthesis from MRI
- Pseudo-PET synthesis from MRI

---

## Dataset

**BraTS 2020 — Brain Tumor Segmentation Challenge** (Kaggle: awsaf49/brats20-dataset-training-validation)

| Property | Value |
|---|---|
| Patients used | 50 |
| MRI modality | T2 |
| Total slices extracted | 3,078 |
| Slice filter | tumor pixels > 50 |
| Image size | 256 × 256 |
| MRI normalization | [−1, 1] (Tanh) |
| Mask normalization | [0, 1] binary |
| Train / Val / Test split | 35 / 7 / 8 patients |
| Train slices | 2,185 |
| Val slices | 436 |
| Test slices | 457 |

Segmentation classes: 0=background, 1=necrotic core, 2=edema, 4=enhancing tumor.
All classes binarized to a single tumor mask for conditioning.

---

## Tumor Evolution Operators

Three biologically-motivated operators simulate tumor state changes between fractions:

<img width="1314" height="985" alt="image" src="https://github.com/user-attachments/assets/ff201d13-a668-4167-8b0f-71b20c9eb225" />


| Operator | Implementation | Biological Meaning | Probability |
|---|---|---|---|
| Shrinkage | Binary erosion (3–8 iter) | Treatment response | 50% |
| Growth | Binary dilation (2–6 iter) | Tumor progression | 30% |
| Necrosis | Random core dropout (20–50%) | Necrotic core formation | 20% |

---

## Architecture

### Generator — Conditional U-Net

```
Input: concat(MRI slice, Evolved Mask) → 2 channels

Encoder:
  Conv(2→64)   → LeakyReLU           [no BN on first layer]
  Conv(64→128) → InstanceNorm → LeakyReLU
  Conv(128→256)→ InstanceNorm → LeakyReLU
  Conv(256→512)→ InstanceNorm → LeakyReLU
  Conv(512→512)→ InstanceNorm → LeakyReLU  × 3

Bottleneck:
  Conv(512→512) → ReLU

Decoder (with skip connections + Dropout on first 3 blocks):
  ConvTranspose(512→512) → InstanceNorm → ReLU → Dropout(0.5)  × 3
  ConvTranspose(1024→512)→ InstanceNorm → ReLU
  ConvTranspose(1024→256)→ InstanceNorm → ReLU
  ConvTranspose(512→128) → InstanceNorm → ReLU
  ConvTranspose(256→64)  → InstanceNorm → ReLU

Output: ConvTranspose(128→1) → Tanh → synthesized MRI in [-1, 1]
```

**Generator params: 54,399,553**

### Discriminator — PatchGAN

```
Input: concat(MRI, Mask) → 2 channels

Conv(2→64)   → LeakyReLU            [no norm]
Conv(64→128) → InstanceNorm → LeakyReLU
Conv(128→256)→ InstanceNorm → LeakyReLU
Conv(256→512)→ InstanceNorm → LeakyReLU
Conv(512→1)  → patch output (30×30)
```

**Discriminator params: 2,762,817**
**Patch size: 30×30** — each output value classifies a 70×70 receptive field as real or fake.

---

## Training

| Setting | Value |
|---|---|
| Loss | GAN (BCEWithLogitsLoss) + L1 (λ=100) |
| Optimizer | Adam (β1=0.5, β2=0.999) |
| Learning rate | 2e-4 |
| LR schedule | Linear decay after epoch 50 |
| Batch size | 4 |
| Max epochs | 100 |
| Early stopping | Patience=15 on val SSIM |
| Mixed precision | torch.cuda.amp |
| Label smoothing | Real=0.9, Fake=0.0 |
| Hardware | Google Colab T4 GPU |
| Random seed | 42 |

**Training strategy:**
- Update Discriminator once per batch
- Update Generator once per batch
- Generator loss = adversarial loss + 100 × L1 loss

---

## Results

### Overall Test Set Performance

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| MAE | 0.003215 | 0.000658 | 0.001664 | 0.005515 |
| MSE | 0.000127 | 0.000044 | 0.000054 | 0.000315 |
| RMSE | 0.011135 | 0.001840 | 0.007338 | 0.017739 |
| NRMSE | 0.011135 | 0.001840 | 0.007338 | 0.017739 |
| **PSNR** | **39.18 dB** | 1.398 | 35.021 | 42.688 |
| **SSIM** | **0.9933** | 0.0026 | 0.9850 | 0.9979 |
| MID | 0.000253 | 0.000547 | -0.001108 | 0.002781 |
| SID | 0.011122 | 0.001821 | 0.007336 | 0.017704 |
| Tumor MAE | 0.032456 | 0.010357 | 0.010879 | 0.087659 |
| Tumor SSIM | 0.910336 | 0.044918 | 0.717064 | 0.983291 |
| BG MAE | 0.002777 | 0.000543 | 0.001562 | 0.004624 |
| **Dice** | **0.9762** | 0.0092 | 0.9373 | 0.9910 |

### Per-Mode Breakdown

| Mode | SSIM | PSNR (dB) | MAE | Dice | Difficulty |
|---|---|---|---|---|---|
| Shrinkage | 0.9945 | 39.72 | 0.0030 | 0.9796 | Easiest |
| Necrosis | 0.9930 | 39.44 | 0.0032 | 0.9753 | Medium |
| Growth | 0.9917 | 38.19 | 0.0036 | 0.9716 | Hardest |

> Growth is hardest to synthesize as it requires generating new tissue
> outside the original anatomical boundary — a fundamentally harder task
> than shrinking or perturbing existing structures.

---

## Qualitative Results

4-panel visualization: Input MRI (T) | Evolved Mask (T+1) | Generated MRI (T+1) | Real MRI

<img width="1352" height="1965" alt="image" src="https://github.com/user-attachments/assets/2c6c01e0-f120-4f6b-9f84-9ff44182f5a0" />


Generated MRI is visually nearly indistinguishable from real MRI across all three
evolution types after only 10 epochs of training.

---

## Difference Maps

Signed difference maps (Generated − Real) overlaid with tumor boundary contour (yellow).
Errors concentrate at tumor boundaries — consistent with high-frequency structural
detail being hardest to learn.

<img width="1790" height="1137" alt="image" src="https://github.com/user-attachments/assets/60d93a61-f2eb-47bc-8c64-914ae2187853" />

<img width="788" height="490" alt="image" src="https://github.com/user-attachments/assets/946b9039-08cc-4302-bdba-46762ad5a375" />

---

## Training Curves

<img width="1352" height="1377" alt="image" src="https://github.com/user-attachments/assets/0a14fe6e-e9fc-4f09-ad69-cb8ca3f7d2d4" />

<img width="1582" height="887" alt="image" src="https://github.com/user-attachments/assets/2a57b951-7829-4cc4-ac5a-dd365a4c6720" />


- L1 loss converges rapidly from epoch 1
- SSIM climbs to 0.993 within 10 epochs
- G loss increases as D gets stronger — classic GAN dynamic
- SSIM vs PSNR scatter shows consistent improvement over epochs

---

## Metrics Table

<img width="1389" height="495" alt="image" src="https://github.com/user-attachments/assets/a3cf17ff-d536-4e43-88be-7438d6875716" />


Full per-mode comparison showing MAE, SSIM, and PSNR across shrinkage, growth, and necrosis.

---

## Setup

### Requirements

```bash
pip install torch torchvision kagglehub nibabel scikit-image
pip install scikit-learn scipy matplotlib seaborn pandas tqdm monai
```

### Run on Google Colab

The entire project is in a single self-contained notebook:

```
TumorConditioned_cGAN_Synthesis.ipynb
```

Open in Colab → Runtime → Run All

The notebook will automatically:
1. Download BraTS2020 via `kagglehub`
2. Extract and preprocess tumor slices
3. Apply evolution operators to generate paired training data
4. Train Generator + Discriminator
5. Evaluate all 12 metrics on the test set
6. Generate all visualizations

---

## Stack

Python · PyTorch · BraTS2020 · kagglehub · nibabel · scikit-image · matplotlib · Google Colab T4

---

## Author

**Ayoub EL KHAIARI**
MSc Advanced Machine Learning and Multimedia Intelligence — USMBA, Fez
[GitHub](https://github.com/Ayoub-Elkhaiari) · [Portfolio](https://ayoub-elkhaiari.netlify.app/)
