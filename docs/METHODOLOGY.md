# Methodology

This document summarizes the OAR-Guard pipeline architecture. For full detail, background literature, and justification, see the complete final report.

## Overview

The pipeline is built in two phases:

1. **Structural foundation** — extracting tumour and organ-at-risk (OAR) structural information from imaging data.
2. **Dose-aware ranking** — using that structural output to predict dose and rank organs by risk (the primary contribution of this project).

Each stage is modular: it consumes the previous stage's output and produces a well-defined output for the next, so individual stages can be developed, tested, and evaluated independently.

## The Ten Pipeline Steps

1. Patient data preparation
2. Preprocessing and formatting
3. Tumour segmentation
4. Organ-at-risk segmentation
5. 3D dose estimation (C3D)
6. Organ mask processing
7. Organ-wise dose statistics
8. Risk score computation
9. Patient-specific organ ranking
10. Visual & tabular output generation

## Dataset

**[OpenKBP](https://github.com/ababier/open-kbp)** — a public head-and-neck radiotherapy planning dataset. Each patient directory contains:

- CT and planning information
- organ-at-risk masks
- target volume information
- clinical dose information
- voxel spacing information

Patients are split into non-overlapping **train / validation / test** sets; the test set is held out entirely for final evaluation.

## Preprocessing

Each patient is converted into a multi-channel 3D tensor stacking:
- anatomical information
- target structure information
- organ-at-risk masks
- possible dose region information

## Structural Segmentation

- **Tumour extraction** — identifies the treatment target volume (the disease-centred reference for downstream risk analysis).
- **Organ-at-risk extraction** — segments the anatomical structures that must be protected during treatment, providing the spatial substrate for all later dose statistics.

Models used: MONAI 3D segmentor, SAM (Segment Anything Model), vision transformers, SegResNetDS.

## C3D Dose Prediction Model

The dose prediction component is the technical core of the system: a **Cascade 3D Convolutional Neural Network (C3D)** that takes the multi-channel patient representation and predicts a complete volumetric dose distribution (Gy per voxel).

**Architecture:** encoder–decoder, operating natively in 3D.
- **Encoder** — progressively abstracts the multi-channel input, learning organ locations, target volume shape, tumour–OAR spatial relationships, and high-risk regions.
- **Decoder** — reconstructs a full-resolution 3D dose map across the entire dose spectrum (low, medium, high, and hotspot regions).
- **Cascade design** — the dose estimate is progressively refined through multiple stages of multi-scale feature extraction, capturing both global dose-field structure and local hotspot detail.

### Training Configuration

| Setting | Value |
|---|---|
| Hardware | NVIDIA RTX 4060, NVIDIA RTX 4090 |
| Optimizer | Adam |
| Initial learning rate | 3 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| LR schedule | Cosine annealing |
| Total iterations | ~80,000 (~160 effective epochs) |

## Organ Mask Processing

Segmentation masks are aligned to be spatially consistent with the predicted dose volume — a prerequisite for computing valid per-organ statistics.

## Organ-Wise Dose Statistics

For each organ:
- **Mean dose** — average exposure over the organ volume
- **Max dose** — highest dose received by any voxel
- **Top-1% dose** — mean dose of the top 1% highest-dose voxels (stable high-dose estimate)
- **Fraction above 20 Gy** — proportion of organ volume receiving a clinically significant dose

## Risk Score

```
Risk Score = 0.5 × Mean Dose + 0.3 × Max Dose + 0.2 × Top-1% Dose
```

- Mean dose is weighted highest (0.5) — sustained exposure across a larger volume matters most clinically.
- Max dose (0.3) captures worst-case exposure, important for serial structures like the spinal cord.
- Top-1% dose (0.2) adds sensitivity to localized irradiation.

This formula is **functional, not clinically standardized** — it's intended for relative quantification (ranking organs against each other within a patient), not absolute clinical risk.

## Organ Ranking

Organs are ranked from highest to lowest risk score. The top three are highlighted as the primary, secondary, and tertiary "Next Organ at Risk" (NOAR).

## Output Generation

- Predicted dose heatmaps (slice-wise)
- Dose contours
- Top-3 organ summary
- Risk-score bar charts
- Detailed ranking tables (exportable)

## Interactive Dashboard

A tabbed web dashboard lets a clinician explore results per patient without writing code:

- **Overview** — dose heatmap + top-ranked organs
- **Heatmap Explorer** — dose maps at various body locations
- **Ranking Table** — full organ ranking with all statistics
- **Downloads** — export results for external use

## Implementation Notes

The pipeline was implemented as seven independently runnable/evaluable modules:

1. Patient-specific data preprocessing
2. Structural image information extraction
3. Dose prediction
4. Organ mask processing
5. Organ-wise dose statistics computation
6. Risk-score ranking
7. Output visualization

Each module takes structured input from the previous module and produces structured output for the next; intermediate outputs (masks, dose volumes, statistics tables) can be saved independently for debugging and verification.
