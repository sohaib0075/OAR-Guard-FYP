# Experiments & Results

## Evaluation Metrics

Two metric families are used, matching the two phases of the pipeline:

**Segmentation quality (structural phase):**
- **Dice Similarity Coefficient** — `Dice = 2|P ∩ G| / (|P| + |G|)`, where P = predicted mask, G = ground-truth mask. 1 = perfect overlap, 0 = no overlap.

**Dose / risk quality (analysis phase):**
- Mean dose, max dose, top-1% dose, fraction above 20 Gy
- Risk score (see [METHODOLOGY.md](METHODOLOGY.md#risk-score))

**Deployment & usability:**
- Inference time, model size, memory usage
- Clinician feedback, visualization clarity

## Segmentation Results

Representative Dice scores for selected anatomical structures:

| Organ | Dice Score |
|---|---|
| Mandible | 0.81 |
| Spinal Cord | 0.78 |
| Brainstem | 0.76 |
| **Mean** | **0.75** |

Larger, geometrically distinct structures (mandible, spine) score higher and more consistently. Smaller, low-contrast structures show more variance across models.

## Sample Organ-at-Risk Ranking (Patient `pt_339`)

| Rank | Organ | Voxels | Mean Dose (Gy) | Max Dose (Gy) | Top-1% Dose (Gy) | Frac. Above 20 Gy | Risk Score |
|---|---|---|---|---|---|---|---|
| 1 | Larynx | 140 | 71.697 | 72.877 | 72.824 | 1.000 | 72.277 |
| 2 | Spinal Cord | 295 | 49.055 | 73.652 | 73.342 | 0.766 | 61.291 |
| 3 | Mandible | 988 | 12.124 | 60.292 | 59.549 | 0.272 | 36.059 |
| 4 | Brainstem | 332 | 10.993 | 59.395 | 59.133 | 0.223 | 35.142 |
| 5 | Left Parotid | 146 | 15.549 | 25.600 | 24.984 | 0.062 | 20.451 |
| 6 | Right Parotid | 110 | 0.401 | 44.124 | 22.062 | 0.009 | 17.850 |

## Output Views

Per patient, the system produces four complementary output views:

1. **Predicted dose heatmap** — slice-wise spatial visualization of dose.
2. **Top-3 organ summary** — highest-risk organs with their key statistics.
3. **Risk-score bar chart** — comparison across all analyzed organs.
4. **Detailed ranking table** — full statistics for every organ, exportable.

## Interpretation

- Larynx and spinal cord dominate the risk ranking in this sample case due to a combination of high mean dose and high fraction of volume above 20 Gy.
- Larger organs (mandible) can still rank moderately high even with lower mean dose, due to their max/top-1% dose components.
- The risk score is designed for **within-patient relative ranking**, not cross-patient or absolute clinical comparison.

## Limitations Observed

- Risk-score weighting is empirical, not clinically validated.
- The system is research-only and not connected to any clinical planning system.
- Output favors clarity/ranking accuracy over full anatomical fidelity.
- Composite dose + organ contour overlays were considered but dropped due to lack of reliable image registration; all visualizations are generated purely from computed statistics instead.
