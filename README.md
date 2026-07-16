# OAR-Guard

**Segmentation, Dose Prediction, and Next-Organ-at-Risk Ranking for Head & Neck Radiotherapy**

Final Year Project — Department of Computer Science, National University of Computer and Emerging Sciences (FAST-NUCES), Islamabad, Pakistan
Session 2022–2026

> Supervised by **Dr. Labiba Fahad**

---

## Team

| Name | Roll Number |
|---|---|
| Sohaib Shahzad | 22I-2034 |
| Maaz Ali | 22I-1873 |
| Siraj Ali | 22I-2033 |

---

## Overview

Radiotherapy for head & neck cancer requires delivering enough radiation to destroy the tumour while sparing nearby organs at risk (OARs) such as the spinal cord, brainstem, parotids, mandible, and larynx — all packed tightly around the tumour site. Manual contouring of these structures is slow, inconsistent between clinicians, and stops at anatomy: it tells you *where* the organs are, but not how much radiation dose they are actually likely to receive.

**OAR-Guard** closes that gap. It's an end-to-end pipeline that goes from raw CT/MRI scans all the way to a ranked, patient-specific list of the organs most at risk of harmful radiation exposure — with an interactive dashboard a clinician can actually use.

The pipeline in one line:

```
CT/MRI scan → Tumour + OAR segmentation → 3D dose prediction (C3D) → per-organ dose statistics → risk score → ranked "Next Organ at Risk" → interactive dashboard
```

## Why this project

Existing research in this space tends to stop at one of two places:
- **Segmentation-only work** — accurate organ/tumour contours, measured by Dice/Hausdorff scores, with no connection to actual dose or clinical risk.
- **Dose-prediction-only work** — predicts a 3D dose volume, but doesn't break it down per-organ or produce anything a planner could rank and act on.

Very little work connects the two into a single, patient-specific pipeline that ends in an interpretable ranking. OAR-Guard is built specifically to close that gap — see [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the full research context.

## Pipeline

1. **Patient data preparation & preprocessing** — CT/MRI normalization, registration, formatting into a multi-channel 3D tensor.
2. **Tumour extraction** — identifies the treatment target volume.
3. **Organ-at-risk extraction** — segments anatomical structures (MONAI 3D segmentor, SAM, vision transformers) that need protecting.
4. **C3D dose prediction** — a cascaded 3D CNN (encoder–decoder) predicts a full volumetric dose map (Gy) from the multi-channel patient representation.
5. **Organ mask processing** — aligns segmentation masks with the predicted dose volume.
6. **Organ-wise dose statistics** — mean dose, max dose, top-1% dose, fraction above 20 Gy, per organ.
7. **Risk score computation**:

   ```
   Risk Score = 0.5 × Mean Dose + 0.3 × Max Dose + 0.2 × Top-1% Dose
   ```

8. **Patient-specific organ ranking** — sorts organs from highest to lowest predicted risk (the "Next Organ at Risk").
9. **Visual & tabular output generation** — dose heatmaps, top-3 organ summary, risk-score bar charts, full ranking table.
10. **Interactive dashboard** — tabbed web interface (Overview, Heatmap Explorer, Ranking Table, Downloads) for exploring per-patient results without writing code.

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for full architectural detail and [`docs/RESULTS.md`](docs/RESULTS.md) for evaluation metrics and results.

## Tech Stack

- **Segmentation:** MONAI, SAM (Segment Anything Model), SegResNetDS, Vision Transformers
- **Dose Prediction:** Cascade 3D CNN (C3D), PyTorch
- **Dataset:** [OpenKBP](https://github.com/ababier/open-kbp) — public head & neck radiotherapy planning dataset
- **Interface:** Interactive web dashboard (clinician-facing)
- **Training hardware:** NVIDIA RTX 4060 / RTX 4090

## Dataset & Large Files

The dataset, trained model checkpoints, and other large artifacts are **not stored in this repository** (GitHub isn't built for multi-GB medical imaging data). They're hosted on Google Drive instead:

🔗 **[Dataset & Model Weights — Google Drive](https://drive.google.com/drive/folders/1SbWAaUVuYMCrhCQ6FJw_bqnsYpwSvDrg?usp=sharing)**

After downloading, place the contents according to [`data/README.md`](data/README.md) and [`models/README.md`](models/README.md) before running the pipeline.

## Repository Structure

```
oar-guard/
├── src/
│   ├── preprocessing/      # CT/MRI loading, normalization, registration
│   ├── segmentation/       # Tumour + OAR segmentation (MONAI, SAM)
│   ├── dose_prediction/    # C3D cascade 3D CNN model + training/inference
│   ├── risk_ranking/       # Dose statistics, risk score, organ ranking
│   └── visualization/      # Heatmaps, bar charts, ranking tables
├── dashboard/               # Interactive clinician-facing dashboard app
├── data/                    # Dataset placeholder — see data/README.md (Drive-hosted)
├── models/                  # Trained model checkpoints — see models/README.md (Drive-hosted)
├── notebooks/                # Exploratory / experiment notebooks
├── results/                  # Generated outputs (heatmaps, tables, exports)
├── docs/
│   ├── METHODOLOGY.md       # Full pipeline & architecture writeup
│   └── RESULTS.md            # Evaluation metrics & results
├── tests/                    # Unit tests
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/oar-guard.git
cd oar-guard

# 2. Create environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Download the dataset & model weights from the Drive link above,
#    then place them per data/README.md and models/README.md

# 4. Run the pipeline (example)
python src/preprocessing/prepare_patient.py --patient pt_339
python src/segmentation/run_segmentation.py --patient pt_339
python src/dose_prediction/predict_dose.py --patient pt_339
python src/risk_ranking/rank_organs.py --patient pt_339

# 5. Launch the dashboard
cd dashboard
python app.py
```

> Exact script names/flags above are placeholders — update this section once your actual scripts are in place.

## Evaluation Metrics

| Metric type | Metrics used |
|---|---|
| Segmentation | Dice Similarity Coefficient |
| Dose / risk | Mean dose, max dose, top-1% dose, fraction above 20 Gy, risk score |
| Deployment | Inference time, model size, memory usage |
| Usability | Clinician feedback, visualization clarity |

Representative Dice scores obtained during development:

| Organ | Dice Score |
|---|---|
| Mandible | 0.81 |
| Spinal Cord | 0.78 |
| Brainstem | 0.76 |
| **Mean** | **0.75** |

Full results are in [`docs/RESULTS.md`](docs/RESULTS.md).

## Limitations

- Risk score weighting (0.5 / 0.3 / 0.2) is empirically chosen, not clinically validated.
- The system is for **research purposes only** — it is not connected to any clinical planning system and dose predictions are not a clinician-approved treatment protocol.
- Output is optimized for accuracy and clarity of risk ranking, not full anatomical fidelity.

## Future Work

- Clinical validation of the risk-scoring formula against expert-annotated cases.
- Extending beyond head & neck to other anatomical regions.
- Tighter integration with clinical planning systems.

## References

Full Harvard-style references are listed in the final report ([`docs/REPORT.pdf`](docs/REPORT.pdf), if included, or your submitted PDF).

## License

This project is released under the [MIT License](LICENSE) unless your department requires otherwise — update as appropriate for your submission.
