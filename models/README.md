# Models

Trained model checkpoints (segmentation models, C3D dose prediction model) are hosted on Google Drive rather than committed to git, since weight files typically exceed GitHub's size limits.

🔗 **Download:** [https://drive.google.com/drive/folders/1SbWAaUVuYMCrhCQ6FJw_bqnsYpwSvDrg?usp=sharing](https://drive.google.com/drive/folders/1SbWAaUVuYMCrhCQ6FJw_bqnsYpwSvDrg?usp=sharing)

## Setup

Download the checkpoints and place them here, e.g.:

```
models/
├── segmentation/
│   ├── oar_segmentation.pth
│   └── tumour_segmentation.pth
└── dose_prediction/
    └── c3d_dose_model.pth
```

Update the paths above to match your actual checkpoint filenames once you add the code that loads them.

Everything under `models/` (except this README) is git-ignored.
