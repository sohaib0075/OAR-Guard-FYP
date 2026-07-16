# Data

The dataset used in this project (**OpenKBP** — head & neck radiotherapy planning cases) is not stored in this repository due to its size.

🔗 **Download:** [https://drive.google.com/drive/folders/1SbWAaUVuYMCrhCQ6FJw_bqnsYpwSvDrg?usp=sharing](https://drive.google.com/drive/folders/1SbWAaUVuYMCrhCQ6FJw_bqnsYpwSvDrg?usp=sharing)

## Setup

1. Download the dataset from the Drive link above.
2. Extract it into this `data/` folder so the structure looks like:

```
data/
├── train/
│   ├── pt_001/
│   ├── pt_002/
│   └── ...
├── validation/
│   └── ...
└── test/
    └── ...
```

Each patient directory contains:
- CT and planning information
- Organ-at-risk masks
- Target volume information
- Clinical dose information
- Voxel spacing information

3. Everything under `data/` (except this README) is git-ignored — it won't be committed.
