# GPCR-FAP GUI

GPCR-FAP is a Streamlit-based GUI for **Class A GPCR** receptor–ligand **multiclass** functional activity prediction (Agonist / Antagonist / Inactive). It combines manuscript-aligned ligand and receptor pocket features with optional **3D structure viewing** (py3Dmol, MBind-style receptor cartoon + query ligand).

Repository: https://github.com/sivaGU/GPCR-FAP

## What This App Provides

- **Functional activity inference** from SMILES (or common structure files) for each bundled receptor target.
- **Models:** Random Forest, LightGBM, XGBoost, and ensemble artifacts under `artifacts/demo_*` (multiclass `predict_proba`, **2103** features per row).
- **Receptor assets:** `Josh_Receptor_Features/` — pocket CSVs, conservation summaries, and PDBs per target (~70 folders).
- **3D view:** Receptor cartoon (tan) plus **your** compound in the orthosteric region (RDKit 3D conformer, centroid aligned using the co-crystal site center only). **Native/co-crystal ligands are not drawn** in the viewer. This is **not** AutoDock/Vina docking.

## Quick Start

1. Clone the repository and open the project folder.
2. Create a virtual environment and install dependencies (see **Run Locally**).
3. Launch Streamlit and open the prediction page.
4. Choose a model, select a receptor, enter SMILES or upload a structure file, and run **Predict**.
5. Review class probabilities and optional 3D receptor + ligand view.

## Run Locally

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Launch:

   ```bash
   streamlit run streamlit_app.py
   ```

3. Open `http://localhost:8501`.

## Requirements

- Python **3.10+** recommended (3.10–3.12 tested).
- Dependencies in `requirements.txt` (RDKit, scikit-learn, LightGBM, XGBoost, Streamlit, py3Dmol, etc.).
- **Trained models:** `artifacts/demo_rf/`, `demo_lightgbm/`, `demo_xgboost/`, `demo_ensemble/` must contain `model_seed*.pkl` or `.joblib` plus `feature_config.json` (and optional `threshold.json`).

## Supported Inputs

- **Ligands:** SMILES string, or upload **SDF, MOL, PDB, PDBQT, MOL2**, or **CSV** (first SMILES column).
- **Receptors:** Selected from folders under `Josh_Receptor_Features/`.

## Receptor Data Path

If `Josh_Receptor_Features/` is in the **repository root**, no configuration is needed. If you keep data under a sibling `GUI_Folder/` next to this project, the app can auto-detect that layout when `GPCR_DATA_ROOT` is unset.

To point elsewhere:

```bash
set GPCR_DATA_ROOT=C:\path\to\folder\that\contains\Josh_Receptor_Features
```

## Project Structure

```
GPCR-FAP/
├── streamlit_app.py
├── README.md
├── requirements.txt
├── data/
│   ├── gpcr_class_a_receptors.txt
│   └── demo_reference.csv
├── src/
│   └── gpcr/
│       ├── predict.py
│       ├── structure_view.py
│       └── cli.py
├── artifacts/
│   ├── demo_rf/
│   ├── demo_lightgbm/
│   ├── demo_xgboost/
│   └── demo_ensemble/
└── Josh_Receptor_Features/
    └── <receptor_name>/
        ├── *_receptor_only.pdb
        ├── *_ligand_only.pdb          # used for site centering only (not shown in 3D)
        ├── *_pocket_residues_with_conservation.csv
        └── ...
```

## Large Repositories

`Josh_Receptor_Features/` contains many PDBs. If GitHub size limits are an issue, distribute data via **Git LFS**, a **release ZIP**, or an external download and set `GPCR_DATA_ROOT`.

## Notes on Validation Scope

- Outputs are **multiclass functional activity** predictions (agonist / antagonist / inactive), not binding free energies or crystallographic poses.
- The **3D panel** places your ligand near the orthosteric site for visualization only; it does not replace AutoDock/Vina or other scored docking pipelines.

## Citation

Cite the **GPCR-FAP / Class A functional activity prediction** manuscript, Zenodo release, or DOI when publishing.

## Contact

Questions, issues, or collaboration requests: **Dr. Sivanesan Dakshanamurthy** — sd233@georgetown.edu
