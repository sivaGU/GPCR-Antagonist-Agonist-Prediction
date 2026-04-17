# GPCR-FAP — Class A functional activity GUI (submission bundle)

Self-contained Streamlit app for **GPCR Class A** receptor–ligand **multiclass** prediction (Agonist / Antagonist / Inactive), with optional **3D structure view** (py3Dmol).

This folder is intended to be the **root of a GitHub repository** (push these contents as `main`).

## What is included

| Path | Purpose |
|------|--------|
| `streamlit_app.py` | Streamlit entrypoint |
| `src/gpcr/` | Feature construction + `load_predictor` |
| `requirements.txt` | Python dependencies |
| `data/` | Optional receptor list fallback + demo CSV |
| `artifacts/` | `feature_config.json` per model family; **add** `model_seed*.pkl` here |
| `Josh_Receptor_Features/` | Pocket CSVs + PDBs per receptor (bundled; ~70 targets) |

## Quick start

```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## Trained models (required for predictions)

Add multiclass estimators (`predict_proba` → 3 columns: Agonist, Antagonist, Inactive; **2103** input features) as:

- `artifacts/demo_rf/model_seed*.pkl` (or `.joblib`)
- `artifacts/demo_lightgbm/model_seed*.pkl`
- `artifacts/demo_xgboost/model_seed*.pkl`
- `artifacts/demo_ensemble/model_seed*.pkl`

Without these files the app still runs, but **Predict** will error until artifacts are present.

## Receptor data path

If `Josh_Receptor_Features/` sits in the **repository root** (as in this bundle), nothing else is required.

To use data elsewhere:

```bash
set GPCR_DATA_ROOT=C:\path\to\folder\that\contains\Josh_Receptor_Features
```

## GitHub and large files

`Josh_Receptor_Features/` contains many PDBs. If the repo exceeds GitHub’s limits, ship receptor data via **Git LFS**, a **release ZIP**, or a **separate data download** and document `GPCR_DATA_ROOT` for users.

## 3D viewer

Install includes **py3Dmol**. The 3D panel shows receptor cartoon + reference + query ligand (RDKit pose aligned to the orthosteric reference ligand); it is **not** a scored AutoDock/Vina run.

## Citation

Use your manuscript / Zenodo / DOI when publishing.
