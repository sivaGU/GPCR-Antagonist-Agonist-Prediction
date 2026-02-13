# GPCR Class A Functional Activity Prediction GUI

A production-ready Streamlit GUI for predicting GPCR Class A receptor-ligand functional activity (Agonist/Antagonist/Inactive) using machine learning models.

## Features

- **Single prediction** — Upload GPCR Class A receptor and ligand (SMILES or structure file), get activity prediction
- **Batch CSV prediction** — Upload a CSV with receptor and ligand columns, download results
- **Multi-class classification** — Predicts Agonist, Antagonist, or Inactive
- **Uncertainty quantification** — Provides error probabilities and confidence intervals
- **Model support** — LightGBM, Random Forest, XGBoost (ensemble support)

## Quick Start

### 1. Create and activate a virtual environment

```bash
python -m venv venv
```

- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If RDKit fails, try:
```bash
pip install rdkit
pip install lightgbm pandas numpy streamlit scikit-learn joblib xgboost
```

### 3. Add ML artifacts

Place your trained model artifacts in the `artifacts/` folder:
- Model files (`.pkl`, `.joblib`, or `.pth` formats)
- `feature_config.json` (feature configuration)
- `threshold.json` (classification thresholds, if applicable)
- Receptor feature files (if needed)

### 4. Run the Streamlit GUI

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`.

## Project Structure

```
.
├── streamlit_app.py       # GPCR GUI
├── requirements.txt       # Dependencies
├── src/
│   └── gpcr/              # Prediction module
│       ├── predict.py     # predict_single, predict_batch, load_predictor
│       └── cli.py         # Command-line interface
├── artifacts/             # Model artifacts (add your models here)
│   ├── feature_config.json
│   ├── threshold.json
│   └── [your model files]
├── example_inputs.csv
└── example_outputs.csv
```

## Usage

### GUI

- **Single prediction:** Upload receptor file/name and ligand SMILES or structure file, click **Predict**
- **Batch CSV:** Upload a CSV with `receptor` and `ligand` (or `smiles`) columns, click **Predict batch**, then **Download CSV**

### CLI

From the project root:

```bash
# Predict single receptor-ligand pair
python -m src.gpcr.cli --receptor "ADRB2" --ligand "CCO" --output out.csv

# Predict from CSV
python -m src.gpcr.cli --input example_inputs.csv --output out.csv
```

## Model Details

Based on the ML GPCR Class A Functional Activity Manuscript:
- **Classes:** Agonist (0), Antagonist (1), Inactive (2)
- **Features:** Ligand features (PhysChem + ECFP) + Receptor features (31) + Interaction terms (14)
- **Models:** LightGBM, Random Forest, XGBoost (ensemble support)
- **Evaluation:** Baseline, Random Stratified, Scaffold Split, LORO (Leave-One-Receptor-Out)

## Requirements

- Python 3.9 or 3.10 (3.11/3.12 usually work)
- Dependencies in `requirements.txt`
- Trained ML model artifacts in `artifacts/` folder

## Notes

This GUI is designed to work with your trained ML models. Once you add your model artifacts to the `artifacts/` folder and update the `predict.py` module to match your feature extraction pipeline, the GUI will be ready to use.
