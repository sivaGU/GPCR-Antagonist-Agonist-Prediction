# Cursor Prompt: Create GPCR Class A Artifacts for the GUI

**Copy everything below this line into your Cursor prompt** so it can help you build the artifact folder to send back. The recipient will drop this folder into their GPCR GUI project so the Demo Prediction Tool works with all four models.

---

## Goal

Create a single folder named **`gpcr_artifacts`** (or `artifacts`) that contains trained model artifacts for a **GPCR Class A Functional Activity** classifier. The GUI expects **4 model types**: **LightGBM**, **XGBoost**, **Random Forest**, and **Ensemble**. Each type must live in its own subfolder with the same structure. When done, zip the folder and send it so it can be placed inside the GUI project as the `artifacts` folder (replacing or merging with the existing one).

---

## Task for Cursor

1. **Create the folder structure** below (all paths relative to the folder you will send).

2. **Train or load 4 model types** (each can be multi-seed ensemble, e.g. 3–5 models per type):
   - **Random Forest** (e.g. `sklearn.ensemble.RandomForestClassifier`)
   - **LightGBM** (e.g. `lightgbm.LGBMClassifier`)
   - **XGBoost** (e.g. `xgboost.XGBClassifier`)
   - **Ensemble** (e.g. average predictions from RF + LightGBM + XGBoost, or your own ensemble of multiple algorithms)

3. **Save models** in each subfolder with this naming:
   - `model_seed0.pkl`, `model_seed1.pkl`, `model_seed2.pkl`, … (use `.joblib` if you prefer; the GUI accepts both `.pkl` and `.joblib`).

4. **Add the two config JSON files** (see contents below) into **each** of the four subfolders.

5. **Ensure model API**: Each saved model must support **one** of:
   - **Option A:** `model.predict_proba(X)` returning a 2D array of shape `(n_samples, 3)` with probabilities for classes **Agonist (0), Antagonist (1), Inactive (2)**.
   - **Option B:** If your library uses a different API, the GUI’s loader expects to get 3-class probability outputs; adapt saving/loading so the final predictor used in the GUI receives feature matrix `X` of shape `(n, 2103)` and returns 3 probabilities per sample.

6. **Feature space**: The GUI builds a **2103-dimensional** feature vector per (receptor, ligand) pair:
   - Ligand: 2058 (e.g. 10 physicochemical + 2048 ECFP4).
   - Receptor: 31.
   - Interaction: 14.
   Train your models on data that uses this same feature construction (or equivalent). At prediction time, the GUI will pass `X` with 2103 columns.

7. **Output**: One folder (e.g. `gpcr_artifacts`) containing exactly the structure below. The recipient will place this as the `artifacts` folder in the GPCR GUI repo (or merge its contents into the existing `artifacts` folder).

---

## Required folder structure

```
gpcr_artifacts/
├── demo_rf/
│   ├── model_seed0.pkl
│   ├── model_seed1.pkl
│   ├── model_seed2.pkl
│   ├── feature_config.json
│   └── threshold.json
├── demo_lightgbm/
│   ├── model_seed0.pkl
│   ├── model_seed1.pkl
│   ├── ...
│   ├── feature_config.json
│   └── threshold.json
├── demo_xgboost/
│   ├── model_seed0.pkl
│   ├── ...
│   ├── feature_config.json
│   └── threshold.json
└── demo_ensemble/
    ├── model_seed0.pkl
    ├── ...
    ├── feature_config.json
    └── threshold.json
```

You can have more than 3 seeds (e.g. `model_seed3.pkl`, `model_seed4.pkl`); the GUI loads all `model_seed*.pkl` (and `model_seed*.joblib`) in each folder.

---

## Config file contents

Put these in **every** subfolder (`demo_rf`, `demo_lightgbm`, `demo_xgboost`, `demo_ensemble`).

**`feature_config.json`:**

```json
{
  "type": "GPCR_ClassA",
  "class_names": ["Agonist", "Antagonist", "Inactive"],
  "class_ids": { "Agonist": 0, "Antagonist": 1, "Inactive": 2 },
  "n_features_total": 2103
}
```

**`threshold.json`:**

```json
{
  "threshold": null,
  "note": "Multi-class: prediction = argmax(probabilities)."
}
```

---

## Checklist before sending

- [ ] Folder name is clear (e.g. `gpcr_artifacts` or `artifacts`).
- [ ] Four subfolders exist: `demo_rf`, `demo_lightgbm`, `demo_xgboost`, `demo_ensemble`.
- [ ] Each subfolder has at least one model file named `model_seed0.pkl` (or `.joblib`).
- [ ] Each subfolder contains `feature_config.json` and `threshold.json`.
- [ ] Each model accepts input shape `(n_samples, 2103)` and yields 3-class probabilities (Agonist, Antagonist, Inactive).
- [ ] Zip the folder and send it; the recipient will add it to the GUI as the `artifacts` folder.

---

## Quick reference for the recipient

- Place the contents of your zip so that the GUI project has:
  - `artifacts/demo_rf/`
  - `artifacts/demo_lightgbm/`
  - `artifacts/demo_xgboost/`
  - `artifacts/demo_ensemble/`
- Each of these must contain `model_seed*.pkl` (or `.joblib`) plus `feature_config.json` and `threshold.json`.
- The GUI’s Demo Prediction Tool lets users choose **Random Forest**, **LightGBM**, **XGBoost**, or **Ensemble** and runs the corresponding demo models.
