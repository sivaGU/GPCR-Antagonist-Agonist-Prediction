# Artifacts Folder

**Current contents:** Slimmed layout (under 25 MB) for deployment.

- **Root:** Config only (`feature_config.json`, `threshold.json`). The main **GPCR Ligand Functional Activity Prediction** tab uses **demo_ensemble/** when no models are in root.
- **demo_rf/** — One Random Forest model (`model_seed0.pkl`, compressed) for the Demo Prediction Tool.
- **demo_lightgbm/** — One LightGBM model.
- **demo_xgboost/** — One XGBoost model.
- **demo_ensemble/** — One ensemble model (also used as default for the main prediction tab).

To reduce size we removed duplicate nested folders, kept a single model per folder, and re-saved with joblib compression. To restore full artifacts (e.g. 3 seeds per folder), replace this folder with your full `gpcr_artifacts` and run `python slim_artifacts.py` from the project root to slim again.

All models expect **2103 features** and output **3-class probabilities** (Agonist, Antagonist, Inactive). Each demo subfolder includes `feature_config.json` and `threshold.json`.

## Run the GUI

From the project root:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then use **Demo Prediction Tool** (choose receptor, model type, run predictions) or **GPCR Ligand Functional Activity Prediction** (single/batch).
