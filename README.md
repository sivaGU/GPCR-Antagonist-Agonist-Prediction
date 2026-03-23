# Streamlit Cloud Upload Bundle

This folder contains the minimum runtime files for deploying the app on Streamlit Cloud.

## Required Streamlit settings

- Main file path: `streamlit_app.py`
- Python version: 3.10 or 3.11 recommended

## Important

To run real predictions, add trained model files to:

- `artifacts/demo_rf/model_seed*.pkl` (or `.joblib`)
- `artifacts/demo_lightgbm/model_seed*.pkl` (or `.joblib`)
- `artifacts/demo_xgboost/model_seed*.pkl` (or `.joblib`)
- `artifacts/demo_ensemble/model_seed*.pkl` (or `.joblib`)

If models are missing, the app UI will run but prediction calls will fail with model-loading errors.

## External receptor feature source

The predictor looks for receptor pocket files from `GPCRtryagain - Delete - Copy` by default.
On Streamlit Cloud, set an environment variable if you keep that dataset in a different path:

- `GPCR_DATA_ROOT=<absolute path to GPCRtryagain folder>`
