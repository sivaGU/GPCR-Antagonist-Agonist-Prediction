# GPCR GUI Setup Guide

## What Has Been Created

A complete Streamlit GUI framework for GPCR Class A Functional Activity Prediction, similar to the MechBBB GUI structure. The framework is ready for you to add your trained ML models.

## Project Structure

```
GPCR Antagonist-Agonist GUI/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── SETUP_GUIDE.md           # This file
├── src/
│   ├── __init__.py
│   └── gpcr/
│       ├── __init__.py
│       ├── predict.py        # Prediction module (needs your feature extraction)
│       └── cli.py            # Command-line interface
├── artifacts/                # Place your ML models here
│   ├── feature_config.json   # Feature configuration
│   └── threshold.json        # Threshold config (optional)
├── example_inputs.csv       # Example input CSV
└── example_outputs.csv       # Example output CSV format
```

## Next Steps

### 1. Add Your Trained Models

Place your trained model files in the `artifacts/` folder:
- `model_seed0.pkl` (or `.joblib`)
- `model_seed1.pkl`
- `model_seed2.pkl`
- ... (as many seeds as you have)

The models should be saved using `joblib.dump()` or compatible format.

### 2. Implement Receptor Feature Extraction

Edit `src/gpcr/predict.py` and implement the `_get_receptor_features()` function:

```python
def _get_receptor_features(receptor_name: str) -> Optional[np.ndarray]:
    """
    Extract receptor features for a given GPCR Class A receptor.
    
    TODO: Replace with actual receptor feature extraction from your pipeline.
    This should return 31 receptor features as a numpy array.
    """
    # TODO: Load receptor features from your database/file
    # Example:
    # receptor_db = load_receptor_database()
    # features = receptor_db.get_features(receptor_name)
    # return features.astype(np.float32)
    
    # Placeholder: return zeros for now
    return np.zeros(RECEPTOR_FEATURES_DIM, dtype=np.float32)
```

### 3. Implement Interaction Feature Computation

Edit `src/gpcr/predict.py` and implement the `_compute_interaction_features()` function:

```python
def _compute_interaction_features(ligand_feats: np.ndarray, receptor_feats: np.ndarray) -> np.ndarray:
    """
    Compute interaction terms between ligand and receptor features.
    
    TODO: Replace with actual interaction feature computation from your pipeline.
    Based on manuscript: 14 interaction terms.
    """
    # TODO: Implement your interaction feature computation
    # Example: element-wise products, differences, etc.
    # return computed_interaction_features.astype(np.float32)
    
    # Placeholder: simple element-wise products
    interaction = np.hstack([
        ligand_feats[:7] * receptor_feats[:7],
        ligand_feats[:7] - receptor_feats[:7],
    ])
    return interaction[:INTERACTION_TERMS_DIM].astype(np.float32)
```

### 4. Verify Feature Dimensions

Ensure your feature extraction matches:
- **Ligand features:** 2058 (10 PhysChem + 2048 ECFP4) ✓ Already implemented
- **Receptor features:** 31 (implement in `_get_receptor_features()`)
- **Interaction features:** 14 (implement in `_compute_interaction_features()`)
- **Total:** 2103 features

### 5. Test the GUI

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

3. Test with a single prediction:
   - Enter a receptor name (e.g., "ADRB2")
   - Enter a ligand SMILES (e.g., "CCO")
   - Click "Predict"

4. Test batch prediction:
   - Upload `example_inputs.csv`
   - Click "Predict batch"
   - Download results

## Model Compatibility

The GUI supports models that:
- Have a `predict_proba()` method returning 3-class probabilities (for sklearn models)
- Have a `predict()` method returning probabilities (for LightGBM/XGBoost)
- Return probabilities as numpy arrays with shape `(n_samples, 3)`

If your models use a different API, modify the `GPCRPredictor.predict()` method in `src/gpcr/predict.py`.

## Features Implemented

✅ Multi-class classification (Agonist/Antagonist/Inactive)
✅ Ensemble model support
✅ Uncertainty quantification (standard error, confidence intervals)
✅ Single and batch prediction modes
✅ Structure file upload support (SDF, MOL, PDB, PDBQT, MOL2)
✅ Ligand structure visualization
✅ Probability distribution visualization (plotly charts)
✅ CLI interface for command-line usage
✅ Error handling and validation

## Notes

- The GUI uses a purple/violet color scheme to distinguish it from MechBBB (blue/teal)
- All prediction results include error probabilities and confidence intervals
- The framework is designed to work with your evaluation scripts (baseline, random stratified, scaffold split, LORO)
- Class encoding: Agonist=0, Antagonist=1, Inactive=2 (as per your evaluation scripts)

## Troubleshooting

**Error: "No model files found"**
- Ensure model files are in `artifacts/` folder
- Check file naming: `model_seed*.pkl` or `model_seed*.joblib`

**Error: "Could not compute features"**
- Implement `_get_receptor_features()` and `_compute_interaction_features()`
- Verify receptor names match your database

**Error: "Model prediction failed"**
- Check model file format (should be joblib/pickle)
- Verify model API matches expected interface
- Check feature dimensions match model expectations

## Contact

Refer to your ML GPCR Class A Functional Activity Manuscript for model details and evaluation procedures.
