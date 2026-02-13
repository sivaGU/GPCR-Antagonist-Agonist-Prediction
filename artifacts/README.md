# Artifacts Folder

**Current contents:** Dummy placeholder models (model_seed0.pkl, model_seed1.pkl, model_seed2.pkl)

These are minimal RandomForest models trained on random data so the app can run. **Replace them with your trained models** when ready.

## To replace with your models

1. Add your model files (e.g., `model_seed0.pkl`, `model_seed1.pkl`, ...)
2. Use the same naming: `model_seed{N}.pkl` or `model_seed{N}.joblib`
3. Models must accept 2103 features and output 3-class probabilities (Agonist, Antagonist, Inactive)

## To regenerate dummy placeholders

Run from project root:
```bash
python create_dummy_artifacts.py
```
