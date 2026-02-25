"""
Create dummy placeholder model artifacts for the GPCR GUI.
Run this script to generate placeholder models that allow the app to run.
Replace these with your trained models later.
"""
import joblib
import numpy as np
import shutil
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# Feature dimensions (must match predict.py)
N_FEATURES = 2103
N_CLASSES = 3  # Agonist, Antagonist, Inactive
N_MODELS = 3   # Dummy ensemble of 3 models

def main():
    artifacts_dir = Path(__file__).parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # Create dummy training data (random) and fit minimal models
    np.random.seed(42)
    X_dummy = np.random.randn(100, N_FEATURES).astype(np.float32)
    y_dummy = np.random.randint(0, N_CLASSES, size=100)

    model_files = []
    for i in range(N_MODELS):
        model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42 + i)
        model.fit(X_dummy, y_dummy)
        out_path = artifacts_dir / f"model_seed{i}.pkl"
        joblib.dump(model, out_path)
        model_files.append(out_path)
        print(f"Created {out_path}")

    # Demo tool: copy same dummies into demo_rf, demo_lightgbm, demo_xgboost
    for sub in ("demo_rf", "demo_lightgbm", "demo_xgboost"):
        sub_dir = artifacts_dir / sub
        sub_dir.mkdir(exist_ok=True)
        for src in model_files:
            shutil.copy2(src, sub_dir / src.name)
        print(f"Created {sub_dir} with {N_MODELS} models")

    print(f"\nDone! Created {N_MODELS} dummy model placeholders in {artifacts_dir}")
    print("Demo folders: demo_rf, demo_lightgbm, demo_xgboost")
    print("Replace these with your trained models when ready.")

if __name__ == "__main__":
    main()
