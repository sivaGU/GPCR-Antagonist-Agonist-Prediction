"""
Slim the artifacts folder to stay under ~25 MB (e.g. for GitHub or deployment):
1. Keep only model_seed0.pkl in each folder (remove model_seed1, model_seed2).
2. Re-save each .pkl with joblib compression to reduce file size.

Run from project root: python slim_artifacts.py
"""
from pathlib import Path
import joblib

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
COMPRESS = 3  # 1-9; higher = smaller files, slower load


def slim_dir(dir_path: Path) -> None:
    """Keep only model_seed0.pkl and re-save with compression."""
    pkl_files = list(dir_path.glob("model_seed*.pkl")) + list(dir_path.glob("model_seed*.joblib"))
    if not pkl_files:
        return

    # Remove extra seeds (keep only seed0)
    for f in pkl_files:
        if "seed0" not in f.name:
            print(f"  Removing {f.relative_to(ARTIFACTS_DIR)}")
            f.unlink()

    seed0 = dir_path / "model_seed0.pkl"
    if not seed0.exists():
        seed0 = dir_path / "model_seed0.joblib"
    if not seed0.exists():
        return

    print(f"  Compressing {seed0.relative_to(ARTIFACTS_DIR)} ...")
    try:
        model = joblib.load(seed0)
    except Exception as e:
        print(f"  Skipping compression (load failed: {e})")
        return
    try:
        tmp = seed0.with_suffix(seed0.suffix + ".tmp")
        joblib.dump(model, tmp, compress=COMPRESS)
        tmp.replace(seed0)
        print(f"  Done. New size: {seed0.stat().st_size / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"  Skipping compression (dump failed: {e})")


def main():
    if not ARTIFACTS_DIR.exists():
        print("artifacts/ not found")
        return

    # Root and each demo_* subdir
    dirs_to_slim = [ARTIFACTS_DIR] + [d for d in ARTIFACTS_DIR.iterdir() if d.is_dir() and d.name.startswith("demo_")]
    for d in dirs_to_slim:
        if list(d.glob("model_seed*.pkl")) or list(d.glob("model_seed*.joblib")):
            print(f"Slimming {d.relative_to(ARTIFACTS_DIR.parent)}")
            slim_dir(d)

    total_mb = sum(f.stat().st_size for f in ARTIFACTS_DIR.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"\nTotal artifacts size: {total_mb:.2f} MB")


if __name__ == "__main__":
    main()
