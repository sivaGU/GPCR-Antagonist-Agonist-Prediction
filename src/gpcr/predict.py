"""
GPCR Class A Functional Activity prediction module.

Predicts Agonist/Antagonist/Inactive for GPCR Class A receptor-ligand pairs.
Supports multi-class classification with uncertainty quantification.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, Dict

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs


@dataclass
class PredictResult:
    """Result of a single prediction."""
    is_valid: bool
    receptor: str
    ligand_smiles: str
    canonical_smiles: str
    predicted_class: str  # "Agonist", "Antagonist", or "Inactive"
    class_id: int  # 0=Agonist, 1=Antagonist, 2=Inactive
    prob_agonist: float
    prob_antagonist: float
    prob_inactive: float
    prob_std_error: Optional[float] = None  # Standard error of the mean probability
    prob_std_dev: Optional[float] = None  # Standard deviation of ensemble predictions
    threshold: Optional[float] = None
    error: str = ""


# Placeholder receptor features - replace with actual receptor feature extraction
# Based on manuscript: 31 receptor features + 14 interaction terms
RECEPTOR_FEATURES_DIM = 31
INTERACTION_TERMS_DIM = 14


def _get_receptor_features(receptor_name: str) -> Optional[np.ndarray]:
    """
    Extract receptor features for a given GPCR Class A receptor.
    
    TODO: Replace with actual receptor feature extraction from your pipeline.
    This should return 31 receptor features as a numpy array.
    """
    # Placeholder: return zeros for now
    # In production, this should load receptor features from a database or file
    return np.zeros(RECEPTOR_FEATURES_DIM, dtype=np.float32)


def _compute_ligand_features(smiles: str) -> Optional[np.ndarray]:
    """
    Compute ligand features: PhysChem (10) + ECFP4 (2048) = 2058 features.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Physicochemical descriptors (10)
    phys = np.array([
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        Descriptors.HeavyAtomCount(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
    ], dtype=np.float32)
    
    # ECFP4 fingerprint (2048 bits)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    
    return np.hstack([phys, arr])


def _compute_interaction_features(ligand_feats: np.ndarray, receptor_feats: np.ndarray) -> np.ndarray:
    """
    Compute interaction terms between ligand and receptor features.
    
    TODO: Replace with actual interaction feature computation from your pipeline.
    Based on manuscript: 14 interaction terms.
    """
    # Placeholder: simple element-wise products and differences
    # In production, implement the actual interaction feature computation
    interaction = np.hstack([
        ligand_feats[:7] * receptor_feats[:7],  # First 7 interaction terms
        ligand_feats[:7] - receptor_feats[:7],  # Next 7 interaction terms
    ])
    return interaction[:INTERACTION_TERMS_DIM].astype(np.float32)


def _canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES."""
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _compute_full_features(receptor_name: str, ligand_smiles: str) -> Optional[np.ndarray]:
    """
    Compute full feature vector: ligand features + receptor features + interaction terms.
    
    Expected dimensions:
    - Ligand: 2058 (10 PhysChem + 2048 ECFP4)
    - Receptor: 31
    - Interaction: 14
    - Total: 2103 features
    """
    ligand_feats = _compute_ligand_features(ligand_smiles)
    if ligand_feats is None:
        return None
    
    receptor_feats = _get_receptor_features(receptor_name)
    if receptor_feats is None:
        return None
    
    interaction_feats = _compute_interaction_features(ligand_feats, receptor_feats)
    
    return np.hstack([ligand_feats, receptor_feats, interaction_feats])


class GPCRPredictor:
    """Loaded predictor state (models, class names, threshold)."""

    def __init__(
        self,
        models: List,  # List of trained models (ensemble)
        class_names: List[str] = None,
        threshold: Optional[float] = None,
    ):
        self.models = models
        self.class_names = class_names or ["Agonist", "Antagonist", "Inactive"]
        self.threshold = threshold

    def predict(self, receptor: str, ligand_smiles: str) -> PredictResult:
        """Run full pipeline for one receptor-ligand pair."""
        canon = _canonicalize_smiles(ligand_smiles)
        if canon is None:
            return PredictResult(
                is_valid=False,
                receptor=receptor,
                ligand_smiles=ligand_smiles,
                canonical_smiles="",
                predicted_class="Unknown",
                class_id=-1,
                prob_agonist=0.0,
                prob_antagonist=0.0,
                prob_inactive=0.0,
                error="Invalid SMILES",
            )
        
        features = _compute_full_features(receptor, canon)
        if features is None:
            return PredictResult(
                is_valid=False,
                receptor=receptor,
                ligand_smiles=ligand_smiles,
                canonical_smiles=canon,
                predicted_class="Unknown",
                class_id=-1,
                prob_agonist=0.0,
                prob_antagonist=0.0,
                prob_inactive=0.0,
                error="Could not compute features",
            )
        
        X = features.reshape(1, -1)
        
        # Ensemble prediction
        all_probs = []
        for model in self.models:
            try:
                # Try predict_proba first (for sklearn models)
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X)[0]
                # Try predict with probability output (for LightGBM/XGBoost)
                elif hasattr(model, 'predict'):
                    # Some models return probabilities directly
                    probs = model.predict(X, raw_score=False)[0]
                    # Ensure 3 classes
                    if len(probs) != 3:
                        # If binary, convert to 3-class
                        probs = np.array([probs[0], probs[1], 0.0])
                else:
                    continue
                
                # Ensure 3 probabilities
                if len(probs) == 3:
                    all_probs.append(probs)
            except Exception as e:
                continue
        
        if not all_probs:
            return PredictResult(
                is_valid=False,
                receptor=receptor,
                ligand_smiles=ligand_smiles,
                canonical_smiles=canon,
                predicted_class="Unknown",
                class_id=-1,
                prob_agonist=0.0,
                prob_antagonist=0.0,
                prob_inactive=0.0,
                error="Model prediction failed",
            )
        
        # Average probabilities across ensemble
        mean_probs = np.mean(all_probs, axis=0)
        std_probs = np.std(all_probs, axis=0)
        
        # Standard error of the mean
        std_error = std_probs / np.sqrt(len(all_probs))
        
        prob_agonist = float(mean_probs[0])
        prob_antagonist = float(mean_probs[1])
        prob_inactive = float(mean_probs[2])
        
        # Predicted class (highest probability)
        predicted_class_id = int(np.argmax(mean_probs))
        predicted_class = self.class_names[predicted_class_id]
        
        return PredictResult(
            is_valid=True,
            receptor=receptor,
            ligand_smiles=ligand_smiles,
            canonical_smiles=canon,
            predicted_class=predicted_class,
            class_id=predicted_class_id,
            prob_agonist=prob_agonist,
            prob_antagonist=prob_antagonist,
            prob_inactive=prob_inactive,
            prob_std_error=float(std_error[predicted_class_id]),
            prob_std_dev=float(std_probs[predicted_class_id]),
            threshold=self.threshold,
            error="",
        )


def load_predictor(artifact_dir: Union[str, Path]) -> GPCRPredictor:
    """
    Load GPCR predictor from artifact directory.
    
    Expected structure:
    artifacts/
        model_seed0.pkl (or .joblib)
        model_seed1.pkl
        ...
        feature_config.json
        threshold.json (optional)
    """
    base = Path(artifact_dir)
    art = base / "artifacts"
    if not art.exists():
        art = base
    
    # Load models
    models = []
    model_files = list(art.glob("model_seed*.pkl")) + list(art.glob("model_seed*.joblib"))
    
    if not model_files:
        # Try alternative naming
        model_files = list(art.glob("*.pkl")) + list(art.glob("*.joblib"))
    
    for model_file in sorted(model_files):
        try:
            models.append(joblib.load(model_file))
        except Exception as e:
            print(f"Warning: Could not load {model_file}: {e}")
    
    if not models:
        raise FileNotFoundError(
            f"No model files found in {art}. "
            f"Expected: model_seed*.pkl or model_seed*.joblib"
        )
    
    # Load config
    class_names = ["Agonist", "Antagonist", "Inactive"]
    threshold = None
    
    config_path = art / "feature_config.json"
    if config_path.exists():
        import json
        with open(config_path, "r") as f:
            config = json.load(f)
            class_names = config.get("class_names", class_names)
    
    threshold_path = art / "threshold.json"
    if threshold_path.exists():
        import json
        with open(threshold_path, "r") as f:
            thresh_data = json.load(f)
            threshold = thresh_data.get("threshold", threshold)
    
    return GPCRPredictor(
        models=models,
        class_names=class_names,
        threshold=threshold,
    )


def predict_single(
    receptor: str,
    ligand_smiles: str,
    artifact_dir: Union[str, Path] = ".",
    predictor: Optional[GPCRPredictor] = None,
) -> PredictResult:
    """Predict for a single receptor-ligand pair."""
    if predictor is None:
        predictor = load_predictor(artifact_dir)
    return predictor.predict(receptor, ligand_smiles)


def predict_batch(
    receptor_ligand_pairs: List[tuple],  # List of (receptor, ligand_smiles) tuples
    artifact_dir: Union[str, Path] = ".",
    predictor: Optional[GPCRPredictor] = None,
) -> List[PredictResult]:
    """Predict for a list of receptor-ligand pairs."""
    if predictor is None:
        predictor = load_predictor(artifact_dir)
    return [predictor.predict(receptor, ligand) for receptor, ligand in receptor_ligand_pairs]
