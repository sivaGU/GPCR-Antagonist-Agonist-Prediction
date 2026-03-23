"""
GPCR Class A Functional Activity Prediction Streamlit GUI.

Run from this folder (project root):
  streamlit run streamlit_app.py
"""
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root (this folder) is on path for src.gpcr
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HANDOFF_DIR = PROJECT_ROOT

import io
import urllib.request
import urllib.parse
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from src.gpcr.predict import (
    predict_single,
    predict_batch,
    load_predictor,
    get_available_receptors,
)

# Data paths for demo tool
DATA_DIR = PROJECT_ROOT / "data"
RECEPTORS_FILE = DATA_DIR / "gpcr_class_a_receptors.txt"
DEMO_REFERENCE_FILE = DATA_DIR / "demo_reference.csv"


def extract_smiles_from_file(file_content: bytes, file_extension: str) -> Optional[str]:
    """
    Extract SMILES string from various molecular file formats.
    Supported formats: SDF, PDB, PDBQT, MOL, MOL2, CSV (first row only).
    """
    try:
        ext = file_extension.lower()
        if ext == ".sdf":
            from io import StringIO
            sdf_data = StringIO(file_content.decode("utf-8"))
            supplier = Chem.SDMolSupplier(sdf_data)
            for m in supplier:
                if m is not None:
                    return Chem.MolToSmiles(m, canonical=True)
        elif ext == ".mol":
            mol = Chem.MolFromMolBlock(file_content.decode("utf-8"))
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        elif ext == ".pdb":
            mol = Chem.MolFromPDBBlock(file_content.decode("utf-8"))
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
            lines = file_content.decode("utf-8").split("\n")
            for line in lines:
                if "SMILES" in line.upper():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "SMILES" in part.upper() and i + 1 < len(parts):
                            potential = parts[i + 1]
                            mol = Chem.MolFromSmiles(potential)
                            if mol:
                                return Chem.MolToSmiles(mol, canonical=True)
        elif ext == ".pdbqt":
            mol = Chem.MolFromPDBBlock(file_content.decode("utf-8"))
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
            lines = file_content.decode("utf-8").split("\n")
            for line in lines:
                if "SMILES" in line.upper():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "SMILES" in part.upper() and i + 1 < len(parts):
                            potential = parts[i + 1]
                            mol = Chem.MolFromSmiles(potential)
                            if mol:
                                return Chem.MolToSmiles(mol, canonical=True)
        elif ext == ".mol2":
            try:
                mol = Chem.MolFromMol2Block(file_content.decode("utf-8"))
                if mol:
                    return Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                pass
        elif ext == ".csv":
            from io import BytesIO
            df = pd.read_csv(BytesIO(file_content))
            col = next((c for c in df.columns if c.lower() in ("smiles", "smi") or c == "SMILES"), None)
            if col and len(df) > 0:
                return str(df[col].iloc[0]).strip()
    except Exception:
        pass
    return None


def fetch_structure_image_from_database(smiles: str, width: int = 400, height: int = 400) -> Optional[bytes]:
    """
    Fetch a 2D structure image for the given SMILES from the NCI CACTUS
    Chemical Identifier Resolver. Returns PNG image bytes or None on failure.
    """
    if not smiles or not str(smiles).strip():
        return None
    try:
        encoded = urllib.parse.quote(str(smiles).strip(), safe="")
        url = (
            f"https://cactus.nci.nih.gov/chemical/structure/{encoded}/image"
            f"?width={width}&height={height}&format=png"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "GPCR-GUI/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                return None
            data = resp.read()
            if not data or len(data) < 100:
                return None
            return data
    except Exception:
        return None


def get_mol_for_drawing(smiles: Optional[str] = None, file_content: Optional[bytes] = None, file_extension: Optional[str] = None):
    """
    Get an RDKit mol for 2D structure drawing (no 3D embedding required).
    Uses uploaded file if present, else SMILES. Returns Chem.Mol or None.
    """
    mol = None
    if file_content is not None and file_extension is not None:
        ext = file_extension.lower()
        try:
            text = file_content.decode("utf-8")
            if ext == ".sdf":
                from io import StringIO
                supplier = Chem.SDMolSupplier(StringIO(text))
                mols = [m for m in supplier if m is not None]
                mol = mols[0] if mols else None
            elif ext == ".mol":
                mol = Chem.MolFromMolBlock(text)
            elif ext in (".pdb", ".pdbqt"):
                mol = Chem.MolFromPDBBlock(text)
            elif ext == ".mol2":
                mol = Chem.MolFromMol2Block(text)
        except Exception:
            mol = None
    if mol is None and smiles is not None:
        smiles_str = str(smiles).strip()
        if smiles_str:
            mol = Chem.MolFromSmiles(smiles_str)
    return mol


def render_ligand_structure(mol, size: int = 400) -> Optional[bytes]:
    """
    Draw the ligand as a 2D chemical structure (atoms and bonds) using RDKit.
    Returns PNG image bytes or None on failure. Used as fallback when database lookup fails.
    """
    if mol is None:
        return None
    try:
        from rdkit.Chem import Draw
        try:
            AllChem.Compute2DCoords(mol)
        except Exception:
            pass
        img = Draw.MolToImage(mol, size=(size, size))
        if img is None:
            return None
        buf = io.BytesIO()
        if hasattr(img, "mode") and img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None


# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="GPCR Class A Functional Activity Prediction",
    page_icon=None,
    layout="wide",
    menu_items={
        "About": "GPCR Class A Functional Activity Prediction GUI - Predicts Agonist/Antagonist/Inactive for receptor-ligand pairs.",
    },
)

# Inject custom CSS - clean dark hero + card layout
st.markdown("""
<style>
    :root { --bg: #f5f8fc; --card: #ffffff; --ink: #0f172a; --brand: #0891b2; --brand2: #2563eb; }
    .stApp { background: var(--bg); color: var(--ink); }
    .main .block-container,
    section.main .block-container {
        background: transparent !important;
        padding: 1.5rem 2rem 3rem 2rem;
        max-width: 1300px;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        color: #f8fafc;
        width: 300px !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li, [data-testid="stSidebar"] h3 { color: #e2e8f0 !important; }
    .stButton > button {
        background: linear-gradient(90deg, var(--brand), var(--brand2));
        color: #fff;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.2);
    }
    .hero {
        background: radial-gradient(circle at 20% 20%, #1d4ed8 0%, #0f172a 60%);
        color: #e2e8f0;
        padding: 1.6rem 1.8rem;
        border-radius: 16px;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 24px rgba(2, 6, 23, 0.25);
    }
    .hero h2 { color: #f8fafc; margin-bottom: 0.4rem; }
    .hero p { margin: 0; color: #cbd5e1; }
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.06);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PREDICTOR (cached)
# ============================================================================

@st.cache_resource
def get_predictor(model_type: Optional[str] = None):
    """Load predictor; for demo tool pass model_type in ('rf', 'lightgbm', 'xgboost')."""
    try:
        return load_predictor(HANDOFF_DIR, model_type=model_type)
    except TypeError:
        # Backward compatibility: older load_predictor() may not accept model_type
        return load_predictor(HANDOFF_DIR)

# ============================================================================
# PAGES
# ============================================================================

def render_home_page():
    """Render the home/dashboard page."""
    st.markdown(
        """
        <div class="hero">
          <h2>GPCR Class A Functional Activity</h2>
          <p>Manuscript-aligned ligand + receptor + interaction feature inference with a streamlined screening workflow.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.title("GPCR Class A Functional Activity Prediction")
    st.caption(
        "Machine learning-based prediction of Agonist/Antagonist/Inactive activity for GPCR Class A receptor-ligand pairs."
    )

    st.sidebar.markdown("### Project Snapshot")
    st.sidebar.markdown(
        """
        - **Model focus:** GPCR Class A functional activity
        - **Classes:** Agonist, Antagonist, Inactive
        - **Features:** Ligand (PhysChem + ECFP) + Receptor (31) + Interaction (14)
        - **Models:** LightGBM, Random Forest, XGBoost (ensemble)
        - **Status:** Ready for ML artifact upload
        """
    )
    st.sidebar.success("Upload your trained models to get started!")

    st.markdown(
        """
        ## Why this app exists
        Drug discovery teams need to predict the functional activity of ligands binding to GPCR Class A receptors.
        This GUI provides a user-friendly interface for predicting whether a ligand acts as an **Agonist**, **Antagonist**, 
        or is **Inactive** for a given GPCR Class A receptor. The model uses machine learning approaches including
        LightGBM, Random Forest, and XGBoost to make predictions with uncertainty quantification.
        """
    )

    st.markdown(
        """
        ### Model highlights
        - **Multi-class classification:** Predicts Agonist (class 0), Antagonist (class 1), or Inactive (class 2)
        - **Feature engineering:** Combines ligand physicochemical properties, ECFP fingerprints, receptor features, and interaction terms
        - **Ensemble support:** Works with multiple model seeds for robust predictions
        - **Uncertainty quantification:** Provides error probabilities and confidence intervals
        - **Evaluation regimes:** Supports baseline, random stratified, scaffold split, and LORO (Leave-One-Receptor-Out) evaluation
        """
    )

    st.divider()

    st.markdown("## Quick start")

    st.info(
        "**Ready to predict!** Use **Demo Prediction Tool** to compare predictions to experimental values (Agonist/Antagonist/Inactive), "
        "or **GPCR Ligand Functional Activity Prediction** for single/batch predictions."
    )

    st.markdown(
        """
        ---
        ### Navigation
        - **Home:** This overview
        - **Documentation:** Setup, model details, and usage
        - **Demo Prediction Tool:** Predicted vs experimental comparison table (RF/LightGBM/XGBoost/Ensemble)
        - **GPCR Ligand Functional Activity Prediction:** Run predictions (receptor + ligand)
        """
    )


def render_documentation_page():
    """Render the documentation page."""
    st.title("Documentation & Runbook")
    st.caption("Reference material for the GPCR Class A Functional Activity Prediction GUI.")

    st.markdown(
        """
        ## Purpose
        This application provides a Streamlit interface for predicting GPCR Class A receptor-ligand functional activity.
        It supports single predictions (receptor name + ligand SMILES/structure file) and batch CSV processing.
        """
    )

    st.markdown(
        """
        ## Repository structure
        ```
        .
        ├── streamlit_app.py       # Main application
        ├── requirements.txt      # Dependencies
        ├── src/gpcr/             # Prediction module
        │   ├── predict.py        # predict_single, predict_batch, load_predictor
        │   └── cli.py           # Command-line interface
        └── artifacts/            # Model artifacts (add your models here)
            ├── model_seed0.pkl (or .joblib)
            ├── model_seed1.pkl
            ├── ...
            ├── feature_config.json
            └── threshold.json (optional)
        ```
        """
    )

    st.markdown(
        """
        ## Local setup
        1. Create and activate a virtual environment (conda, venv, or poetry).
        2. Install dependencies: `pip install -r requirements.txt`.
        3. Ensure `GPCRtryagain - Delete - Copy` is available (or set `GPCR_DATA_ROOT`).
        4. **Add your trained ML models** to the `artifacts/` folder (see below).
        5. Launch the app: `streamlit run streamlit_app.py`.
        6. Streamlit will open at `http://localhost:8501`. Use the sidebar to switch between pages.
        """
    )

    st.markdown(
        """
        ## Model overview
        - **Classes:** Agonist (0), Antagonist (1), Inactive (2)
        - **Features:** 
          - Ligand: 10 physicochemical descriptors + 2048-bit ECFP4 = 2058 features
          - Receptor: 31 pocket features from `Josh_Receptor_Features`
          - Interaction: 14 manuscript-style ligand × receptor terms
          - **Total: 2103 features**
        - **Models:** Ensemble of LightGBM, Random Forest, or XGBoost models
        - **Evaluation:** Baseline, Random Stratified, Scaffold Split, LORO
        """
    )

    st.markdown(
        """
        ## Adding your ML artifacts
        
        Place your trained model files in the `artifacts/` folder:
        - `model_seed0.pkl` (or `.joblib`)
        - `model_seed1.pkl`
        - `model_seed2.pkl`
        - ... (as many seeds as you have)
        
        Optionally create:
        - `feature_config.json`: Feature configuration (class names, etc.)
        - `threshold.json`: Classification thresholds (if applicable)
        """
    )

    st.markdown(
        """
        ## Uncertainty Quantification
        The model provides uncertainty estimates for each prediction:
        - **Standard Error:** Calculated from the variance across the ensemble models (std_dev / √n).
        - **95% Confidence Interval:** Probability ± 2×SE, providing a range within which the true probability likely falls.
        - **Display:** Single predictions show probability distributions and confidence intervals. Batch CSV outputs include columns for standard error and CI bounds.
        """
    )

    st.markdown(
        """
        ## CLI usage
        From the project folder:
        ```bash
        python -m src.gpcr.cli --receptor "ADRB2" --ligand "CCO" --output out.csv
        python -m src.gpcr.cli --input example_inputs.csv --output out.csv
        ```
        Output columns: receptor, ligand_smiles, canonical_smiles, predicted_class, class_id, prob_agonist, prob_antagonist, prob_inactive, prob_std_error, error.
        """
    )

    st.success("Questions? Refer to the ML GPCR Class A Functional Activity Manuscript for model details.")


def _load_receptor_list():
    """Load receptor names from GPCRtryagain features folder, fallback to local text file."""
    external = get_available_receptors()
    if external:
        return external
    if not RECEPTORS_FILE.exists():
        return []
    try:
        lines = RECEPTORS_FILE.read_text(encoding="utf-8").strip().splitlines()
        return [s.strip() for s in lines if s.strip()]
    except Exception:
        return []


def _load_demo_reference():
    """Load demo reference data (receptor, ligand, experimental_class) for comparison table."""
    if not DEMO_REFERENCE_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(DEMO_REFERENCE_FILE, encoding="utf-8")
    except pd.errors.ParserError:
        df = pd.read_csv(DEMO_REFERENCE_FILE, encoding="utf-8", engine="python", on_bad_lines="skip")
    return df


def render_demo_prediction_page():
    """Render the Demo Prediction Tool page: predicted vs experimental comparison table."""
    st.title("Demo Prediction Tool")
    st.caption(
        "Compare model predictions to experimental values (Agonist / Antagonist / Inactive) "
        "using Random Forest, LightGBM, XGBoost, or Ensemble."
    )

    ref_df = _load_demo_reference()
    if ref_df.empty or "smiles" not in ref_df.columns or "experimental_class" not in ref_df.columns:
        st.warning(
            "Demo reference data not found or missing columns. Add data/demo_reference.csv with columns: "
            "receptor, name, smiles, experimental_class (Agonist/Antagonist/Inactive)."
        )
        return

    st.sidebar.markdown("### Demo settings")
    model_type_label = st.sidebar.selectbox(
        "Model",
        options=["Random Forest", "LightGBM", "XGBoost", "Ensemble"],
        index=0,
        key="demo_model",
    )
    model_type_map = {"Random Forest": "rf", "LightGBM": "lightgbm", "XGBoost": "xgboost", "Ensemble": "ensemble"}
    model_type = model_type_map[model_type_label]

    try:
        predictor = get_predictor(model_type)
    except Exception as e:
        st.error(f"Could not load {model_type_label} model: {e}")
        st.info(
            "Ensure artifacts/demo_rf, demo_lightgbm, demo_xgboost, and/or demo_ensemble exist with model_seed*.pkl."
        )
        return

    # Run predictions for all reference rows
    pairs = [(str(row["receptor"]), str(row["smiles"])) for _, row in ref_df.iterrows()]
    with st.spinner(f"Running {model_type_label} on {len(ref_df)} reference compounds..."):
        results = predict_batch(pairs, predictor=predictor)

    # Build comparison table: experimental vs predicted
    out = ref_df[["receptor", "name", "smiles", "experimental_class"]].copy()
    out["predicted_class"] = [r.predicted_class for r in results]
    out["P(Agonist)"] = [round(r.prob_agonist, 4) for r in results]
    out["P(Antagonist)"] = [round(r.prob_antagonist, 4) for r in results]
    out["P(Inactive)"] = [round(r.prob_inactive, 4) for r in results]
    out["match"] = [
        "✓" if str(row["experimental_class"]).strip().lower() == str(row["predicted_class"]).strip().lower() else "✗"
        for _, row in out.iterrows()
    ]
    out = out.rename(columns={"match": "Match"})

    st.markdown(f"**Model:** {model_type_label} · **Reference compounds:** {len(ref_df)}")

    # Summary metrics
    n_match = out["Match"].eq("✓").sum()
    accuracy = n_match / len(out) * 100 if len(out) else 0
    st.metric("Agreement with experiment", f"{n_match} / {len(out)} ({accuracy:.1f}%)")

    st.subheader("Predicted vs experimental")
    st.dataframe(
        out[
            [
                "receptor",
                "name",
                "experimental_class",
                "predicted_class",
                "P(Agonist)",
                "P(Antagonist)",
                "P(Inactive)",
                "Match",
            ]
        ],
        use_container_width=True,
        height=400,
    )
    st.download_button(
        "Download comparison (CSV)",
        out.to_csv(index=False),
        f"demo_predicted_vs_experimental_{model_type}.csv",
        "text/csv",
        key="demo_download",
    )


def render_gpcr_prediction_page():
    """Render the GPCR Ligand Functional Activity Prediction page."""
    st.markdown(
        """
        <div class="hero">
          <h2>Screen Ligands Against Class A GPCRs</h2>
          <p>Select a receptor, provide SMILES or upload structural files, and get Agonist/Antagonist/Inactive probabilities.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.title("GPCR Ligand Functional Activity Prediction")
    st.markdown(
        """
        Predict GPCR Class A receptor-ligand functional activity. Choose a model, select a receptor and provide a ligand (SMILES or structure file),
        or upload a CSV file for batch processing. The model outputs probabilities for Agonist, Antagonist, and Inactive classes.
        
        **Input modes:** Single receptor-ligand pair | Batch (CSV with receptor and ligand columns)
        """
    )

    st.markdown("#### Select model")
    model_type_label = st.selectbox(
        "Model type",
        ["Random Forest", "LightGBM", "XGBoost", "Ensemble"],
        key="gpcr_pred_model",
        help="Select which trained model to use for predictions.",
    )
    st.caption("Choose **Random Forest**, **LightGBM**, **XGBoost**, or **Ensemble** for predictions.")
    model_type_map = {"Random Forest": "rf", "LightGBM": "lightgbm", "XGBoost": "xgboost", "Ensemble": "ensemble"}
    model_type = model_type_map[model_type_label]

    try:
        predictor = get_predictor(model_type)
    except Exception as e:
        st.error(f"Could not load {model_type_label} model: {e}")
        st.info(
            "Ensure the **artifacts/** folder contains demo subfolders (e.g. **artifacts/demo_rf/**) with:\n"
            "- Model files (model_seed*.pkl or model_seed*.joblib)\n"
            "- feature_config.json\n"
            "- threshold.json"
        )
        return

    st.sidebar.markdown("### Model Info")
    st.sidebar.info(
        f"**Model:** {model_type_label}\n\n"
        f"**Loaded:** {len(predictor.models)} model(s)\n\n"
        f"**Classes:** {', '.join(predictor.class_names)}"
    )

    st.divider()

    input_mode = st.radio(
        "Input mode",
        ["Single receptor-ligand pair", "Batch (CSV)"],
        horizontal=True,
        key="input_mode",
    )

    if input_mode == "Single receptor-ligand pair":
        receptors = _load_receptor_list()
        if not receptors:
            st.warning("GPCR Class A receptor list not found (data/gpcr_class_a_receptors.txt). Add it to enable receptor selection.")
        receptor_options = ["Select receptor..."] + (receptors if receptors else [])
        receptor_input = st.selectbox(
            "GPCR Class A Receptor",
            options=receptor_options,
            key="receptor_input",
            help="Select the receptor to predict functional activity for.",
        )
        receptor_selected = receptor_input if (receptor_input and receptor_input != "Select receptor...") else ""

        ligand_input = st.text_input(
            "Ligand SMILES (or upload a structure file below)",
            placeholder="e.g. CCO, c1ccccc1",
            key="ligand_input",
        )
        
        st.markdown("**Or upload a ligand structure file:**")
        structure_file = st.file_uploader(
            "Upload ligand structure file",
            type=["sdf", "mol", "pdb", "pdbqt", "mol2", "csv"],
            key="structure_upload",
            help="Supported: SDF, MOL, PDB, PDBQT, MOL2, CSV (first smiles row).",
        )
        
        ligand_to_use = None
        if structure_file:
            content = structure_file.read()
            ext = os.path.splitext(structure_file.name)[1]
            extracted = extract_smiles_from_file(content, ext)
            if extracted:
                ligand_to_use = extracted
                st.session_state.structure_file_content = content
                st.session_state.structure_file_ext = ext
                st.success(f"Extracted SMILES from {structure_file.name}")
            else:
                st.session_state.structure_file_content = None
                st.session_state.structure_file_ext = None
                st.error(f"Could not extract SMILES from {ext.upper()} file. Try SMILES input instead.")
        elif ligand_input and ligand_input.strip():
            ligand_to_use = ligand_input.strip()
            st.session_state.structure_file_content = None
            st.session_state.structure_file_ext = None
        
        if st.button("Predict", type="primary", key="btn_single"):
            if receptor_selected and ligand_to_use:
                result = predict_single(
                    receptor_selected,
                    ligand_to_use,
                    predictor=predictor,
                )
                if result.is_valid:
                    st.success("Valid input")
                    
                    # Main prediction
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Class", result.predicted_class)
                    with col2:
                        st.metric("Receptor", result.receptor)
                    with col3:
                        st.metric("Class ID", result.class_id)
                    
                    # Probability distributions
                    st.subheader("Probability Distributions")
                    prob_col1, prob_col2, prob_col3 = st.columns(3)
                    with prob_col1:
                        st.metric("P(Agonist)", f"{result.prob_agonist:.4f}")
                    with prob_col2:
                        st.metric("P(Antagonist)", f"{result.prob_antagonist:.4f}")
                    with prob_col3:
                        st.metric("P(Inactive)", f"{result.prob_inactive:.4f}")
                    
                    # Visualize probabilities
                    import plotly.graph_objects as go
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Agonist', 'Antagonist', 'Inactive'],
                            y=[result.prob_agonist, result.prob_antagonist, result.prob_inactive],
                            marker_color=['#7E57C2', '#673AB7', '#512DA8'],
                            text=[f'{result.prob_agonist:.3f}', f'{result.prob_antagonist:.3f}', f'{result.prob_inactive:.3f}'],
                            textposition='auto',
                        )
                    ])
                    fig.update_layout(
                        title="Class Probability Distribution",
                        xaxis_title="Class",
                        yaxis_title="Probability",
                        yaxis_range=[0, 1],
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Uncertainty analysis
                    if result.prob_std_error is not None:
                        st.markdown("#### Uncertainty Analysis")
                        err_col1, err_col2, err_col3 = st.columns(3)
                        with err_col1:
                            error_pct = result.prob_std_error * 100
                            st.metric("Standard Error", f"± {error_pct:.2f}%")
                        with err_col2:
                            prob_max = max(result.prob_agonist, result.prob_antagonist, result.prob_inactive)
                            ci_lower = max(0.0, prob_max - 2 * result.prob_std_error)
                            st.metric("95% CI Lower", f"{ci_lower:.4f}")
                        with err_col3:
                            ci_upper = min(1.0, prob_max + 2 * result.prob_std_error)
                            st.metric("95% CI Upper", f"{ci_upper:.4f}")
                        st.info(
                            f"**Prediction Range:** Highest probability = {prob_max:.4f} ± {result.prob_std_error:.4f} "
                            f"(95% confidence interval: [{ci_lower:.4f}, {ci_upper:.4f}])"
                        )
                    
                    # Ligand structure visualization
                    st.subheader("Ligand Structure")
                    smiles_for_lookup = (result.canonical_smiles or result.ligand_smiles or "").strip()
                    img_bytes = fetch_structure_image_from_database(smiles_for_lookup) if smiles_for_lookup else None
                    source_label = "NCI CACTUS Chemical Structure Resolver"
                    if img_bytes is None:
                        file_content = st.session_state.get("structure_file_content")
                        file_ext = st.session_state.get("structure_file_ext")
                        mol = get_mol_for_drawing(
                            smiles_for_lookup if smiles_for_lookup else None,
                            file_content=file_content,
                            file_extension=file_ext,
                        )
                        img_bytes = render_ligand_structure(mol) if mol else None
                        source_label = "RDKit (database lookup unavailable)"
                    if img_bytes:
                        st.image(io.BytesIO(img_bytes), use_container_width=False, width=400)
                        st.caption(f"2D structure · Source: {source_label}")
                    else:
                        st.warning(
                            "Could not retrieve or draw structure for this molecule."
                            + (f" (SMILES: {result.canonical_smiles})" if result.canonical_smiles else "")
                        )
                else:
                    st.error(result.error)
            else:
                st.warning("Please select a GPCR Class A receptor and provide ligand SMILES or upload a structure file.")

    else:
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            key="csv_upload",
        )
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            receptor_col = next(
                (c for c in df.columns if c.lower() in ("receptor", "receptor_name", "gpcr")),
                None
            )
            ligand_col = next(
                (c for c in df.columns if c.lower() in ("ligand", "smiles", "canonical_smiles", "smi")),
                None
            )
            if receptor_col is None:
                st.error("CSV must have a 'receptor' column.")
                st.info(f"Available columns: {', '.join(df.columns)}")
            elif ligand_col is None:
                st.error("CSV must have a 'ligand' or 'smiles' column.")
                st.info(f"Available columns: {', '.join(df.columns)}")
            else:
                if st.button("Predict batch", type="primary", key="btn_batch"):
                    pairs = list(zip(df[receptor_col].astype(str), df[ligand_col].astype(str)))
                    results = predict_batch(pairs, predictor=predictor)
                    
                    df_out = df.copy()
                    df_out["predicted_class"] = [r.predicted_class for r in results]
                    df_out["class_id"] = [r.class_id for r in results]
                    df_out["prob_agonist"] = [r.prob_agonist for r in results]
                    df_out["prob_antagonist"] = [r.prob_antagonist for r in results]
                    df_out["prob_inactive"] = [r.prob_inactive for r in results]
                    df_out["prob_std_error"] = [
                        f"{r.prob_std_error:.6f}" if r.prob_std_error is not None else ""
                        for r in results
                    ]
                    df_out["prob_std_error_pct"] = [
                        f"{r.prob_std_error * 100:.2f}%" if r.prob_std_error is not None else ""
                        for r in results]
                    df_out["canonical_smiles"] = [r.canonical_smiles for r in results]
                    df_out["error"] = [r.error for r in results]

                    st.subheader("Results")
                    st.dataframe(df_out, use_container_width=True)

                    st.subheader("Download results")
                    st.download_button(
                        "Download CSV",
                        df_out.to_csv(index=False),
                        "gpcr_predictions.csv",
                        "text/csv",
                        key="download_csv",
                    )
        else:
            st.info("Upload a CSV file with 'receptor' and 'ligand' (or 'smiles') columns to run batch predictions.")

    st.divider()
    st.caption(
        "GPCR Class A Functional Activity Prediction. Multi-class classification: Agonist/Antagonist/Inactive."
    )


# ============================================================================
# MAIN - NAVIGATION
# ============================================================================

def main():
    """Main app entry point with navigation."""
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"

    st.sidebar.markdown("### Navigation")
    st.sidebar.markdown("")

    if st.sidebar.button("Home", use_container_width=True, key="nav_home"):
        st.session_state.current_page = "Home"

    if st.sidebar.button("Documentation", use_container_width=True, key="nav_docs"):
        st.session_state.current_page = "Documentation"

    if st.sidebar.button("Demo Prediction Tool", use_container_width=True, key="nav_demo"):
        st.session_state.current_page = "Demo Prediction Tool"

    if st.sidebar.button("GPCR Ligand Functional Activity Prediction", use_container_width=True, key="nav_prediction"):
        st.session_state.current_page = "GPCR Ligand Functional Activity Prediction"

    st.sidebar.markdown("---")

    if st.session_state.current_page == "Home":
        render_home_page()
    elif st.session_state.current_page == "Documentation":
        render_documentation_page()
    elif st.session_state.current_page == "Demo Prediction Tool":
        render_demo_prediction_page()
    elif st.session_state.current_page == "GPCR Ligand Functional Activity Prediction":
        render_gpcr_prediction_page()


if __name__ == "__main__":
    main()
