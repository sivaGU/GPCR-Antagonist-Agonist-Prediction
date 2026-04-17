"""
GPCR receptor + query-ligand 3D viewer (py3Dmol), styled after MBind's complex viewer.

Not AutoDock/Vina docking: the query ligand is embedded in 3D with RDKit and translated
so its heavy-atom centroid matches the co-crystal binding-site center (from *_ligand_only.pdb
when present). The native ligand is not drawn—only receptor cartoon + user ligand sticks.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import py3Dmol
except ImportError:
    py3Dmol = None

# MBind-aligned palette: tan cartoon receptor, greenCarbon query sticks
RECEPTOR_CARTOON_HEX = "d2b48c"


def _resolve_data_root() -> Path:
    env_root = os.environ.get("GPCR_DATA_ROOT", "").strip()
    if env_root:
        return Path(env_root)
    gpcr_main = Path(__file__).resolve().parents[2]
    if (gpcr_main / "Josh_Receptor_Features").is_dir():
        return gpcr_main
    sibling = gpcr_main.parent / "GUI_Folder"
    if (sibling / "Josh_Receptor_Features").is_dir():
        return sibling
    return gpcr_main.parent / "GPCRtryagain - Delete - Copy"


def resolve_receptor_structure_paths(receptor_folder: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Return (receptor_only_pdb, reference_ligand_only_pdb) under Josh_Receptor_Features/<name>/.
    """
    root = _resolve_data_root()
    pocket = root / "Josh_Receptor_Features" / str(receptor_folder).strip()
    if not pocket.is_dir():
        return None, None
    rec = next(iter(sorted(pocket.glob("*_receptor_only.pdb"))), None)
    ref_lig = next(iter(sorted(pocket.glob("*_ligand_only.pdb"))), None)
    return rec, ref_lig


def _pdb_heavy_atom_com(pdb_text: str) -> Optional[np.ndarray]:
    coords: list[list[float]] = []
    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) < 38:
            continue
        try:
            elem = line[76:78].strip() or line[12:16].strip()[:1]
        except Exception:
            elem = ""
        if elem.upper() in ("H", "D"):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        coords.append([x, y, z])
    if not coords:
        return None
    return np.asarray(coords, dtype=np.float64).mean(axis=0)


def smiles_to_pdb_block_aligned(
    smiles: str,
    target_com: np.ndarray,
    random_seed: int = 42,
) -> Optional[str]:
    """ETKDG 3D structure for SMILES, translated so ligand COM matches target_com."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    embed_code = -1
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        embed_code = AllChem.EmbedMolecule(mol, params)
    except AttributeError:
        embed_code = AllChem.EmbedMolecule(mol, randomSeed=random_seed)
    if embed_code != 0:
        if AllChem.EmbedMolecule(mol, randomSeed=random_seed) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass

    conf = mol.GetConformer()
    positions = []
    for i in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetAtomicNum() <= 1:
            continue
        p = conf.GetAtomPosition(i)
        positions.append([p.x, p.y, p.z])
    if not positions:
        return None
    lig_com = np.mean(positions, axis=0)
    delta = target_com - lig_com
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(
            i,
            (p.x + float(delta[0]), p.y + float(delta[1]), p.z + float(delta[2])),
        )
    mol_h = Chem.RemoveHs(mol)
    return Chem.MolToPDBBlock(mol_h, confId=0)


def build_gpcr_complex_view_html(
    receptor_pdb_text: str,
    query_ligand_pdb_text: str,
    width: int = 720,
    height: int = 520,
) -> Optional[str]:
    """MBind-style py3Dmol HTML: receptor cartoon + query ligand sticks only (no co-crystal ligand)."""
    if py3Dmol is None:
        return None
    if not receptor_pdb_text.strip() or not query_ligand_pdb_text.strip():
        return None
    try:
        view = py3Dmol.view(width=width, height=height)
        view.addModel(receptor_pdb_text, "pdb")
        view.setStyle({"model": 0}, {"cartoon": {"color": f"0x{RECEPTOR_CARTOON_HEX}"}})
        view.addModel(query_ligand_pdb_text, "pdb")
        view.setStyle({"model": 1}, {"stick": {"radius": 0.13, "colorscheme": "greenCarbon"}})
        view.zoomTo()
        return view._make_html()
    except Exception:
        return None


def build_aligned_complex_html_for_receptor(
    receptor_folder: str,
    canonical_smiles: str,
    width: int = 720,
    height: int = 520,
) -> Tuple[Optional[str], str]:
    """
    Load PDB assets for receptor_folder, align query ligand to reference ligand COM, return (html, status_message).
    """
    rec_path, ref_lig_path = resolve_receptor_structure_paths(receptor_folder)
    if rec_path is None or not rec_path.is_file():
        return None, "No receptor PDB found for this target (expected *_receptor_only.pdb)."
    rec_text = rec_path.read_text(encoding="utf-8", errors="ignore")
    ref_text = ref_lig_path.read_text(encoding="utf-8", errors="ignore") if ref_lig_path and ref_lig_path.is_file() else ""
    target_com = _pdb_heavy_atom_com(ref_text) if ref_text.strip() else _pdb_heavy_atom_com(rec_text)
    if target_com is None:
        return None, "Could not derive a binding-site center from the reference structures."
    query_pdb = smiles_to_pdb_block_aligned(canonical_smiles, target_com)
    if not query_pdb:
        return None, "RDKit could not build a 3D conformer for this SMILES."
    # Reference ligand PDB is used only for binding-site centroid; not rendered in the viewer.
    html = build_gpcr_complex_view_html(rec_text, query_pdb, width=width, height=height)
    if html is None:
        if py3Dmol is None:
            return None, "Install py3Dmol for the 3D viewer: pip install py3Dmol"
        return None, "3D viewer failed to render (check PDB/SMILES)."
    return html, "ok"


def py3dmol_available() -> bool:
    return py3Dmol is not None
