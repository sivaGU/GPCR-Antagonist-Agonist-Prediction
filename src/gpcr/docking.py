"""
Post-prediction docking helpers for the GPCR Streamlit GUI.

This module keeps the docking trigger explicit (user-clicked) and uses receptor-specific
grid centers from each receptor folder's `<id>_ligand_only.pdb`.
"""
from __future__ import annotations

import os
import re
import json
import stat
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .structure_view import resolve_receptor_structure_paths

try:
    import py3Dmol
except ImportError:
    py3Dmol = None


# Match MBind defaults for consistency.
DEFAULT_EXHAUSTIVENESS = 64
DEFAULT_NUM_MODES = 10
DEFAULT_SEED = 42
DEFAULT_TIMEOUT_S = 300

# Reuse the same flat receptor/ligand colors as the GUI viewer.
RECEPTOR_CARTOON_HEX = "d2b48c"
DOCKED_LIGAND_STICK_HEX = "35c86d"


@dataclass
class DockingResult:
    ok: bool
    message: str
    receptor_name: str
    canonical_smiles: str
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    score_kcal_mol: Optional[float]
    engine: str
    command: str
    out_pose_path: Optional[str]
    log_path: Optional[str]
    html: Optional[str]


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _mbind_source_files_dir(project_root: Path) -> Path:
    # Workspace layout in this project: <Final GUI>/<repo>, with MBind-main as sibling.
    return project_root.parent / "MBind-main" / "Files_for_GUI"


def ensure_docking_files_folder(project_root: Optional[Path] = None) -> Tuple[Path, str]:
    """
    Ensure a local docking-files folder exists inside this GUI project.
    Copies/syncs assets from sibling MBind-main/Files_for_GUI when available.
    """
    root = project_root or _resolve_project_root()
    dst = root / "Docking_Files"
    src = _mbind_source_files_dir(root)
    dst.mkdir(parents=True, exist_ok=True)
    if not src.is_dir():
        _write_receptor_grid_manifest(root, dst)
        _ensure_local_binaries_executable(dst)
        return dst, (
            f"Docking_Files created at {dst}, but source folder was not found at {src}. "
            "Place MBind-main beside this project to sync AD4 helper files automatically."
        )
    copied = 0
    for path in src.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            src_size = path.stat().st_size
            needs_copy = (not out.exists()) or (out.stat().st_size != src_size)
        except OSError:
            needs_copy = True
        if needs_copy:
            shutil.copy2(path, out)
            copied += 1
    _write_receptor_grid_manifest(root, dst)
    _ensure_local_binaries_executable(dst)
    return dst, f"Docking_Files synced from MBind source ({copied} updated file(s))."


def _write_receptor_grid_manifest(project_root: Path, docking_files_dir: Path) -> None:
    receptor_root = project_root / "Josh_Receptor_Features"
    if not receptor_root.is_dir():
        return
    manifest = {}
    for folder in sorted(p.name for p in receptor_root.iterdir() if p.is_dir()):
        rec, lig = resolve_receptor_structure_paths(folder)
        if rec is None or lig is None or not lig.is_file():
            continue
        coords = _ligand_heavy_coords_from_pdb(lig.read_text(encoding="utf-8", errors="ignore"))
        if coords is None:
            continue
        center, size = _grid_from_ligand_coords(coords)
        pdb_id = lig.name.replace("_ligand_only.pdb", "")
        manifest[folder] = {
            "pdb_id": pdb_id,
            "center_x": center[0],
            "center_y": center[1],
            "center_z": center[2],
            "size_x": size[0],
            "size_y": size[1],
            "size_z": size[2],
        }
    out = docking_files_dir / "receptor_grid_boxes.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _is_executable_on_platform(path: Path) -> bool:
    if not path.is_file():
        return False
    if os.name == "nt":
        return path.suffix.lower() == ".exe"
    return os.access(path, os.X_OK)


def _try_make_executable(path: Path) -> bool:
    if not path.is_file() or os.name == "nt":
        return _is_executable_on_platform(path)
    try:
        mode = path.stat().st_mode
        os.chmod(path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        return False
    return os.access(path, os.X_OK)


def _ensure_local_binaries_executable(files_dir: Path) -> None:
    """
    Linux deployments often copy binaries without +x bit. Set execute bits proactively.
    """
    if os.name == "nt":
        return
    for name in ("vina", "smina", "gnina", "autogrid4", "autodock4"):
        p = files_dir / name
        if p.is_file():
            _try_make_executable(p)


def _pdb_line_element_symbol(line: str) -> str:
    elem = ""
    if len(line) >= 78:
        elem = line[76:78].strip().upper()
    if elem:
        return elem
    if len(line) < 16:
        return ""
    name = line[12:16].strip().upper()
    if not name:
        return ""
    two = name[:2]
    if two in ("BR", "CL", "FE", "ZN", "MG", "CA", "NA", "MN", "CU", "CO", "NI", "SE"):
        return two
    if name[0].isdigit() and len(name) > 1 and name[1].isalpha():
        return name[1:2]
    return name[0]


def _ligand_heavy_coords_from_pdb(pdb_text: str) -> Optional[np.ndarray]:
    coords = []
    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) < 54:
            continue
        elem = _pdb_line_element_symbol(line)
        if elem in ("H", "D"):
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
    return np.asarray(coords, dtype=np.float64)


def _grid_from_ligand_coords(coords: np.ndarray) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    center = coords.mean(axis=0)
    span = np.ptp(coords, axis=0)
    # Keep box dimensions in requested range (15-20 A), with small pad around the ligand extent.
    size = np.clip(span + 10.0, 15.0, 20.0)
    return (float(center[0]), float(center[1]), float(center[2])), (
        float(size[0]),
        float(size[1]),
        float(size[2]),
    )


def compute_receptor_grid_params(receptor_folder: str) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]], str]:
    rec_path, lig_path = resolve_receptor_structure_paths(receptor_folder)
    if rec_path is None or not rec_path.is_file():
        return None, None, "Missing receptor-only PDB for this target."
    if lig_path is None or not lig_path.is_file():
        return None, None, "Missing ligand-only PDB for this target."
    lig_text = lig_path.read_text(encoding="utf-8", errors="ignore")
    coords = _ligand_heavy_coords_from_pdb(lig_text)
    if coords is None:
        return None, None, "Could not parse heavy-atom coordinates from ligand-only PDB."
    center, size = _grid_from_ligand_coords(coords)
    return center, size, "ok"


def _select_docking_engine(files_dir: Path) -> Tuple[Optional[str], Optional[Path]]:
    is_windows = os.name == "nt"
    local_candidates = [
        "smina.exe" if is_windows else "smina",
        "gnina.exe" if is_windows else "gnina",
        "vina.exe" if is_windows else "vina",
    ]
    for name in local_candidates:
        p = files_dir / name
        if p.is_file():
            if _is_executable_on_platform(p) or _try_make_executable(p):
                return name.split(".")[0].lower(), p

    # PATH fallback
    path_candidates = ["smina", "gnina", "vina"]
    for cmd in path_candidates:
        found = shutil.which(cmd)
        if found:
            return cmd, Path(found)
        # Windows convenience fallback when PATH entry is `vina.exe`.
        if is_windows and cmd == "vina":
            found_exe = shutil.which("vina.exe")
            if found_exe:
                return "vina", Path(found_exe)
    return None, None


def _sanitize_receptor_pdb_for_view(pdb_text: str) -> str:
    protein_resn = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HIE", "HID", "HIP",
        "ILE", "LEU", "LYS", "MET", "MSE", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    }
    out = []
    for line in pdb_text.splitlines():
        if line.startswith("HETATM"):
            continue
        if line.startswith("ATOM") and len(line) > 20:
            if line[17:20].strip().upper() not in protein_resn:
                continue
        out.append(line)
    return "\n".join(out)


def _extract_top_pose_pdb_block(docked_path: Path) -> Optional[str]:
    try:
        lines = docked_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    in_model = False
    saw_model = False
    pose_lines = []
    for line in lines:
        if line.startswith("MODEL"):
            if saw_model:
                break
            saw_model = True
            in_model = True
            continue
        if line.startswith("ENDMDL"):
            if in_model:
                break
            continue
        if saw_model and not in_model:
            continue
        if line.startswith(("ATOM", "HETATM")):
            pose_lines.append(line)
    if not pose_lines:
        return None
    return "\n".join(pose_lines) + "\n"


def _extract_best_score_kcal_mol(docked_path: Path) -> Optional[float]:
    try:
        txt = docked_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    # Vina/SMINA/GNINA often emit: REMARK VINA RESULT: -7.2 ...
    m = re.search(r"REMARK\s+VINA\s+RESULT:\s+(-?\d+(?:\.\d+)?)", txt)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _extract_best_score_from_stdout(stdout_text: str) -> Optional[float]:
    # Vina/smina table usually has a first row: "1   -7.2 ..."
    for line in (stdout_text or "").splitlines():
        m = re.match(r"^\s*1\s+(-?\d+(?:\.\d+)?)\b", line)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def _build_docked_complex_html(receptor_pdb_text: str, pose_pdb_text: str, width: int = 720, height: int = 520) -> Optional[str]:
    if py3Dmol is None:
        return None
    if not receptor_pdb_text.strip() or not pose_pdb_text.strip():
        return None
    try:
        view = py3Dmol.view(width=width, height=height)
        view.addModel(receptor_pdb_text, "pdb")
        view.setStyle({"model": 0}, {"cartoon": {"color": f"0x{RECEPTOR_CARTOON_HEX}"}})
        view.addModel(pose_pdb_text, "pdb")
        view.setStyle({"model": 1}, {"stick": {"radius": 0.16, "color": f"0x{DOCKED_LIGAND_STICK_HEX}"}})
        view.zoomTo()
        return view._make_html()
    except Exception:
        return None


def _smiles_to_sdf_file(smiles: str, out_path: Path) -> Tuple[bool, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Could not parse canonical SMILES for docking."
    mol = Chem.AddHs(mol)
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = DEFAULT_SEED
        emb = AllChem.EmbedMolecule(mol, params)
    except Exception:
        emb = AllChem.EmbedMolecule(mol, randomSeed=DEFAULT_SEED)
    if emb != 0:
        return False, "RDKit failed to generate a 3D conformer for docking."
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out_path))
    writer.write(mol)
    writer.close()
    return True, "ok"


def _obabel_convert(in_file: Path, out_file: Path, in_format: str, out_format: str) -> Tuple[bool, str]:
    """
    Convert structure file format with Open Babel when available.
    """
    obabel = shutil.which("obabel")
    if not obabel:
        return False, "Open Babel (`obabel`) is not available in PATH."
    cmd = [
        obabel,
        "-i",
        in_format,
        str(in_file),
        "-o",
        out_format,
        "-O",
        str(out_file),
    ]
    # Add hydrogens and Gasteiger charges for docking-friendly PDBQT outputs.
    if out_format.lower() == "pdbqt":
        cmd += ["-h", "--partialcharge", "gasteiger"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception as e:
        return False, f"Open Babel conversion failed to launch: {e}"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return False, f"Open Babel conversion failed: {err[:400]}"
    if not out_file.exists() or out_file.stat().st_size == 0:
        return False, f"Open Babel conversion produced no output file: {out_file}"
    return True, "ok"


def _prepare_inputs_for_engine(
    engine: str,
    receptor_src: Path,
    ligand_sdf: Path,
    run_dir: Path,
) -> Tuple[Optional[Path], Optional[Path], str]:
    """
    Prepare receptor/ligand inputs for the selected engine.
    - smina/gnina can consume receptor PDB and ligand SDF directly.
    - vina generally expects PDBQT for both receptor and ligand, so convert as needed.
    """
    if engine in ("smina", "gnina"):
        return receptor_src, ligand_sdf, "ok"

    # Vina path: use PDBQT receptor and ligand.
    receptor_for_docking = receptor_src
    ligand_for_docking = run_dir / "query_ligand.pdbqt"
    ok_lig, msg_lig = _obabel_convert(ligand_sdf, ligand_for_docking, "sdf", "pdbqt")
    if not ok_lig:
        return None, None, (
            "Could not prepare ligand PDBQT for Vina from SMILES-derived SDF. "
            f"{msg_lig}"
        )

    if receptor_src.suffix.lower() != ".pdbqt":
        receptor_for_docking = run_dir / f"{receptor_src.stem}.pdbqt"
        ok_rec, msg_rec = _obabel_convert(receptor_src, receptor_for_docking, "pdb", "pdbqt")
        if not ok_rec:
            return None, None, (
                "Could not prepare receptor PDBQT for Vina from receptor PDB. "
                f"{msg_rec}"
            )
    return receptor_for_docking, ligand_for_docking, "ok"


def run_single_receptor_docking(
    receptor_folder: str,
    canonical_smiles: str,
    base_exhaustiveness: int = DEFAULT_EXHAUSTIVENESS,
    base_num_modes: int = DEFAULT_NUM_MODES,
) -> DockingResult:
    project_root = _resolve_project_root()
    files_dir, sync_msg = ensure_docking_files_folder(project_root)
    engine, engine_path = _select_docking_engine(files_dir)
    if engine_path is None:
        return DockingResult(
            ok=False,
            message=(
                f"{sync_msg} No docking engine found. Add `smina`/`gnina`/`vina` binary to `{files_dir}` "
                "or install one in PATH. On Linux, ensure the file has execute permission (`chmod +x`)."
            ),
            receptor_name=receptor_folder,
            canonical_smiles=canonical_smiles,
            center=(0.0, 0.0, 0.0),
            size=(0.0, 0.0, 0.0),
            score_kcal_mol=None,
            engine="none",
            command="",
            out_pose_path=None,
            log_path=None,
            html=None,
        )

    rec_path, lig_path = resolve_receptor_structure_paths(receptor_folder)
    if rec_path is None or not rec_path.is_file():
        return DockingResult(
            ok=False,
            message="No receptor-only PDB found for selected receptor.",
            receptor_name=receptor_folder,
            canonical_smiles=canonical_smiles,
            center=(0.0, 0.0, 0.0),
            size=(0.0, 0.0, 0.0),
            score_kcal_mol=None,
            engine=engine or "unknown",
            command="",
            out_pose_path=None,
            log_path=None,
            html=None,
        )
    if lig_path is None or not lig_path.is_file():
        return DockingResult(
            ok=False,
            message="No ligand-only PDB found for selected receptor.",
            receptor_name=receptor_folder,
            canonical_smiles=canonical_smiles,
            center=(0.0, 0.0, 0.0),
            size=(0.0, 0.0, 0.0),
            score_kcal_mol=None,
            engine=engine or "unknown",
            command="",
            out_pose_path=None,
            log_path=None,
            html=None,
        )

    lig_coords = _ligand_heavy_coords_from_pdb(lig_path.read_text(encoding="utf-8", errors="ignore"))
    if lig_coords is None:
        return DockingResult(
            ok=False,
            message="Could not parse ligand-only PDB coordinates to build a docking grid.",
            receptor_name=receptor_folder,
            canonical_smiles=canonical_smiles,
            center=(0.0, 0.0, 0.0),
            size=(0.0, 0.0, 0.0),
            score_kcal_mol=None,
            engine=engine or "unknown",
            command="",
            out_pose_path=None,
            log_path=None,
            html=None,
        )
    center, size = _grid_from_ligand_coords(lig_coords)

    runs_dir = project_root / "docking_runs"
    run_dir = runs_dir / f"{receptor_folder}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ligand_sdf = run_dir / "query_ligand.sdf"
    docked_out = run_dir / "docked_topposes.pdbqt"
    log_path = run_dir / "docking.log"

    ok_prep, prep_msg = _smiles_to_sdf_file(canonical_smiles, ligand_sdf)
    if not ok_prep:
        return DockingResult(
            ok=False,
            message=prep_msg,
            receptor_name=receptor_folder,
            canonical_smiles=canonical_smiles,
            center=center,
            size=size,
            score_kcal_mol=None,
            engine=engine or "unknown",
            command="",
            out_pose_path=None,
            log_path=None,
            html=None,
        )

    receptor_for_docking, ligand_for_docking, prep_engine_msg = _prepare_inputs_for_engine(
        engine=engine or "unknown",
        receptor_src=rec_path,
        ligand_sdf=ligand_sdf,
        run_dir=run_dir,
    )
    if receptor_for_docking is None or ligand_for_docking is None:
        return DockingResult(
            ok=False,
            message=prep_engine_msg,
            receptor_name=receptor_folder,
            canonical_smiles=canonical_smiles,
            center=center,
            size=size,
            score_kcal_mol=None,
            engine=engine or "unknown",
            command="",
            out_pose_path=None,
            log_path=None,
            html=None,
        )

    cmd = [
        str(engine_path),
        "-r",
        str(receptor_for_docking),
        "-l",
        str(ligand_for_docking),
        "--center_x",
        str(center[0]),
        "--center_y",
        str(center[1]),
        "--center_z",
        str(center[2]),
        "--size_x",
        str(size[0]),
        "--size_y",
        str(size[1]),
        "--size_z",
        str(size[2]),
        "--num_modes",
        str(int(base_num_modes)),
        "--exhaustiveness",
        str(int(base_exhaustiveness)),
        "--seed",
        str(DEFAULT_SEED),
        "-o",
        str(docked_out),
    ]

    if engine == "gnina":
        cmd += ["--scoring", "vina"]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return DockingResult(
            ok=False,
            message=f"Docking timed out after {DEFAULT_TIMEOUT_S}s.",
            receptor_name=receptor_folder,
            canonical_smiles=canonical_smiles,
            center=center,
            size=size,
            score_kcal_mol=None,
            engine=engine or "unknown",
            command=" ".join(cmd),
            out_pose_path=None,
            log_path=None,
            html=None,
        )
    except Exception as e:
        return DockingResult(
            ok=False,
            message=f"Docking failed to launch: {e}",
            receptor_name=receptor_folder,
            canonical_smiles=canonical_smiles,
            center=center,
            size=size,
            score_kcal_mol=None,
            engine=engine or "unknown",
            command=" ".join(cmd),
            out_pose_path=None,
            log_path=None,
            html=None,
        )

    log_path.write_text(
        "\n".join(
            [
                f"Engine: {engine_path}",
                f"Command: {' '.join(cmd)}",
                f"Return code: {proc.returncode}",
                "",
                "---- STDOUT ----",
                proc.stdout or "",
                "",
                "---- STDERR ----",
                proc.stderr or "",
            ]
        ),
        encoding="utf-8",
    )

    if proc.returncode != 0:
        err_tail = (proc.stderr or proc.stdout or "").strip()
        return DockingResult(
            ok=False,
            message=(
                "Docking engine returned a non-zero exit code. "
                f"See log: {log_path}. "
                f"Engine output: {err_tail[:220]}"
            ),
            receptor_name=receptor_folder,
            canonical_smiles=canonical_smiles,
            center=center,
            size=size,
            score_kcal_mol=None,
            engine=engine or "unknown",
            command=" ".join(cmd),
            out_pose_path=None,
            log_path=str(log_path),
            html=None,
        )
    if not docked_out.exists() or docked_out.stat().st_size == 0:
        return DockingResult(
            ok=False,
            message=f"Docking completed but no output pose was written: {docked_out}",
            receptor_name=receptor_folder,
            canonical_smiles=canonical_smiles,
            center=center,
            size=size,
            score_kcal_mol=None,
            engine=engine or "unknown",
            command=" ".join(cmd),
            out_pose_path=None,
            log_path=str(log_path),
            html=None,
        )

    score = _extract_best_score_from_stdout(proc.stdout or "")
    if score is None:
        score = _extract_best_score_kcal_mol(docked_out)
    pose_text = _extract_top_pose_pdb_block(docked_out)
    rec_text = _sanitize_receptor_pdb_for_view(rec_path.read_text(encoding="utf-8", errors="ignore"))
    html = _build_docked_complex_html(rec_text, pose_text or "")

    return DockingResult(
        ok=True,
        message=f"{sync_msg} Docking completed successfully.",
        receptor_name=receptor_folder,
        canonical_smiles=canonical_smiles,
        center=center,
        size=size,
        score_kcal_mol=score,
        engine=engine or "unknown",
        command=" ".join(cmd),
        out_pose_path=str(docked_out),
        log_path=str(log_path),
        html=html,
    )

