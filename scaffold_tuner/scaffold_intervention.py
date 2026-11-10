# -*- coding: utf-8 -*-
"""
scaffold_intervention.py  –  ScaffoldTuner v2

Primary workflow
----------------
    desired descriptor change  (e.g. HBA +1)
            ↓
    Step 1: descriptor-indexed retrieval
            library[delta_HBA == +1]   ← fragment-level pre-filter only
            ↓
    Step 2: exact R-group match  (default; match_mode="exact")
            parent R-group == from_fragment  (canonical SMILES)
            ↓
    Step 3: candidate generation
            ↓
    Step 4: RDKit sanitization  (inside build_molecule)
            ↓
    Step 5: parent scaffold retained as substructure
            generated_mol.HasSubstructMatch(parent_scaffold_mol)
            ↓
    Step 6: whole-molecule descriptor recomputation (RDKit)
            actual_delta = generated_HBA - parent_HBA
            ↓
    Step 7: exact requested delta validation
            actual_delta == requested_delta
            ↓
    Step 8: optional strict filtering
            ↓
    Step 9: ranking (collateral_change → match type → frequency)
            ↓
    Step 10: optional MaxMin diversity selection
            ↓
    Top-N candidates with full provenance

Key difference from CReM
-------------------------
CReM indexes replacements by local chemical context (ECFP-like radius).
ScaffoldTuner v2 indexes by *desired descriptor delta*:

    desired delta → retrieve matching transformations → exact R-group match
    → generate → whole-molecule exact validation

Similarity-based matching
--------------------------
match_mode="similarity" is an *optional exploratory mode only*.
It must NOT be used in benchmark comparisons.

Scaffold preservation
---------------------
"Scaffold preserved" means generated_mol.HasSubstructMatch(parent_scaffold).
This does NOT mean the generated molecule's Murcko scaffold is identical to
the parent's.  The parent scaffold is *retained as a substructure*.

remove_ar
---------
AR -1 is intentionally unsupported.  Removing an aromatic ring from an
R-group would require modifying the scaffold itself.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdRGroupDecomposition
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Descriptor helpers
# ──────────────────────────────────────────────────────────────────────────────

DESCRIPTOR_NAMES = [
    "NumHAcceptors",
    "NumHDonors",
    "NumAromaticRings",
    "NumRotatableBonds",
    "NumHeteroatoms",
]

_DESC_INDEX: dict[str, int] = {
    "HBA":  0,
    "HBD":  1,
    "AR":   2,
    "RotB": 3,
    "HA":   4,
}

_SUPPORTED: set[tuple[str, int]] = {
    ("HBD", +1), ("HBD", -1),
    ("HBA", +1), ("HBA", -1),
    ("AR",  +1),
    ("RotB",+1), ("RotB",-1),
    ("HA",  +1), ("HA",  -1),
}

_MODE_MAP: dict[str, tuple[str, int]] = {
    "add_hbd":    ("HBD",  +1),
    "remove_hbd": ("HBD",  -1),
    "add_hba":    ("HBA",  +1),
    "remove_hba": ("HBA",  -1),
    "add_ar":     ("AR",   +1),
    "add_rb":     ("RotB", +1),
    "remove_rb":  ("RotB", -1),
    "add_ha":     ("HA",   +1),
    "remove_ha":  ("HA",   -1),
}


def count_features(mol: Chem.Mol) -> tuple[int, ...]:
    """Return (HBA, HBD, NumAromaticRings, NumRotatableBonds, NumHeteroatoms)."""
    return (
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumHDonors(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumHeteroatoms(mol),
    )


def _morgan_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Transformation library
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_COLS: set[str] = {
    "from_fragment", "to_fragment",
    "delta_HBD", "delta_HBA", "delta_AR", "delta_RotB",
}

_LIB_CACHE: dict[str, pd.DataFrame] = {}


def load_transformation_library(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a transformation library from a CSV, TSV, or Parquet file.

    Required columns
    ----------------
    from_fragment, to_fragment,
    delta_HBD, delta_HBA, delta_AR, delta_RotB

    Optional columns (auto-filled with defaults if absent)
    -------------------------------------------------------
    delta_HA   (default 0)
    frequency  (default 1)
    source     (default "unknown")

    IMPORTANT: delta_* are fragment-level values used as a retrieval
    pre-filter only.  The final acceptance criterion is the whole-molecule
    actual_delta recomputed by RDKit after generation.
    """
    key = str(path)
    if key in _LIB_CACHE:
        return _LIB_CACHE[key]

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Transformation library not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(p)
    elif suffix in (".tsv", ".txt"):
        df = pd.read_csv(p, sep="\t")
    elif suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Use .csv, .tsv, or .parquet."
        )

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Transformation library is missing required columns: {missing}"
        )

    for col, default in [("delta_HA", 0), ("frequency", 1)]:
        if col not in df.columns:
            df[col] = default
    if "source" not in df.columns:
        df["source"] = "unknown"

    for col in ["delta_HBD", "delta_HBA", "delta_AR", "delta_RotB",
                "delta_HA", "frequency"]:
        df[col] = df[col].fillna(0).astype(int)

    _LIB_CACHE[key] = df
    return df


def retrieve_transformations(
    library: pd.DataFrame,
    target: str,
    delta: int,
) -> pd.DataFrame:
    """
    Step 1: descriptor-indexed retrieval.

    Return library rows whose fragment-level delta for *target* equals *delta*.
    This is a pre-filter only; whole-molecule validation happens in Steps 6-7.
    """
    col = f"delta_{target}"
    if col not in library.columns:
        raise ValueError(f"Column '{col}' not found in transformation library.")
    return library[library[col] == delta].copy()


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Murcko scaffold and R-group decomposition
# ──────────────────────────────────────────────────────────────────────────────

def make_murcko_core_with_rlabels(
    parent_smiles: str,
    max_cuts: Optional[int] = None,
) -> tuple[Chem.Mol, Chem.Mol, Chem.Mol]:
    """
    Extract the Murcko scaffold and label each scaffold–sidechain boundary
    with [*:N].

    Returns (parent_mol, scaffold_mol, core_with_attachment_labels).
    """
    parent_mol = Chem.MolFromSmiles(parent_smiles)
    if parent_mol is None:
        raise ValueError("Invalid parent SMILES.")

    scaffold_mol = MurckoScaffold.GetScaffoldForMol(parent_mol)
    if scaffold_mol is None:
        raise ValueError("Failed to generate Murcko scaffold.")

    matches = parent_mol.GetSubstructMatches(scaffold_mol)
    if not matches:
        raise ValueError("Murcko scaffold did not match the parent molecule.")

    match = matches[0]
    core_atom_idxs = set(match)
    parent_to_core = {p: c for c, p in enumerate(match)}

    boundary_bonds = [
        b for b in parent_mol.GetBonds()
        if (b.GetBeginAtomIdx() in core_atom_idxs) ^ (b.GetEndAtomIdx() in core_atom_idxs)
    ]
    if max_cuts is not None:
        boundary_bonds = boundary_bonds[:max_cuts]

    rw_core = Chem.RWMol(scaffold_mol)
    for rlabel, bond in enumerate(boundary_bonds, start=1):
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        core_parent_idx = a1 if a1 in core_atom_idxs else a2
        core_idx = parent_to_core[core_parent_idx]
        dummy = Chem.Atom(0)
        dummy.SetAtomMapNum(rlabel)
        dummy_idx = rw_core.AddAtom(dummy)
        rw_core.AddBond(core_idx, dummy_idx, bond.GetBondType())

    core_with_labels = rw_core.GetMol()
    Chem.SanitizeMol(core_with_labels)
    return parent_mol, scaffold_mol, core_with_labels


def get_r_labels(core_mol: Chem.Mol) -> list[int]:
    """Return sorted atom-map numbers of all [*:N] dummy atoms."""
    return sorted({
        atom.GetAtomMapNum()
        for atom in core_mol.GetAtoms()
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() > 0
    })


def get_original_rgroups(
    parent_smiles: str,
    core_with_labels: Chem.Mol,
) -> dict[str, str]:
    """R-group decomposition → {"R1": SMILES, "R2": SMILES, ...}."""
    parent_mol = Chem.MolFromSmiles(parent_smiles)
    if parent_mol is None:
        raise ValueError("Invalid parent SMILES.")

    params = rdRGroupDecomposition.RGroupDecompositionParameters()
    params.labels = rdRGroupDecomposition.RGroupLabels.AtomMapLabels
    params.rgroupLabelling = rdRGroupDecomposition.RGroupLabelling.AtomMap
    params.onlyMatchAtRGroups = True

    rgd = rdRGroupDecomposition.RGroupDecomposition(core_with_labels, params)
    if rgd.Add(parent_mol) < 0:
        raise ValueError("RGroupDecomposition.Add() failed.")
    if not rgd.Process():
        raise RuntimeError("RGroupDecomposition.Process() failed.")

    rows = rgd.GetRGroupsAsRows()
    if not rows:
        raise RuntimeError("No R-group decomposition result.")

    return {
        key: Chem.MolToSmiles(mol)
        for key, mol in rows[0].items()
        if key != "Core" and mol is not None
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Fragment attachment
# ──────────────────────────────────────────────────────────────────────────────

def _get_dummy_info(
    mol: Chem.Mol, map_num: int
) -> tuple[int, int, Chem.rdchem.BondType]:
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == map_num:
            nbr = atom.GetNeighbors()[0]
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx())
            return atom.GetIdx(), nbr.GetIdx(), bond.GetBondType()
    raise ValueError(f"Dummy atom [*:{map_num}] not found.")


def attach_fragment(
    scaffold_mol: Chem.Mol,
    fragment_mol: Chem.Mol,
    map_num: int,
) -> Chem.Mol:
    """Join scaffold and fragment at [*:map_num]; remove dummy atoms."""
    s_dummy, s_nbr, s_bond = _get_dummy_info(scaffold_mol, map_num)
    f_dummy, f_nbr, _      = _get_dummy_info(fragment_mol, map_num)

    offset = scaffold_mol.GetNumAtoms()
    emol = Chem.EditableMol(Chem.CombineMols(scaffold_mol, fragment_mol))
    emol.AddBond(s_nbr, f_nbr + offset, order=s_bond)
    for idx in sorted([s_dummy, f_dummy + offset], reverse=True):
        emol.RemoveAtom(idx)

    mol = emol.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def build_molecule(scaffold_smiles: str, rgroups: dict[str, str]) -> Chem.Mol:
    """Attach all R-groups to scaffold in ascending site-number order."""
    mol = Chem.MolFromSmiles(scaffold_smiles)
    if mol is None:
        raise ValueError("Invalid scaffold SMILES.")
    for site in sorted(rgroups, key=lambda x: int(x[1:])):
        frag = Chem.MolFromSmiles(rgroups[site])
        if frag is None:
            raise ValueError(f"Invalid fragment SMILES: {rgroups[site]}")
        mol = attach_fragment(mol, frag, int(site[1:]))
    return mol


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Transformation applicability (Step 2)
# ──────────────────────────────────────────────────────────────────────────────

def normalize_fragment_smiles(smi: str) -> str:
    """Canonicalize fragment SMILES; normalize attachment dummy to [*:1]."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(1)
    return Chem.MolToSmiles(mol)


def extract_rgroups_for_smiles(
    parent_smiles: str,
    max_cuts: Optional[int] = None,
) -> list[str] | None:
    """
    Public utility: extract and normalize R-group SMILES for a parent molecule.

    This function is the single authoritative implementation shared by
    propose_structures() (generation) and compute_applicability.py
    (evaluation).  Using the same function in both places guarantees that
    the R-group definition is identical across generation and evaluation.

    Parameters
    ----------
    parent_smiles : SMILES of the parent molecule
    max_cuts      : maximum number of scaffold attachment points to label
                    (None = use all boundary bonds, which is the default
                    used by propose_structures)

    Returns
    -------
    List of normalized R-group SMILES strings (attachment dummy = [*:1])
    for every attachment site, or None if scaffold / R-group extraction fails
    for any reason (invalid SMILES, no boundary bonds, decomposition failure).

    Each string is the output of normalize_fragment_smiles(), so it can be
    directly compared to library from_fragment values.
    """
    try:
        _, _, core = make_murcko_core_with_rlabels(parent_smiles, max_cuts=max_cuts)
        rgroups    = get_original_rgroups(parent_smiles, core)
    except Exception:
        return None

    if not rgroups:
        return None

    return [normalize_fragment_smiles(smi) for smi in rgroups.values()]


def _exact_match(rgroup_smi: str, from_frag_smi: str) -> bool:
    """
    Return True iff parent R-group equals library from_fragment
    (canonical SMILES comparison after normalizing attachment dummy).
    """
    return (
        normalize_fragment_smiles(rgroup_smi)
        == normalize_fragment_smiles(from_frag_smi)
    )


def _fragment_tanimoto(smi_a: str, smi_b: str, radius: int = 2, n_bits: int = 2048) -> float:
    """Morgan-fingerprint Tanimoto between two fragment SMILES. 0.0 on failure."""
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None:
        return 0.0
    fp_a = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol_a, radius, nBits=n_bits)
    fp_b = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol_b, radius, nBits=n_bits)
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def _get_applicable_transforms(
    current_rgroup_smi: str,
    candidate_transforms: pd.DataFrame,
    match_mode: str,
    similarity_threshold: float,
) -> pd.DataFrame:
    """
    Step 2: filter candidate_transforms to those applicable at this site.

    match_mode="exact"  (default)
        Exact canonical SMILES match only.  Returns empty DataFrame when
        no exact match exists — no silent fallback.

    match_mode="similarity"  (optional exploratory mode)
        Tries exact first; if empty, falls back to Tanimoto >= threshold.
        NOT for benchmark use.
    """
    if match_mode not in ("exact", "similarity"):
        raise ValueError(
            f"match_mode must be 'exact' or 'similarity', got '{match_mode}'."
        )

    mask_exact = candidate_transforms["from_fragment"].apply(
        lambda frag: _exact_match(current_rgroup_smi, frag)
    )
    exact = candidate_transforms[mask_exact].copy()

    if not exact.empty:
        exact["match_type"]       = "exact"
        exact["match_similarity"] = 1.0
        return exact

    if match_mode == "exact":
        return pd.DataFrame()

    # Similarity fallback (only when explicitly requested)
    sims = candidate_transforms["from_fragment"].apply(
        lambda frag: _fragment_tanimoto(current_rgroup_smi, frag)
    )
    mask_sim = sims >= similarity_threshold
    if not mask_sim.any():
        return pd.DataFrame()

    similar = candidate_transforms[mask_sim].copy()
    similar["match_type"]       = "similarity"
    similar["match_similarity"] = sims[mask_sim].values
    return similar.sort_values("match_similarity", ascending=False)


def _adapt_to_fragment(to_frag_smi: str, map_num: int) -> str:
    """Replace attachment dummy in to_frag_smi with [*:map_num]."""
    mol = Chem.MolFromSmiles(to_frag_smi)
    if mol is None:
        raise ValueError(f"Invalid to_fragment SMILES: {to_frag_smi}")
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(map_num)
    return Chem.MolToSmiles(mol)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Candidate ranking
# ──────────────────────────────────────────────────────────────────────────────

def _collateral_change(
    parent_feats: tuple[int, ...],
    gen_feats: tuple[int, ...],
    target_idx: int,
) -> int:
    """Sum |Δ| for all non-target descriptors (lower = fewer side-effects)."""
    return sum(
        abs(gen_feats[i] - parent_feats[i])
        for i in range(len(parent_feats))
        if i != target_idx
    )


def _rank_candidates(candidates: list[dict], target_idx: int) -> list[dict]:
    """
    Sort by:
        1. collateral_change ascending
        2. match_type: exact before similarity
        3. transformation_frequency descending
        4. match_similarity descending
    """
    def key(c: dict) -> tuple:
        return (
            c["collateral_change"],
            0 if c.get("match_type", "exact") == "exact" else 1,
            -c.get("transformation_frequency", 0),
            -c.get("match_similarity", 1.0),
        )
    return sorted(candidates, key=key)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Diversity-aware MaxMin selection
# ──────────────────────────────────────────────────────────────────────────────

def _maxmin_select(
    candidates: list[dict],
    n: int,
    fp_radius: int = 2,
    fp_bits: int = 2048,
) -> list[dict]:
    """
    Select up to *n* structurally diverse candidates using MaxMin on
    Morgan fingerprints.  Applied AFTER all filtering and ranking.
    """
    if len(candidates) <= n:
        return candidates

    fps, valid_indices = [], []
    for idx, rec in enumerate(candidates):
        mol = Chem.MolFromSmiles(rec["generated_smiles"])
        if mol is not None:
            fps.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, fp_radius, nBits=fp_bits
            ))
            valid_indices.append(idx)

    if not valid_indices:
        return candidates[:n]

    selected_vi = [0]
    remaining_vi = list(range(1, len(valid_indices)))

    while len(selected_vi) < n and remaining_vi:
        best_vi, best_min_dist = None, -1.0
        for vi in remaining_vi:
            min_dist = min(
                1.0 - DataStructs.TanimotoSimilarity(fps[vi], fps[sel])
                for sel in selected_vi
            )
            if min_dist > best_min_dist:
                best_min_dist, best_vi = min_dist, vi
        if best_vi is None:
            break
        selected_vi.append(best_vi)
        remaining_vi.remove(best_vi)

    return [candidates[valid_indices[vi]] for vi in selected_vi]


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Main API
# ──────────────────────────────────────────────────────────────────────────────

def propose_structures(
    parent_smiles: str,
    # primary API
    target: Optional[str] = None,
    delta: Optional[int] = None,
    transformation_library: Optional[pd.DataFrame] = None,
    # legacy API
    mode: Optional[str] = None,
    # shared options
    max_cuts: Optional[int] = None,
    max_candidates: int = 10,
    shuffle_sites: bool = False,
    random_seed: Optional[int] = None,
    strict: bool = False,
    # matching
    match_mode: str = "exact",
    similarity_threshold: float = 0.4,
    # diversity
    diversity: bool = False,
) -> list[dict]:
    """
    Propose candidate structures in which one descriptor changes by exactly
    *delta* units, using only transformations from the library whose
    from_fragment exactly matches the parent R-group (default).

    Primary API (v2)
    ----------------
    propose_structures(
        parent_smiles,
        target="HBA",   # HBD | HBA | AR | RotB | HA
        delta=1,
        transformation_library=lib,
    )

    Legacy API (backward-compatible)
    ---------------------------------
    propose_structures(parent_smiles, mode="add_hba", transformation_library=lib)

    Parameters
    ----------
    match_mode          : "exact" (default) or "similarity" (exploratory only)
    similarity_threshold: Tanimoto threshold for similarity fallback
    diversity           : if True, apply MaxMin selection on final results
    strict              : if True, all non-target descriptors must be unchanged

    Returns
    -------
    List of dicts; each contains:
        parent_smiles, generated_smiles,
        target_descriptor, requested_delta, library_delta, actual_delta,
        site, from_fragment, to_fragment,
        transformation_frequency, transformation_source,
        match_type, match_similarity,
        parent_scaffold_smiles, generated_murcko_scaffold,
        scaffold_preserved, collateral_change,
        parent_features, generated_features
    """
    # ── resolve target / delta ────────────────────────────────────────────────
    if mode is not None and target is None:
        if mode not in _MODE_MAP:
            raise ValueError(
                f"Unknown mode '{mode}'. "
                f"Supported modes: {sorted(_MODE_MAP.keys())}. "
                "Note: 'remove_ar' is not supported in v2."
            )
        target, delta = _MODE_MAP[mode]

    if target is None or delta is None:
        raise ValueError(
            "Specify target+delta (e.g. target='HBA', delta=1) "
            "or the legacy mode string (e.g. mode='add_hba')."
        )

    if (target, delta) not in _SUPPORTED:
        if target == "AR" and delta == -1:
            raise ValueError(
                "AR -1 (remove_ar) is not supported in v2. "
                "Removing an aromatic ring from an R-group contradicts "
                "the parent scaffold retention guarantee."
            )
        raise ValueError(
            f"Unsupported (target, delta) = ({target!r}, {delta}). "
            f"Supported: {sorted(_SUPPORTED)}."
        )

    if transformation_library is None:
        raise ValueError(
            "transformation_library is required. "
            "Load one with load_transformation_library(path)."
        )

    target_idx = _DESC_INDEX[target]

    if random_seed is not None:
        random.seed(random_seed)

    # ── scaffold / R-group decomposition ─────────────────────────────────────
    parent_mol, scaffold_mol, core = make_murcko_core_with_rlabels(
        parent_smiles, max_cuts=max_cuts
    )
    core_smiles           = Chem.MolToSmiles(core)
    labels                = get_r_labels(core)
    if not labels:
        return []

    parent_canon          = Chem.MolToSmiles(parent_mol)
    parent_feats          = count_features(parent_mol)
    parent_scaffold_smi   = Chem.MolToSmiles(scaffold_mol)
    original_rgroups      = get_original_rgroups(parent_smiles, core)

    # ── Step 1: descriptor-indexed retrieval ─────────────────────────────────
    candidate_transforms = retrieve_transformations(
        transformation_library, target, delta
    )
    if candidate_transforms.empty:
        return []

    if shuffle_sites:
        labels = random.sample(labels, len(labels))

    # ── Steps 2–8: generation and validation ─────────────────────────────────
    all_candidates: list[dict] = []
    seen: set[str] = set()

    for label in labels:
        site = f"R{label}"
        current_rgroup_smi = original_rgroups.get(site)
        if current_rgroup_smi is None:
            continue

        # Step 2: R-group matching (exact by default; no silent fallback)
        applicable = _get_applicable_transforms(
            current_rgroup_smi, candidate_transforms,
            match_mode, similarity_threshold
        )
        if applicable.empty:
            continue

        for _, row in applicable.iterrows():
            try:
                # Step 3: generate candidate
                new_frag_smi = _adapt_to_fragment(row["to_fragment"], label)
                rgroups      = {**original_rgroups, site: new_frag_smi}
                gen_mol      = build_molecule(core_smiles, rgroups)
                gen_canon    = Chem.MolToSmiles(gen_mol)

                if gen_canon == parent_canon or gen_canon in seen:
                    continue
                seen.add(gen_canon)

                # Step 5: parent scaffold retained as substructure
                scaffold_preserved = bool(
                    gen_mol.HasSubstructMatch(scaffold_mol)
                )
                if not scaffold_preserved:
                    continue

                # Step 6: whole-molecule descriptor recomputation
                gen_feats = count_features(gen_mol)

                gen_scaffold_mol = MurckoScaffold.GetScaffoldForMol(gen_mol)
                gen_scaffold_smi = (
                    Chem.MolToSmiles(gen_scaffold_mol)
                    if gen_scaffold_mol is not None else ""
                )

                # Step 7: exact requested delta validation
                # library_delta is fragment-level (pre-filter); actual_delta
                # is whole-molecule (final criterion).
                actual_delta  = gen_feats[target_idx] - parent_feats[target_idx]
                library_delta = int(row.get(f"delta_{target}", delta))

                if actual_delta != delta:
                    continue

                # Step 8: strict filtering
                other_unchanged = all(
                    gen_feats[i] == parent_feats[i]
                    for i in range(len(parent_feats))
                    if i != target_idx
                )
                if strict and not other_unchanged:
                    continue

                collateral = _collateral_change(parent_feats, gen_feats, target_idx)

                all_candidates.append({
                    "parent_smiles":             parent_canon,
                    "generated_smiles":          gen_canon,
                    "target_descriptor":         target,
                    "requested_delta":           delta,
                    "library_delta":             library_delta,
                    "actual_delta":              actual_delta,
                    "site":                      site,
                    "from_fragment":             row["from_fragment"],
                    "to_fragment":               row["to_fragment"],
                    "transformation_frequency":  int(row.get("frequency", 1)),
                    "transformation_source":     str(row.get("source", "unknown")),
                    "match_type":                str(row.get("match_type", "exact")),
                    "match_similarity":          float(row.get("match_similarity", 1.0)),
                    "parent_scaffold_smiles":    parent_scaffold_smi,
                    "generated_murcko_scaffold": gen_scaffold_smi,
                    "scaffold_preserved":        scaffold_preserved,
                    "collateral_change":         collateral,
                    "parent_features":           parent_feats,
                    "generated_features":        gen_feats,
                })

            except Exception:
                continue

    if not all_candidates:
        return []

    # Step 9: ranking
    ranked = _rank_candidates(all_candidates, target_idx)

    # Step 10: optional diversity selection
    if diversity:
        return _maxmin_select(ranked, max_candidates)

    return ranked[:max_candidates]


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Display helper
# ──────────────────────────────────────────────────────────────────────────────

def print_proposals(results: list[dict]) -> None:
    """Pretty-print results from propose_structures()."""
    if not results:
        print("No valid candidates found.")
        return

    for i, rec in enumerate(results, start=1):
        print(f"=== Candidate {i} ===")
        print(
            f"target: {rec['target_descriptor']}  "
            f"requested: {rec['requested_delta']:+d}  "
            f"library: {rec['library_delta']:+d}  "
            f"actual: {rec['actual_delta']:+d}  "
            f"site: {rec['site']}  "
            f"collateral: {rec['collateral_change']}"
        )
        print(f"from → to : {rec['from_fragment']} → {rec['to_fragment']}")
        print(
            f"freq: {rec['transformation_frequency']}  "
            f"source: {rec['transformation_source']}  "
            f"match: {rec['match_type']} ({rec['match_similarity']:.3f})"
        )
        print(
            f"scaffold_preserved: {rec['scaffold_preserved']}  "
            f"parent scaffold: {rec['parent_scaffold_smiles']}"
        )
        print(f"parent   : {rec['parent_smiles']}")
        print(f"generated: {rec['generated_smiles']}")
        print(f"features {DESCRIPTOR_NAMES}")
        print(f"  parent   : {rec['parent_features']}")
        print(f"  generated: {rec['generated_features']}")
        print()


# ──────────────────────────────────────────────────────────────────────────────
# 10.  Version
# ──────────────────────────────────────────────────────────────────────────────

from scaffold_tuner import __version__  # noqa: E402


def get_version() -> str:
    """Return the installed version of scaffold-tuner."""
    return __version__
