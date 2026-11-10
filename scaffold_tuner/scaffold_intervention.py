# -*- coding: utf-8 -*-
"""
scaffold_intervention.py  –  ScaffoldTuner

For a given parent molecule (SMILES) and a desired descriptor change
(target, delta), this module proposes modified candidate structures by:

    1. Extracting the Murcko scaffold and R-group sites
    2. Substituting each of the 25 R-groups from the unified pool
       at every scaffold attachment point
    3. Recalculating whole-molecule descriptors with RDKit
    4. Retaining only candidates where actual_delta == requested_delta
    5. Optionally enforcing that no other descriptor changes (strict=True)

The unified R-group pool contains 25 literature-informed fragments
(rgroup_pool.R_GROUP_SMILES).  No descriptor-specific sub-pool is used.

AR -1 (remove_ar) is not supported in the current scaffold-retaining
workflow and raises NotImplementedError.
"""

from __future__ import annotations

import random
from typing import Optional

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdRGroupDecomposition
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from .rgroup_pool import R_GROUP_SMILES


# ──────────────────────────────────────────────────────────────────────────────
# Descriptor helpers
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

# Legacy mode strings for backward compatibility
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


# ──────────────────────────────────────────────────────────────────────────────
# Scaffold and R-group extraction
# ──────────────────────────────────────────────────────────────────────────────

def make_murcko_core_with_rlabels(
    parent_smiles: str,
    max_cuts: Optional[int] = None,
) -> tuple[Chem.Mol, Chem.Mol, Chem.Mol]:
    """
    Extract the Murcko scaffold and label each scaffold–sidechain boundary
    bond with a dummy atom [*:N].

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
    """Return sorted attachment-site map numbers from the labeled core."""
    return sorted({
        atom.GetAtomMapNum()
        for atom in core_mol.GetAtoms()
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() > 0
    })


def get_original_rgroups(
    parent_smiles: str,
    core_with_labels: Chem.Mol,
) -> dict[str, str]:
    """Return {"R1": SMILES, "R2": SMILES, ...} for each attachment site."""
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
# Fragment attachment
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


def _attach_fragment(
    scaffold_mol: Chem.Mol,
    fragment_mol: Chem.Mol,
    map_num: int,
) -> Chem.Mol:
    """Join scaffold and fragment at [*:map_num] and return the product."""
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


def _build_molecule(scaffold_smiles: str, rgroups: dict[str, str]) -> Chem.Mol:
    """Attach all R-groups to the scaffold in ascending site-number order."""
    mol = Chem.MolFromSmiles(scaffold_smiles)
    if mol is None:
        raise ValueError("Invalid scaffold SMILES.")
    for site in sorted(rgroups, key=lambda x: int(x[1:])):
        frag = Chem.MolFromSmiles(rgroups[site])
        if frag is None:
            raise ValueError(f"Invalid fragment SMILES: {rgroups[site]}")
        mol = _attach_fragment(mol, frag, int(site[1:]))
    return mol


# ──────────────────────────────────────────────────────────────────────────────
# Fragment normalization and pool utilities
# ──────────────────────────────────────────────────────────────────────────────

def normalize_fragment_smiles(smi: str) -> str:
    """Canonicalize a fragment SMILES; normalize the attachment dummy to [*:1]."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(1)
    return Chem.MolToSmiles(mol)


def _adapt_to_site(pool_smi: str, map_num: int) -> str:
    """Replace the [*:1] dummy in a pool SMILES with [*:map_num]."""
    mol = Chem.MolFromSmiles(pool_smi)
    if mol is None:
        raise ValueError(f"Invalid pool SMILES: {pool_smi}")
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(map_num)
    return Chem.MolToSmiles(mol)


# ──────────────────────────────────────────────────────────────────────────────
# Public utility: R-group extraction (shared with evaluation scripts)
# ──────────────────────────────────────────────────────────────────────────────

def extract_rgroups_for_smiles(
    parent_smiles: str,
    max_cuts: Optional[int] = None,
) -> list[str] | None:
    """
    Extract and normalize the R-group SMILES for a parent molecule.

    Returns a list of normalized R-group SMILES (attachment dummy = [*:1]),
    or None if extraction fails for any reason.
    """
    try:
        _, _, core = make_murcko_core_with_rlabels(parent_smiles, max_cuts=max_cuts)
        rgroups    = get_original_rgroups(parent_smiles, core)
    except Exception:
        return None
    if not rgroups:
        return None
    return [normalize_fragment_smiles(smi) for smi in rgroups.values()]


# ──────────────────────────────────────────────────────────────────────────────
# Ranking and diversity selection
# ──────────────────────────────────────────────────────────────────────────────

def _collateral_change(
    parent_feats: tuple[int, ...],
    gen_feats: tuple[int, ...],
    target_idx: int,
) -> int:
    """Sum of |Δ| for all non-target descriptors."""
    return sum(
        abs(gen_feats[i] - parent_feats[i])
        for i in range(len(parent_feats))
        if i != target_idx
    )


def _maxmin_select(
    candidates: list[dict],
    n: int,
    fp_radius: int = 2,
    fp_bits: int = 2048,
) -> list[dict]:
    """
    Select up to *n* structurally diverse candidates using MaxMin on
    Morgan fingerprints.  Applied after all filtering and ranking.
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
# Main API
# ──────────────────────────────────────────────────────────────────────────────

def propose_structures(
    parent_smiles: str,
    target: Optional[str] = None,
    delta: Optional[int] = None,
    mode: Optional[str] = None,       # legacy mode string
    max_cuts: Optional[int] = None,
    max_candidates: int = 10,
    shuffle_sites: bool = False,
    random_seed: Optional[int] = None,
    strict: bool = False,
    diversity: bool = False,
) -> list[dict]:
    """
    Propose candidate structures in which one molecular descriptor changes
    by exactly *delta* units.

    All 25 R-groups from the unified pool (rgroup_pool.R_GROUP_SMILES) are
    evaluated at every scaffold attachment site.  Candidates are accepted
    only when the whole-molecule RDKit descriptor satisfies:

        actual_delta == requested_delta

    Parameters
    ----------
    parent_smiles  : SMILES of the parent molecule
    target         : descriptor to change — HBD, HBA, AR, RotB, or HA
    delta          : requested change — +1 or -1 (AR supports +1 only)
    mode           : legacy mode string, e.g. "add_hba" (overrides target/delta)
    max_cuts       : maximum number of R-group sites to edit (None = all)
    max_candidates : maximum number of candidates to return
    shuffle_sites  : randomize site evaluation order (requires random_seed)
    random_seed    : seed for reproducibility
    strict         : if True, only return candidates where no non-target
                     descriptor changes
    diversity      : if True, apply MaxMin diversity selection on the
                     final candidate set

    Returns
    -------
    List of dicts, each with:
        parent_smiles, generated_smiles,
        target_descriptor, requested_delta, actual_delta,
        site, to_fragment,
        parent_scaffold_smiles, generated_murcko_scaffold,
        scaffold_preserved, collateral_change,
        parent_features, generated_features
    """
    # ── resolve target / delta ────────────────────────────────────────────────
    if mode is not None and target is None:
        if mode == "remove_ar":
            raise NotImplementedError(
                "Aromatic-ring removal is reserved for future workflow backends "
                "and is not supported in the current scaffold-retaining implementation."
            )
        if mode not in _MODE_MAP:
            raise ValueError(
                f"Unknown mode '{mode}'. "
                f"Supported modes: {sorted(_MODE_MAP.keys())}."
            )
        target, delta = _MODE_MAP[mode]

    if target is None or delta is None:
        raise ValueError(
            "Specify target+delta (e.g. target='HBA', delta=1) "
            "or a legacy mode string (e.g. mode='add_hba')."
        )

    if target == "AR" and delta == -1:
        raise NotImplementedError(
            "Aromatic-ring removal is reserved for future workflow backends "
            "and is not supported in the current scaffold-retaining implementation."
        )

    if (target, delta) not in _SUPPORTED:
        raise ValueError(
            f"Unsupported (target, delta) = ({target!r}, {delta}). "
            f"Supported combinations: {sorted(_SUPPORTED)}."
        )

    target_idx = _DESC_INDEX[target]

    if random_seed is not None:
        random.seed(random_seed)

    # ── scaffold and R-group extraction ──────────────────────────────────────
    parent_mol, scaffold_mol, core = make_murcko_core_with_rlabels(
        parent_smiles, max_cuts=max_cuts
    )
    core_smiles         = Chem.MolToSmiles(core)
    labels              = get_r_labels(core)
    if not labels:
        return []

    parent_canon        = Chem.MolToSmiles(parent_mol)
    parent_feats        = count_features(parent_mol)
    parent_scaffold_smi = Chem.MolToSmiles(scaffold_mol)
    original_rgroups    = get_original_rgroups(parent_smiles, core)

    if shuffle_sites:
        labels = random.sample(labels, len(labels))

    # ── candidate generation and validation ───────────────────────────────────
    all_candidates: list[dict] = []
    seen: set[str] = set()

    for label in labels:
        site = f"R{label}"
        for pool_idx, pool_smi in enumerate(R_GROUP_SMILES):
            try:
                adapted   = _adapt_to_site(pool_smi, label)
                rgroups   = {**original_rgroups, site: adapted}
                gen_mol   = _build_molecule(core_smiles, rgroups)
                gen_canon = Chem.MolToSmiles(gen_mol)
            except Exception:
                continue

            if gen_canon == parent_canon or gen_canon in seen:
                continue

            # Scaffold retention check
            if not gen_mol.HasSubstructMatch(scaffold_mol):
                continue

            # Whole-molecule descriptor recomputation
            gen_feats = count_features(gen_mol)

            gen_scaffold_mol = MurckoScaffold.GetScaffoldForMol(gen_mol)
            gen_scaffold_smi = (
                Chem.MolToSmiles(gen_scaffold_mol)
                if gen_scaffold_mol is not None else ""
            )

            # Exact delta validation (whole molecule)
            actual_delta = gen_feats[target_idx] - parent_feats[target_idx]
            if actual_delta != delta:
                continue

            # Strict mode: no other descriptor may change
            if strict and any(
                gen_feats[i] != parent_feats[i]
                for i in range(len(parent_feats))
                if i != target_idx
            ):
                continue

            seen.add(gen_canon)

            collateral = _collateral_change(parent_feats, gen_feats, target_idx)
            all_candidates.append({
                "parent_smiles":             parent_canon,
                "generated_smiles":          gen_canon,
                "target_descriptor":         target,
                "requested_delta":           delta,
                "actual_delta":              actual_delta,
                "site":                      site,
                "to_fragment":               normalize_fragment_smiles(pool_smi),
                "parent_scaffold_smiles":    parent_scaffold_smi,
                "generated_murcko_scaffold": gen_scaffold_smi,
                "scaffold_preserved":        True,
                "collateral_change":         collateral,
                "parent_features":           parent_feats,
                "generated_features":        gen_feats,
                "_pool_index":               pool_idx,
            })

    if not all_candidates:
        return []

    # Ranking: collateral_change ascending, then pool order
    ranked = sorted(
        all_candidates,
        key=lambda c: (c["collateral_change"], c["_pool_index"]),
    )
    for rec in ranked:
        rec.pop("_pool_index")

    if diversity:
        return _maxmin_select(ranked, max_candidates)
    return ranked[:max_candidates]


# ──────────────────────────────────────────────────────────────────────────────
# Display helper
# ──────────────────────────────────────────────────────────────────────────────

def print_proposals(results: list[dict]) -> None:
    """Pretty-print the list returned by propose_structures()."""
    if not results:
        print("No valid candidates found.")
        return
    for i, rec in enumerate(results, start=1):
        print(f"=== Candidate {i} ===")
        print(
            f"target: {rec['target_descriptor']}  "
            f"requested: {rec['requested_delta']:+d}  "
            f"actual: {rec['actual_delta']:+d}  "
            f"site: {rec['site']}  "
            f"collateral: {rec['collateral_change']}"
        )
        print(f"to_fragment      : {rec['to_fragment']}")
        print(f"scaffold_preserved: {rec['scaffold_preserved']}")
        print(f"parent   : {rec['parent_smiles']}")
        print(f"generated: {rec['generated_smiles']}")
        print(f"features {DESCRIPTOR_NAMES}")
        print(f"  parent   : {rec['parent_features']}")
        print(f"  generated: {rec['generated_features']}")
        print()


# ──────────────────────────────────────────────────────────────────────────────
# Version
# ──────────────────────────────────────────────────────────────────────────────

from scaffold_tuner import __version__  # noqa: E402


def get_version() -> str:
    """Return the installed version of scaffold-tuner."""
    return __version__
