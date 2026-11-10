#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_transformation_library.py  –  ScaffoldTuner v2

Generate a descriptor-indexed R-group transformation library from a set of
input compounds (SMILES).

Typical usage
-------------
    python build_transformation_library.py \
        --input chembl_compounds.csv \
        --output transformation_library.csv \
        --smiles-col smiles \
        --id-col chembl_id \
        --min-frequency 2 \
        --max-heavy-atoms 10

The script:
    1. Reads input SMILES (CSV / TSV / .smi).
    2. For every compound, extracts the Murcko scaffold and decomposes
       R-groups with RDKit.
    3. Groups compounds by scaffold canonical SMILES.
    4. Within each group, finds all pairs sharing the same scaffold and
       differing at exactly one R-group site (single-site transformation
       pairs, analogous to matched molecular pairs).
    5. Computes fragment-level descriptor deltas for each pair.
    6. Aggregates identical (from_fragment, to_fragment) pairs and counts
       their frequency.
    7. Filters by min_frequency and writes the library CSV.
    8. Prints a detailed summary:
         - source compound and scaffold counts  (from source data, not
           from the aggregated DataFrame)
         - per-descriptor transformation counts and observations
         - frequency distribution

Note on delta columns
---------------------
All delta_* columns are computed at the FRAGMENT level.  They are used
as a retrieval pre-filter in propose_structures() only.  The final
acceptance criterion is the whole-molecule actual_delta recomputed by
RDKit after generation.  Fragment-level and whole-molecule deltas can
differ, especially for RotB.
"""

import argparse
import itertools
import logging
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdRGroupDecomposition
from rdkit.Chem.Scaffolds import MurckoScaffold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Descriptor helpers
# ──────────────────────────────────────────────────────────────────────────────

def _count_features(mol: Chem.Mol) -> tuple:
    """Return (HBA, HBD, NumAromaticRings, NumRotatableBonds, NumHeteroatoms)."""
    return (
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumHDonors(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumHeteroatoms(mol),
    )


def _normalize_attachment(smi: str) -> str:
    """Canonicalize fragment SMILES, normalizing attachment dummy to [*:1]."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(1)
    return Chem.MolToSmiles(mol)


def _heavy_atom_count(smi: str) -> int:
    """Count non-hydrogen, non-dummy atoms in a fragment SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (0, 1))


def _mol_wt_fragment(smi: str) -> float:
    """Return molecular weight of the fragment (including the dummy atom)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0.0
    return Descriptors.MolWt(mol)


# ──────────────────────────────────────────────────────────────────────────────
# Scaffold / R-group extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_rgroups(smiles: str, max_heavy_atoms: int) -> dict | None:
    """
    Extract the Murcko scaffold and all R-groups for a single compound.

    Returns:
        {
            "scaffold_smi": canonical scaffold SMILES,
            "rgroups": {"R1": frag_smi, "R2": frag_smi, ...}
        }
    or None if extraction fails.

    The scaffold_smi is stored in the returned dict so that scaffold
    statistics (unique scaffold count) can be computed from the source
    data directly, before the transformation aggregation step.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None:
        return None

    matches = mol.GetSubstructMatches(scaffold)
    if not matches:
        return None

    match = matches[0]
    core_atom_idxs = set(match)
    parent_to_core  = {p: c for c, p in enumerate(match)}

    boundary_bonds = [
        b for b in mol.GetBonds()
        if (b.GetBeginAtomIdx() in core_atom_idxs) ^ (b.GetEndAtomIdx() in core_atom_idxs)
    ]
    if not boundary_bonds:
        return None

    rw_core = Chem.RWMol(scaffold)
    for rlabel, bond in enumerate(boundary_bonds, start=1):
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        core_parent_idx = a1 if a1 in core_atom_idxs else a2
        core_idx = parent_to_core[core_parent_idx]

        dummy = Chem.Atom(0)
        dummy.SetAtomMapNum(rlabel)
        didx = rw_core.AddAtom(dummy)
        rw_core.AddBond(core_idx, didx, bond.GetBondType())

    try:
        core_with_labels = rw_core.GetMol()
        Chem.SanitizeMol(core_with_labels)
    except Exception:
        return None

    params = rdRGroupDecomposition.RGroupDecompositionParameters()
    params.labels = rdRGroupDecomposition.RGroupLabels.AtomMapLabels
    params.rgroupLabelling = rdRGroupDecomposition.RGroupLabelling.AtomMap
    params.onlyMatchAtRGroups = True

    rgd = rdRGroupDecomposition.RGroupDecomposition(core_with_labels, params)
    if rgd.Add(mol) < 0:
        return None
    if not rgd.Process():
        return None

    rows = rgd.GetRGroupsAsRows()
    if not rows:
        return None

    rgroups = {}
    for key, fmol in rows[0].items():
        if key == "Core" or fmol is None:
            continue
        frag_smi = Chem.MolToSmiles(fmol)
        if _heavy_atom_count(frag_smi) > max_heavy_atoms:
            continue
        rgroups[key] = frag_smi

    if not rgroups:
        return None

    return {
        "scaffold_smi": Chem.MolToSmiles(scaffold),   # canonical Murcko SMILES
        "rgroups":      rgroups,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Transformation extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_transformations(records: list[dict], source: str) -> list[dict]:
    """
    Find all single-site R-group transformation pairs and compute
    fragment-level descriptor deltas.

    A valid pair (A, B) must:
    - Share the same Murcko scaffold canonical SMILES
    - Have the same set of R-group site labels
    - Differ at exactly one R-group site

    Both directions (A→B and B→A) are recorded.
    """
    by_scaffold: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_scaffold[rec["scaffold_smi"]].append(rec)

    transformations: list[dict] = []

    for scaffold_smi, group in by_scaffold.items():
        if len(group) < 2:
            continue

        for rec_a, rec_b in itertools.combinations(group, 2):
            rg_a = rec_a["rgroups"]
            rg_b = rec_b["rgroups"]

            if set(rg_a.keys()) != set(rg_b.keys()):
                continue

            diff_sites = [
                site for site in rg_a
                if _normalize_attachment(rg_a[site]) != _normalize_attachment(rg_b[site])
            ]

            if len(diff_sites) != 1:
                continue

            site = diff_sites[0]
            frag_a_smi = _normalize_attachment(rg_a[site])
            frag_b_smi = _normalize_attachment(rg_b[site])

            mol_a = Chem.MolFromSmiles(frag_a_smi)
            mol_b = Chem.MolFromSmiles(frag_b_smi)
            if mol_a is None or mol_b is None:
                continue

            feats_a = _count_features(mol_a)
            feats_b = _count_features(mol_b)
            mw_a    = _mol_wt_fragment(frag_a_smi)
            mw_b    = _mol_wt_fragment(frag_b_smi)

            for from_smi, to_smi, from_feats, to_feats, from_mw, to_mw in [
                (frag_a_smi, frag_b_smi, feats_a, feats_b, mw_a, mw_b),
                (frag_b_smi, frag_a_smi, feats_b, feats_a, mw_b, mw_a),
            ]:
                transformations.append({
                    "from_fragment":    from_smi,
                    "to_fragment":      to_smi,
                    "delta_HBA":        to_feats[0] - from_feats[0],
                    "delta_HBD":        to_feats[1] - from_feats[1],
                    "delta_AR":         to_feats[2] - from_feats[2],
                    "delta_RotB":       to_feats[3] - from_feats[3],
                    "delta_HA":         to_feats[4] - from_feats[4],
                    "delta_MW":         round(to_mw - from_mw, 4),
                    "from_heavy_atoms": _heavy_atom_count(from_smi),
                    "to_heavy_atoms":   _heavy_atom_count(to_smi),
                    "source":           source,
                })

    return transformations


# ──────────────────────────────────────────────────────────────────────────────
# Summary statistics  (FIX: unique scaffold count from source data)
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(
    df_filtered: pd.DataFrame,
    n_source_compounds: int,
    n_rgroup_records: int,
    n_failed: int,
    rgroup_records: list[dict],           # source-side records for scaffold count
) -> None:
    """
    Print a detailed summary of the transformation library.

    Unique Murcko scaffold count is computed from *rgroup_records* (the
    list of successful R-group extraction results), NOT from the aggregated
    transformation DataFrame.  This ensures the number reflects the actual
    scaffold diversity of the source dataset rather than only scaffolds
    that produced at least one transformation pair.
    """
    tasks = [
        ("HBD", +1, "delta_HBD"),
        ("HBD", -1, "delta_HBD"),
        ("HBA", +1, "delta_HBA"),
        ("HBA", -1, "delta_HBA"),
        ("RotB",+1, "delta_RotB"),
        ("RotB",-1, "delta_RotB"),
        ("AR",  +1, "delta_AR"),
    ]

    # Unique scaffold count from source data (fix for N/A issue)
    n_unique_scaffolds = len({rec["scaffold_smi"] for rec in rgroup_records})

    n_unique_rgroups = len(
        set(df_filtered["from_fragment"].tolist())
        | set(df_filtered["to_fragment"].tolist())
    )
    n_unique_trans = len(df_filtered)
    total_obs      = int(df_filtered["frequency"].sum())

    print("\n" + "=" * 60)
    print("  Transformation Library Summary")
    print("=" * 60)
    print(f"  Source compounds         : {n_source_compounds:>10,}")
    print(f"  R-group extraction OK    : {n_rgroup_records:>10,}")
    print(f"  R-group extraction failed: {n_failed:>10,}")
    print(f"  Unique Murcko scaffolds  : {n_unique_scaffolds:>10,}")
    print(f"  Unique R-groups          : {n_unique_rgroups:>10,}")
    print(f"  Unique transformations   : {n_unique_trans:>10,}")
    print(f"  Total observations       : {total_obs:>10,}")

    print()
    print("  Descriptor-indexed transformations")
    print("  " + "-" * 44)
    print(f"  {'Task':<10}  {'Unique trans':>14}  {'Total obs':>12}")
    print("  " + "-" * 44)
    for desc, dlt, col in tasks:
        if col not in df_filtered.columns:
            continue
        sub = df_filtered[df_filtered[col] == dlt]
        print(
            f"  {desc} {dlt:+d}    "
            f"{len(sub):>14,}  "
            f"{int(sub['frequency'].sum()):>12,}"
        )

    print()
    print("  Frequency distribution")
    print("  " + "-" * 30)
    for min_f, label in [
        (None, "freq == 1"),
        (2,    "freq >= 2"),
        (5,    "freq >= 5"),
        (10,   "freq >= 10"),
    ]:
        if min_f is None:
            n = int((df_filtered["frequency"] == 1).sum())
        else:
            n = int((df_filtered["frequency"] >= min_f).sum())
        print(f"  {label:<12}: {n:>8,}")

    print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Input reading
# ──────────────────────────────────────────────────────────────────────────────

def _read_smiles(path: Path, smiles_col: str, id_col: str | None) -> list[dict]:
    """Read input file and return list of {"smiles": ..., "id": ...} dicts."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".tsv", ".txt"):
        try:
            df = pd.read_csv(path, sep="\t")
            if smiles_col not in df.columns:
                raise ValueError
        except Exception:
            rows = []
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    rows.append({
                        smiles_col: parts[0],
                        "id": parts[1] if len(parts) > 1 else f"mol_{len(rows)}",
                    })
            df = pd.DataFrame(rows)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported input format: {suffix}")

    if smiles_col not in df.columns:
        raise ValueError(
            f"SMILES column '{smiles_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    records = []
    for _, row in df.iterrows():
        smi = str(row[smiles_col]).strip()
        cid = str(row[id_col]) if id_col and id_col in df.columns else str(len(records))
        records.append({"smiles": smi, "id": cid})
    return records


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a ScaffoldTuner v2 transformation library from a SMILES dataset."
        )
    )
    parser.add_argument("--input",  "-i", required=True,
        help="Input file (CSV / TSV / .smi).")
    parser.add_argument("--output", "-o", default="transformation_library.csv",
        help="Output CSV path (default: transformation_library.csv).")
    parser.add_argument("--smiles-col", default="smiles",
        help="Name of the SMILES column (default: smiles).")
    parser.add_argument("--id-col", default=None,
        help="Name of the compound ID column (optional).")
    parser.add_argument("--source", default="ChEMBL",
        help="Source label written to the 'source' column (default: ChEMBL).")
    parser.add_argument("--min-frequency", type=int, default=2,
        help="Minimum observed count to include a transformation (default: 2).")
    parser.add_argument("--max-heavy-atoms", type=int, default=10,
        help="Max R-group size in heavy atoms, excluding [*] (default: 10).")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    log.info("Reading input: %s", input_path)
    raw_records = _read_smiles(input_path, args.smiles_col, args.id_col)
    log.info("  %d compounds read", len(raw_records))

    log.info("Extracting Murcko scaffolds and R-groups ...")
    rgroup_records: list[dict] = []
    n_failed = 0
    for i, rec in enumerate(raw_records):
        if i % 1000 == 0 and i > 0:
            log.info("  processed %d / %d", i, len(raw_records))
        result = _extract_rgroups(rec["smiles"], args.max_heavy_atoms)
        if result is None:
            n_failed += 1
            continue
        rgroup_records.append(result)

    log.info("  R-group extraction: %d succeeded, %d failed",
             len(rgroup_records), n_failed)

    if not rgroup_records:
        log.error("No valid R-group records found.  Check your input file.")
        sys.exit(1)

    log.info("Extracting R-group transformations ...")
    transformations = _extract_transformations(rgroup_records, args.source)
    log.info("  %d raw transformation instances found", len(transformations))

    if not transformations:
        log.error(
            "No transformations found.  The dataset may be too small or "
            "contain only compounds without shared scaffolds."
        )
        sys.exit(1)

    log.info("Aggregating (min_frequency=%d) ...", args.min_frequency)
    df = pd.DataFrame(transformations)
    group_cols = [
        "from_fragment", "to_fragment",
        "delta_HBA", "delta_HBD", "delta_AR", "delta_RotB", "delta_HA",
        "delta_MW", "from_heavy_atoms", "to_heavy_atoms", "source",
    ]
    df_agg = (
        df.groupby(group_cols, as_index=False)
          .size()
          .rename(columns={"size": "frequency"})
    )
    df_filtered = (
        df_agg[df_agg["frequency"] >= args.min_frequency]
        .copy()
        .sort_values("frequency", ascending=False)
        .reset_index(drop=True)
    )
    log.info("  %d unique transformations; %d pass min_frequency=%d",
             len(df_agg), len(df_filtered), args.min_frequency)

    col_order = [
        "from_fragment", "to_fragment",
        "delta_HBD", "delta_HBA", "delta_AR", "delta_RotB", "delta_HA",
        "delta_MW", "from_heavy_atoms", "to_heavy_atoms",
        "frequency", "source",
    ]
    df_out = df_filtered[[c for c in col_order if c in df_filtered.columns]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    log.info("Transformation library written to: %s", output_path)

    # Print detailed summary (unique scaffold count from source data)
    _print_summary(
        df_filtered,
        n_source_compounds=len(raw_records),
        n_rgroup_records=len(rgroup_records),
        n_failed=n_failed,
        rgroup_records=rgroup_records,    # passed for accurate scaffold count
    )


if __name__ == "__main__":
    main()
