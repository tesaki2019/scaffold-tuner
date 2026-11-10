#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_applicability.py  –  ScaffoldTuner v2

Compute exact-match applicability of the transformation library against
a set of benchmark parent molecules.

For each supported descriptor task (HBD ±1, HBA ±1, RotB ±1, AR +1), a
parent molecule is "applicable" if at least one transformation in the
library satisfies:

    from_fragment == parent_R-group  (exact canonical SMILES)
    delta_{descriptor} == requested_delta

R-group extraction uses extract_rgroups_for_smiles() from scaffold_intervention,
the same function used during candidate generation.  This guarantees that the
R-group definition is identical between generation and evaluation.

Applicability denominator
-------------------------
PRIMARY: n_total_parents (all input parents, including extraction failures).
This is the method-level applicability — what fraction of the input benchmark
can ScaffoldTuner v2 actually serve?

SECONDARY: n_extraction_success (parents for which R-group extraction
succeeded).  Reported as a diagnostic value only.

    overall_applicability_percent  = n_applicable / n_total_parents * 100
    applicability_among_extracted  = n_applicable / n_extraction_success * 100

Usage
-----
    python scripts/compute_applicability.py \
        --library transformation_library.csv \
        --parents benchmark_parents.csv \
        --smiles-col smiles \
        --output applicability_summary.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from scaffold_tuner.scaffold_intervention import (
    extract_rgroups_for_smiles,
    normalize_fragment_smiles,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Supported tasks
TASKS = [
    ("HBD", +1, "delta_HBD"),
    ("HBD", -1, "delta_HBD"),
    ("HBA", +1, "delta_HBA"),
    ("HBA", -1, "delta_HBA"),
    ("RotB",+1, "delta_RotB"),
    ("RotB",-1, "delta_RotB"),
    ("AR",  +1, "delta_AR"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Core applicability computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_applicability(
    library: pd.DataFrame,
    parent_smiles_list: list[str],
) -> pd.DataFrame:
    """
    Compute exact-match applicability for each task in TASKS.

    Parameters
    ----------
    library             : DataFrame with transformation library
    parent_smiles_list  : list of parent SMILES (all benchmark parents)

    Returns
    -------
    DataFrame with columns:
        task, target, delta,
        n_total_parents, n_extraction_success, n_extraction_failed,
        n_applicable,
        overall_applicability_percent,
        applicability_among_extracted_percent
    """
    n_total = len(parent_smiles_list)

    # Pre-build normalized from_fragment sets per task for O(1) lookup
    task_from_sets: dict[tuple[str, int], set[str]] = {}
    for target, dlt, col in TASKS:
        if col not in library.columns:
            task_from_sets[(target, dlt)] = set()
            continue
        sub = library[library[col] == dlt]
        task_from_sets[(target, dlt)] = set(
            sub["from_fragment"].apply(normalize_fragment_smiles).tolist()
        )

    # Extract R-groups for all parents once (shared across all tasks).
    # Uses extract_rgroups_for_smiles() — the same function as generation.
    log.info("Extracting R-groups for %d parent molecules ...", n_total)
    extracted: list[list[str] | None] = []
    n_failed = 0
    for i, smi in enumerate(parent_smiles_list):
        if i % 500 == 0 and i > 0:
            log.info("  processed %d / %d", i, n_total)
        rgs = extract_rgroups_for_smiles(smi)
        extracted.append(rgs)
        if rgs is None:
            n_failed += 1

    n_success = n_total - n_failed
    log.info("  R-group extraction: %d OK, %d failed", n_success, n_failed)

    # Count applicable parents per task
    results = []
    for target, dlt, col in TASKS:
        from_set = task_from_sets[(target, dlt)]
        n_applicable = 0

        for rgs in extracted:
            if rgs is None:
                # Extraction failed → not applicable (counted in denominator)
                continue
            if any(rg in from_set for rg in rgs):
                n_applicable += 1

        # Primary: denominator = n_total (all input parents)
        overall_pct = n_applicable / n_total * 100 if n_total > 0 else 0.0

        # Secondary: denominator = n_success (extraction successes only)
        among_extracted_pct = (
            n_applicable / n_success * 100 if n_success > 0 else 0.0
        )

        results.append({
            "task":                                f"{target} {dlt:+d}",
            "target":                              target,
            "delta":                               dlt,
            "n_total_parents":                     n_total,
            "n_extraction_success":                n_success,
            "n_extraction_failed":                 n_failed,
            "n_applicable":                        n_applicable,
            "overall_applicability_percent":       round(overall_pct, 1),
            "applicability_among_extracted_percent": round(among_extracted_pct, 1),
        })

    return pd.DataFrame(results)


def print_applicability(df: pd.DataFrame) -> None:
    """Print applicability results as a formatted table."""
    print("\n" + "=" * 76)
    print("  Exact-match Applicability")
    print("=" * 76)
    print(
        f"  {'Task':<10}  {'Total':>7}  {'OK':>7}  {'Failed':>7}  "
        f"{'Appl.':>7}  {'Overall%':>10}  {'Among OK%':>11}"
    )
    print("  " + "-" * 64)
    for _, row in df.iterrows():
        print(
            f"  {row['task']:<10}  "
            f"{row['n_total_parents']:>7,}  "
            f"{row['n_extraction_success']:>7,}  "
            f"{row['n_extraction_failed']:>7,}  "
            f"{row['n_applicable']:>7,}  "
            f"{row['overall_applicability_percent']:>9.1f}%  "
            f"{row['applicability_among_extracted_percent']:>10.1f}%"
        )
    print("=" * 76)
    print("  NOTE: Overall% = n_applicable / n_total_parents  (primary)")
    print("        Among OK% = n_applicable / n_extraction_success  (diagnostic)")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Input reading
# ──────────────────────────────────────────────────────────────────────────────

def _read_smiles_col(path: Path, smiles_col: str) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".tsv", ".txt"):
        try:
            df = pd.read_csv(path, sep="\t")
            if smiles_col not in df.columns:
                raise ValueError
        except Exception:
            smiles = []
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        smiles.append(parts[0])
            return smiles
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    if smiles_col not in df.columns:
        raise ValueError(
            f"Column '{smiles_col}' not found. Available: {list(df.columns)}"
        )
    return df[smiles_col].dropna().astype(str).tolist()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute exact-match applicability of a ScaffoldTuner v2 "
            "transformation library against benchmark parent molecules."
        )
    )
    parser.add_argument("--library",    "-l", required=True,
        help="Transformation library CSV.")
    parser.add_argument("--parents",    "-p", required=True,
        help="Benchmark parent molecules (CSV / TSV / .smi).")
    parser.add_argument("--smiles-col", default="smiles",
        help="Name of the SMILES column (default: smiles).")
    parser.add_argument("--output",     "-o", default="applicability_summary.csv",
        help="Output CSV (default: applicability_summary.csv).")
    args = parser.parse_args()

    lib_path = Path(args.library)
    if not lib_path.exists():
        log.error("Library not found: %s", lib_path)
        sys.exit(1)

    log.info("Loading transformation library: %s", lib_path)
    df_lib = pd.read_csv(lib_path) if lib_path.suffix == ".csv" else pd.read_parquet(lib_path)
    for col in ["delta_HBD", "delta_HBA", "delta_AR", "delta_RotB", "delta_HA", "frequency"]:
        if col in df_lib.columns:
            df_lib[col] = df_lib[col].fillna(0).astype(int)
    log.info("  %d transformations loaded", len(df_lib))

    log.info("Loading parent molecules: %s", args.parents)
    try:
        parent_smiles = _read_smiles_col(Path(args.parents), args.smiles_col)
    except Exception as e:
        log.error("Failed to read parent file: %s", e)
        sys.exit(1)
    log.info("  %d parent SMILES loaded", len(parent_smiles))

    df_result = compute_applicability(df_lib, parent_smiles)
    print_applicability(df_result)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(out_path, index=False)
    log.info("Applicability summary saved to: %s", out_path)


if __name__ == "__main__":
    main()
