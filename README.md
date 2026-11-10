# ScaffoldTuner

A Python library for proposing structurally modified drug-like molecules
using a **descriptor-indexed R-group transformation library**.

Given a parent molecule and a desired descriptor change (e.g. `HBA +1`),
ScaffoldTuner retrieves observed R-group transformations from a pre-built
library, applies only those whose `from_fragment` exactly matches a parent
R-group, and validates the exact descriptor change on the full generated
molecule with RDKit.

---

## Core concept

ScaffoldTuner v2 is built around one idea:

> **desired descriptor change → transformation retrieval → generation → exact validation**

This differs from CReM, which starts from chemical context:

| | CReM | ScaffoldTuner v2 |
|---|---|---|
| **Retrieval key** | Local chemical context (ECFP-like radius) | Desired descriptor delta |
| **Starting point** | Structure → find compatible replacements | Descriptor goal → find matching transformations |
| **R-group match** | Context radius match | Exact canonical SMILES (default) |
| **Scaffold** | Not constrained | Parent scaffold retained as substructure |
| **Validation** | Context match | Whole-molecule RDKit recomputation |

### Pipeline

```
desired descriptor change  (e.g. HBA +1)
        ↓
Step 1: descriptor-indexed retrieval
        library[delta_HBA == +1]   ← fragment-level pre-filter
        ↓
Step 2: exact R-group match  (match_mode="exact", default)
        parent R-group == from_fragment  (canonical SMILES)
        ↓
Step 3: candidate generation
        ↓
Step 4: RDKit sanitization
        ↓
Step 5: parent scaffold retained as substructure
        generated_mol.HasSubstructMatch(parent_scaffold_mol)
        ↓
Step 6: whole-molecule descriptor recomputation
        actual_delta = generated_HBA - parent_HBA
        ↓
Step 7: exact requested delta validation
        actual_delta == requested_delta
        ↓
Step 8: optional strict filtering
        ↓
Step 9: ranking
        ↓
Step 10: optional MaxMin diversity selection
        ↓
Top-N candidates + full provenance
```

---

## Scaffold preservation

"Scaffold preserved" means:

```python
generated_mol.HasSubstructMatch(parent_scaffold_mol)  # True
```

The parent Murcko scaffold is **retained as a substructure** in the
generated molecule.  This does NOT mean the generated molecule's own
Murcko scaffold is identical to the parent's.  For example, adding an
aromatic ring to a side-chain (AR +1) may extend the generated molecule's
Murcko scaffold while still containing the parent scaffold as a substructure.

---

## Note on delta values

The library stores **fragment-level** delta values (`delta_HBA`, etc.).
These are used as a retrieval pre-filter (Step 1) only.

The final acceptance criterion is always the **whole-molecule actual_delta**
computed by RDKit after generation (Step 7).  Fragment-level and
whole-molecule deltas can differ, particularly for `RotB`, because rotatable
bond count depends on the attachment bond and ring context of the full molecule.

Each result record clearly distinguishes:

- `library_delta`  — fragment-level delta from the library (retrieval only)
- `actual_delta`   — whole-molecule delta confirmed by RDKit (final criterion)

---

## Exact matching and similarity fallback

**Exact R-group matching is used by default (`match_mode="exact"`).**
Only transformations whose `from_fragment` is canonically identical to the
parent R-group are applied.  If no exact match exists for a given site,
zero candidates are returned for that site — no silent fallback.

Similarity-based fallback (`match_mode="similarity"`) is available as an
**optional exploratory mode only**.  It must NOT be used in benchmark
comparisons, because it weakens the traceability to observed transformations.

---

## Supported modifications

### Supported

| target | delta | Legacy mode | Description |
|---|---|---|---|
| `HBD` | +1 | `add_hbd` | Increase hydrogen bond donors |
| `HBD` | -1 | `remove_hbd` | Decrease hydrogen bond donors |
| `HBA` | +1 | `add_hba` | Increase hydrogen bond acceptors |
| `HBA` | -1 | `remove_hba` | Decrease hydrogen bond acceptors |
| `AR`  | +1 | `add_ar` | Increase number of aromatic rings |
| `RotB`| +1 | `add_rb` | Increase number of rotatable bonds |
| `RotB`| -1 | `remove_rb` | Decrease number of rotatable bonds |
| `HA`  | +1 | `add_ha` | Increase number of heteroatoms |
| `HA`  | -1 | `remove_ha` | Decrease number of heteroatoms |

### Not supported

| target | delta | Reason |
|---|---|---|
| `AR` | -1 | Removing an aromatic ring from an R-group requires modifying the scaffold. Raises `ValueError`. |

---

## Requirements

- Python 3.8+
- [RDKit](https://www.rdkit.org/)
- pandas

```bash
conda install -c conda-forge rdkit pandas
```

---

## Installation

```bash
# From GitHub
pip install git+https://github.com/tesaki2019/scaffold-tuner.git

# Local development
git clone https://github.com/tesaki2019/scaffold-tuner.git
cd scaffold-tuner
pip install -e .
```

### Google Colab

```python
!pip install rdkit pandas
!pip install git+https://github.com/tesaki2019/scaffold-tuner.git
```

> **Note:** In Google Colab, packages reset when the session ends.
> Run the installation cell at the beginning of each new session.

---

## Building the transformation library

The transformation library is a CSV file of observed R-group transformations
with pre-computed fragment-level descriptor deltas, derived from ChEMBL or
any other compound dataset.

```bash
python scripts/build_transformation_library.py \
    --input chembl_compounds.csv \
    --output transformation_library.csv \
    --smiles-col smiles \
    --id-col chembl_id \
    --min-frequency 2 \
    --max-heavy-atoms 10 \
    --source ChEMBL
```

After building, the script prints the actual statistics for your library:

```
============================================================
  Transformation Library Summary
============================================================
  Source compounds         :  (actual value)
  R-group extraction OK    :  (actual value)
  R-group extraction failed:  (actual value)
  Unique Murcko scaffolds  :  (actual value)   ← from source data
  Unique R-groups          :  (actual value)
  Unique transformations   :  (actual value)
  Total observations       :  (actual value)

  Descriptor-indexed transformations
  ----------------------------------------
  Task          Unique trans    Total obs
  HBD +1             (actual)     (actual)
  ...

  Frequency distribution
  ----------------------------------------
  freq == 1   :     (actual)
  freq >= 2   :     (actual)
  ...
============================================================
```

The "Unique Murcko scaffolds" count is computed from the source compounds
directly (before transformation aggregation).

### Library format

| Column | Type | Description |
|---|---|---|
| `from_fragment` | str | Source R-group SMILES, attachment as `[*]` |
| `to_fragment` | str | Target R-group SMILES, attachment as `[*]` |
| `delta_HBD` | int | Fragment-level ΔNumHDonors |
| `delta_HBA` | int | Fragment-level ΔNumHAcceptors |
| `delta_AR` | int | Fragment-level ΔNumAromaticRings |
| `delta_RotB` | int | Fragment-level ΔNumRotatableBonds |
| `delta_HA` | int | Fragment-level ΔNumHeteroatoms (default 0) |
| `frequency` | int | Observed count (default 1) |
| `source` | str | Source label, e.g. `"ChEMBL"` (default `"unknown"`) |

---

## Usage

### New API (v2)

```python
from scaffold_tuner.scaffold_intervention import (
    load_transformation_library,
    propose_structures,
    print_proposals,
)

# Load once; cached in memory
lib = load_transformation_library("transformation_library.csv")

parent_smiles = "c1ccc(NC(=O)c2ccccc2)cc1"

results = propose_structures(
    parent_smiles=parent_smiles,
    target="HBA",   # HBD | HBA | AR | RotB | HA
    delta=1,        # +1 or -1  (AR: +1 only)
    transformation_library=lib,
    max_candidates=10,
    match_mode="exact",   # always use "exact" for benchmarks
)

print_proposals(results)
```

### Legacy API (backward-compatible)

```python
results = propose_structures(
    parent_smiles=parent_smiles,
    mode="add_hba",             # converted to target="HBA", delta=+1
    transformation_library=lib,
)
```

### With strict mode

```python
# strict=True: target descriptor changes by delta AND no other descriptor changes
results = propose_structures(
    parent_smiles=parent_smiles,
    target="HBA", delta=1,
    transformation_library=lib,
    strict=True,
)
```

### With diversity selection

```python
results = propose_structures(
    parent_smiles=parent_smiles,
    target="HBA", delta=1,
    transformation_library=lib,
    max_candidates=10,
    diversity=True,    # MaxMin selection after ranking
)
```

### Exploratory mode (NOT for benchmarks)

```python
# similarity fallback: only when match_mode="similarity" is explicit
results = propose_structures(
    parent_smiles=parent_smiles,
    target="HBA", delta=1,
    transformation_library=lib,
    match_mode="similarity",
    similarity_threshold=0.5,
)
```

### Example output

```
=== Candidate 1 ===
target: HBA  requested: +1  library: +1  actual: +1  site: R1  collateral: 1
from → to : [*:1]C → [*:1]OC
freq: 253  source: ChEMBL  match: exact (1.000)
scaffold_preserved: True  parent scaffold: c1ccc(cc1)
parent   : O=C(Nc1ccccc1)c1ccc(C)cc1
generated: COc1ccc(C(=O)Nc2ccccc2)cc1
features ['NumHAcceptors', 'NumHDonors', 'NumAromaticRings', 'NumRotatableBonds', 'NumHeteroatoms']
  parent   : (1, 1, 2, 2, 2)
  generated: (2, 1, 2, 3, 3)
```

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `parent_smiles` | str | required | SMILES of the parent molecule |
| `target` | str | — | Descriptor: `HBD`, `HBA`, `AR`, `RotB`, `HA` |
| `delta` | int | — | Desired change: `+1` or `-1` |
| `transformation_library` | DataFrame | required | From `load_transformation_library()` |
| `mode` | str | `None` | Legacy mode string (overrides target/delta) |
| `max_cuts` | int | `None` | Max attachment points to label |
| `max_candidates` | int | `10` | Maximum candidates to return |
| `shuffle_sites` | bool | `False` | Randomize attachment site order |
| `random_seed` | int | `None` | Seed for reproducibility |
| `strict` | bool | `False` | Only return candidates with no non-target descriptor change |
| `match_mode` | str | `"exact"` | `"exact"` (benchmark) or `"similarity"` (exploratory) |
| `similarity_threshold` | float | `0.4` | Tanimoto threshold for similarity fallback |
| `diversity` | bool | `False` | MaxMin diversity selection on final candidates |

---

## Result fields

| Field | Description |
|---|---|
| `parent_smiles` | Canonical SMILES of the parent |
| `generated_smiles` | Canonical SMILES of the candidate |
| `target_descriptor` | e.g. `"HBA"` |
| `requested_delta` | Delta requested by user |
| `library_delta` | Fragment-level delta from library (retrieval index only) |
| `actual_delta` | Whole-molecule delta confirmed by RDKit (== `requested_delta`) |
| `site` | Attachment site, e.g. `"R2"` |
| `from_fragment` | Source R-group from the library |
| `to_fragment` | Replacement R-group from the library |
| `transformation_frequency` | Observed count in source dataset |
| `transformation_source` | Source label, e.g. `"ChEMBL"` |
| `match_type` | `"exact"` or `"similarity"` |
| `match_similarity` | Tanimoto similarity (1.0 for exact) |
| `parent_scaffold_smiles` | Murcko scaffold of the parent |
| `generated_murcko_scaffold` | Murcko scaffold of the generated molecule |
| `scaffold_preserved` | True if parent scaffold is retained as substructure |
| `collateral_change` | Sum \|Δ\| for all non-target descriptors |
| `parent_features` | `(HBA, HBD, AR, RotB, HA)` of parent |
| `generated_features` | `(HBA, HBD, AR, RotB, HA)` of candidate |

---

## Applicability analysis

Use `scripts/compute_applicability.py` to measure library coverage:

```bash
python scripts/compute_applicability.py \
    --library transformation_library.csv \
    --parents benchmark_parents.csv \
    --smiles-col smiles \
    --output applicability_summary.csv
```

For each task, this reports the fraction of benchmark parent molecules for
which at least one exact-match transformation exists.

---

## Tests

```bash
pip install pytest
pytest tests/ -v
```

All 35 tests pass:

| Test class | What is verified |
|---|---|
| `TestExactMatchingDefault` | `match_type="exact"` and `match_similarity=1.0` by default |
| `TestNoSilentSimilarityFallback` | `match_mode="exact"` returns 0 candidates (no fallback) when no exact match |
| `TestExactDeltaHBA` | All candidates: `generated_HBA - parent_HBA == 1` |
| `TestExactDeltaHBD` | All candidates: `generated_HBD - parent_HBD == 1` |
| `TestRotBWholeMolecule` | RotB validated on full molecule; `actual_delta == requested_delta` |
| `TestScaffoldRetention` | All candidates: `HasSubstructMatch(parent_scaffold)` |
| `TestStrictMode` | `strict=True`: all non-target descriptors unchanged; `collateral_change == 0` |
| `TestNoDuplicates` | No duplicate `generated_smiles` in a single call |
| `TestRemoveArUnsupported` | `AR -1` raises `ValueError` |
| `TestProvenanceFields` | All required provenance keys present |
| `TestReproducibility` | Same seed → same results |
| `TestLoadValidCsv` | Library loading returns correct DataFrame |
| `TestLoadMissingFile` | `FileNotFoundError` on missing path |
| `TestLoadMissingRequiredColumns` | `ValueError` on missing columns |
| `TestOptionalColumnDefaults` | Optional columns filled with defaults |
| `TestRetrieveExactDelta` | Returns only rows matching delta |
| `TestRetrieveEmptyWhenNoMatch` | Returns empty DataFrame when no match |
| `TestLoadCachesResult` | Repeated calls return cached DataFrame |

---

## Project structure

```
scaffold-tuner/
├── scaffold_tuner/
│   ├── __init__.py
│   └── scaffold_intervention.py   # library: propose_structures() and helpers
│
├── scripts/
│   ├── build_transformation_library.py   # build library from SMILES dataset
│   └── compute_applicability.py          # exact-match coverage analysis
│
├── tests/
│   ├── conftest.py                        # shared fixtures
│   ├── test_scaffold_intervention.py      # Tests 1–11
│   └── test_transformation_library.py     # Tests A–G
│
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
