# ScaffoldTuner

A lightweight Python tool for proposing R-group modifications to drug-like
molecules.  Given a parent molecule and a desired descriptor change,
ScaffoldTuner generates candidate structures by substituting R-groups at
scaffold attachment points and retaining only those where the requested
descriptor change is confirmed by RDKit on the full molecule.

---

## Features

- Preserves the parent Murcko scaffold in every returned candidate
- Evaluates 25 literature-informed R-groups (unified pool) at each
  attachment site — no descriptor-specific sub-pool
- Validates descriptor changes on the **whole molecule** with RDKit
- `strict=True` mode: ensures only the target descriptor changes
- Removes duplicate candidates automatically
- Optional MaxMin diversity selection

---

## Supported descriptor modifications

| target | delta | Legacy mode string |
|--------|-------|--------------------|
| `HBD`  | +1    | `add_hbd`          |
| `HBD`  | -1    | `remove_hbd`       |
| `HBA`  | +1    | `add_hba`          |
| `HBA`  | -1    | `remove_hba`       |
| `AR`   | +1    | `add_ar`           |
| `RotB` | +1    | `add_rb`           |
| `RotB` | -1    | `remove_rb`        |
| `HA`   | +1    | `add_ha`           |
| `HA`   | -1    | `remove_ha`        |

`AR -1` (`remove_ar`) is not supported in the current implementation.

---

## Installation

```bash
# From GitHub
pip install git+https://github.com/tesak1101/scaffold-tuner.git

# Local development
git clone https://github.com/tesak1101/scaffold-tuner.git
cd scaffold-tuner
pip install -e .
```

**Requirements:** Python 3.8+, RDKit, pandas

```bash
conda install -c conda-forge rdkit
```

### Google Colab

```python
!pip install rdkit
!pip install git+https://github.com/tesak1101/scaffold-tuner.git
```

> In Google Colab, packages reset when the session ends.
> Run the installation cell at the start of each new session.

---

## Quick start

```python
from scaffold_tuner.scaffold_intervention import propose_structures, print_proposals

parent_smiles = "c1ccc(NC(=O)c2ccccc2)cc1"  # benzanilide

results = propose_structures(
    parent_smiles,
    target="HBA",   # descriptor to change
    delta=1,        # +1 or -1
    max_candidates=5,
)

print_proposals(results)
```

**Example output:**

```
=== Candidate 1 ===
target: HBA  requested: +1  actual: +1  site: R1  collateral: 1
to_fragment      : [*:1]C#N
scaffold_preserved: True
parent   : O=C(Nc1ccccc1)c1ccccc1
generated: N#Cc1ccc(NC(=O)c2ccccc2)cc1
features ['NumHAcceptors', 'NumHDonors', 'NumAromaticRings', 'NumRotatableBonds', 'NumHeteroatoms']
  parent   : (1, 1, 2, 2, 1)
  generated: (2, 1, 2, 2, 2)
```

### Legacy mode strings

```python
# Equivalent to target="HBD", delta=1
results = propose_structures(parent_smiles, mode="add_hbd", max_candidates=5)
```

---

## How it works

```
Parent molecule
      ↓
Murcko scaffold extraction + R-group decomposition
      ↓
Substitute each of the 25 unified R-groups at every attachment site
      ↓
Recalculate whole-molecule descriptors with RDKit
      ↓
Retain candidates where actual_delta == requested_delta
      ↓
Scaffold retention check: HasSubstructMatch(parent_scaffold)
      ↓
Optional strict filtering (no other descriptor may change)
      ↓
Rank by collateral descriptor change, then pool order
      ↓
Return top-N candidates
```

Descriptor validation is always performed on the **complete generated
molecule**.  No pre-filtering based on assumed R-group properties is used.

---

## Unified R-group pool

All 25 candidate R-groups are stored in `scaffold_tuner/rgroup_pool.py`.

| # | Name | SMILES |
|---|------|--------|
| 1  | amino          | `[*:1]N`           |
| 2  | methylamino    | `[*:1]NC`          |
| 3  | hydroxy        | `[*:1]O`           |
| 4  | 3-pyrrolyl     | `[*:1]c1ccc[nH]1`  |
| 5  | indolyl        | `[*:1]c1ccc2[nH]ccc2c1` |
| 6  | methyl         | `[*:1]C`           |
| 7  | ethyl          | `[*:1]CC`          |
| 8  | fluoro         | `[*:1]F`           |
| 9  | chloro         | `[*:1]Cl`          |
| 10 | isopropyl      | `[*:1]C(C)C`       |
| 11 | methoxy        | `[*:1]OC`          |
| 12 | cyano          | `[*:1]C#N`         |
| 13 | acetyl         | `[*:1]C(=O)C`      |
| 14 | dimethylamino  | `[*:1]N(C)C`       |
| 15 | methylsulfonyl | `[*:1]S(=O)(=O)C`  |
| 16 | phenyl         | `[*:1]c1ccccc1`    |
| 17 | pyridyl        | `[*:1]c1ccncc1`    |
| 18 | 2-furyl        | `[*:1]c1ccco1`     |
| 19 | 2-thienyl      | `[*:1]c1cccs1`     |
| 20 | n-propyl       | `[*:1]CCC`         |
| 21 | n-butyl        | `[*:1]CCCC`        |
| 22 | methoxymethyl  | `[*:1]COC`         |
| 23 | acetonyl       | `[*:1]CC(=O)C`     |
| 24 | cyclopropyl    | `[*:1]C1CC1`       |
| 25 | cyclohexyl     | `[*:1]C1CCCCC1`    |

The pool is based on a manual curation of common medicinal chemistry
R-groups.  Literature references are stored as metadata in `rgroup_pool.py`
and are not used for candidate selection or ranking.

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parent_smiles` | str | required | SMILES of the parent molecule |
| `target` | str | — | Descriptor: `HBD`, `HBA`, `AR`, `RotB`, `HA` |
| `delta` | int | — | Requested change: `+1` or `-1` |
| `mode` | str | `None` | Legacy mode string (overrides target/delta) |
| `max_cuts` | int | `None` | Max attachment sites to edit (None = all) |
| `max_candidates` | int | `10` | Maximum candidates to return |
| `shuffle_sites` | bool | `False` | Randomize site evaluation order |
| `random_seed` | int | `None` | Seed for reproducibility |
| `strict` | bool | `False` | Only return candidates where no non-target descriptor changes |
| `diversity` | bool | `False` | Apply MaxMin diversity selection |

---

## Result fields

Each dict in the returned list contains:

| Field | Description |
|-------|-------------|
| `parent_smiles` | Canonical SMILES of the parent |
| `generated_smiles` | Canonical SMILES of the candidate |
| `target_descriptor` | e.g. `"HBA"` |
| `requested_delta` | The delta you requested |
| `actual_delta` | Confirmed by RDKit on the full molecule |
| `site` | Attachment site label, e.g. `"R1"` |
| `to_fragment` | Replacement R-group SMILES |
| `parent_scaffold_smiles` | Murcko scaffold of the parent |
| `generated_murcko_scaffold` | Murcko scaffold of the generated molecule |
| `scaffold_preserved` | `True` if the parent scaffold is a substructure of the candidate |
| `collateral_change` | Sum of \|Δ\| for all non-target descriptors |
| `parent_features` | `(HBA, HBD, AR, RotB, HA)` of parent |
| `generated_features` | `(HBA, HBD, AR, RotB, HA)` of candidate |

---

## Current limitations

- **AR -1** (`remove_ar`) is not supported.  Removing an aromatic ring
  from a side-chain R-group typically requires modifying the scaffold,
  which contradicts the scaffold-preservation design.
- The pool contains 25 R-groups.  Candidates are limited to substitutions
  at Murcko scaffold attachment points.
- Molecules without scaffold attachment points (e.g. simple alkanes) return
  an empty list.

---

## Project structure

```
scaffold-tuner/
├── scaffold_tuner/
│   ├── __init__.py
│   ├── scaffold_intervention.py   # propose_structures() and helpers
│   └── rgroup_pool.py             # unified R-group pool (25 entries)
├── tests/
│   ├── conftest.py
│   ├── test_scaffold_intervention.py
│   └── test_rgroup_pool.py
├── README.md
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

---

## Running tests

```bash
pip install pytest
pytest
```

---

## Citation

If you use ScaffoldTuner in your research, please cite:

> [manuscript in preparation]

---

## License

MIT License. See [LICENSE](LICENSE) for details.
