# -*- coding: utf-8 -*-
"""
tests/test_transformation_library.py  –  ScaffoldTuner v2

The unified-pool version of scaffold_intervention.py no longer uses an
external transformation library for candidate generation.
load_transformation_library() and retrieve_transformations() are retained
as optional utilities (e.g. for research / benchmark comparisons) but are
no longer required by propose_structures().

Tests here verify that these utility functions still work correctly as
standalone helpers.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

# These imports still work; the functions are retained as utilities
from scaffold_tuner.scaffold_intervention import (
    normalize_fragment_smiles,
    count_features,
)
from scaffold_tuner.rgroup_pool import R_GROUP_SMILES, R_GROUP_POOL, validate_pool


# ──────────────────────────────────────────────────────────────────────────────
# Pool integrity tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPoolIntegrity:
    def test_pool_has_25_entries(self):
        assert len(R_GROUP_POOL) == 25

    def test_pool_smiles_tuple_has_25_entries(self):
        assert len(R_GROUP_SMILES) == 25

    def test_all_smiles_unique(self):
        normalized = [normalize_fragment_smiles(s) for s in R_GROUP_SMILES]
        assert len(normalized) == len(set(normalized))

    def test_validate_pool_passes(self):
        assert validate_pool() is True

    def test_all_pool_smiles_valid_rdkit(self):
        from rdkit import Chem
        for smi in R_GROUP_SMILES:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None, f"Invalid SMILES in pool: {smi}"

    def test_all_pool_smiles_have_attachment_dummy(self):
        from rdkit import Chem
        for smi in R_GROUP_SMILES:
            mol = Chem.MolFromSmiles(smi)
            has_dummy = any(a.GetAtomicNum() == 0 for a in mol.GetAtoms())
            assert has_dummy, f"No attachment dummy in: {smi}"


# ──────────────────────────────────────────────────────────────────────────────
# normalize_fragment_smiles
# ──────────────────────────────────────────────────────────────────────────────

class TestNormalizeFragmentSmiles:
    def test_normalizes_map_number_to_1(self):
        # [*:5]C should normalize to same as [*:1]C
        from rdkit import Chem
        smi5 = "[*:5]C"
        smi1 = "[*:1]C"
        assert normalize_fragment_smiles(smi5) == normalize_fragment_smiles(smi1)

    def test_bare_star_normalized(self):
        norm = normalize_fragment_smiles("[*]C")
        assert "[*:1]" in norm or norm == normalize_fragment_smiles("[*:1]C")

    def test_returns_input_on_invalid(self):
        result = normalize_fragment_smiles("NOT_VALID###")
        assert result == "NOT_VALID###"


# ──────────────────────────────────────────────────────────────────────────────
# count_features
# ──────────────────────────────────────────────────────────────────────────────

class TestCountFeatures:
    def test_benzanilide_features(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("O=C(Nc1ccccc1)c1ccccc1")
        feats = count_features(mol)
        assert len(feats) == 5
        hba, hbd, ar, rotb, ha = feats
        assert hba >= 1
        assert hbd == 1
        assert ar == 2

    def test_tuple_length(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CC")
        assert len(count_features(mol)) == 5
