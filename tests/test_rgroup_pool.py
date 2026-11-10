# -*- coding: utf-8 -*-
"""Tests for the unified R-group pool (rgroup_pool.py)."""

from rdkit import Chem

from scaffold_tuner.rgroup_pool import (
    R_GROUP_POOL, R_GROUP_SMILES, R_GROUP_BY_NAME, validate_pool
)
from scaffold_tuner.scaffold_intervention import normalize_fragment_smiles


class TestPoolContents:
    def test_25_entries(self):
        assert len(R_GROUP_POOL) == 25
        assert len(R_GROUP_SMILES) == 25

    def test_all_smiles_unique(self):
        normalized = [normalize_fragment_smiles(s) for s in R_GROUP_SMILES]
        assert len(set(normalized)) == 25

    def test_all_names_unique(self):
        names = [item["name"] for item in R_GROUP_POOL]
        assert len(set(names)) == 25

    def test_validate_pool_passes(self):
        assert validate_pool() is True


class TestPoolEntries:
    def test_all_smiles_valid_rdkit(self):
        for smi in R_GROUP_SMILES:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None, f"Invalid SMILES: {smi}"

    def test_all_entries_have_attachment_dummy(self):
        for smi in R_GROUP_SMILES:
            mol = Chem.MolFromSmiles(smi)
            has_dummy = any(a.GetAtomicNum() == 0 for a in mol.GetAtoms())
            assert has_dummy, f"No attachment dummy in: {smi}"

    def test_required_fields_present(self):
        required = {"id", "name", "smiles", "evidence_level",
                    "references", "evidence_note"}
        for item in R_GROUP_POOL:
            missing = required - set(item)
            assert not missing, f"Missing fields in {item['name']}: {missing}"

    def test_by_name_lookup(self):
        assert "methyl"  in R_GROUP_BY_NAME
        assert "phenyl"  in R_GROUP_BY_NAME
        assert "amino"   in R_GROUP_BY_NAME
        assert "hydroxy" in R_GROUP_BY_NAME
