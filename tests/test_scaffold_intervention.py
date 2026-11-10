# -*- coding: utf-8 -*-
"""Tests for propose_structures() and the unified R-group pool."""

import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold

from scaffold_tuner.rgroup_pool import R_GROUP_SMILES, validate_pool
from scaffold_tuner.scaffold_intervention import (
    propose_structures,
    normalize_fragment_smiles,
    extract_rgroups_for_smiles,
)


# ── Pool integrity ────────────────────────────────────────────────────────────

class TestPool:
    def test_25_unique_entries(self):
        assert len(R_GROUP_SMILES) == 25
        normalized = [normalize_fragment_smiles(s) for s in R_GROUP_SMILES]
        assert len(set(normalized)) == 25, "Pool contains duplicate R-groups"

    def test_validate_pool(self):
        assert validate_pool() is True

    def test_no_descriptor_specific_subpool(self):
        import scaffold_tuner.scaffold_intervention as si
        for name in ("_FRAGMENT_TEMPLATES", "_add_hbd_pool", "_add_hba_pool",
                     "_add_ar_pool", "_add_rb_pool", "_remove_hbd_pool",
                     "_remove_hba_pool", "_remove_rb_pool"):
            assert not hasattr(si, name), f"{name} must not exist"


# ── HBD ──────────────────────────────────────────────────────────────────────

class TestHBD:
    def _parent_hbd(self, smi):
        return Lipinski.NumHDonors(Chem.MolFromSmiles(smi))

    def test_plus1(self, parent_methyl):
        results = propose_structures(parent_methyl, target="HBD", delta=1,
                                     max_candidates=25)
        assert len(results) > 0
        p = self._parent_hbd(parent_methyl)
        for r in results:
            assert r["actual_delta"] == 1
            assert Lipinski.NumHDonors(Chem.MolFromSmiles(r["generated_smiles"])) - p == 1

    def test_minus1(self, parent_methyl):
        # Use a parent with an HBD-containing R-group
        parent = "O=C(Nc1ccccc1)c1ccc(N)cc1"  # para-amino
        results = propose_structures(parent, target="HBD", delta=-1,
                                     max_candidates=25)
        assert len(results) > 0
        p = self._parent_hbd(parent)
        for r in results:
            assert r["actual_delta"] == -1
            assert Lipinski.NumHDonors(Chem.MolFromSmiles(r["generated_smiles"])) - p == -1


# ── HBA ──────────────────────────────────────────────────────────────────────

class TestHBA:
    def _parent_hba(self, smi):
        return Lipinski.NumHAcceptors(Chem.MolFromSmiles(smi))

    def test_plus1(self, parent_methyl):
        results = propose_structures(parent_methyl, target="HBA", delta=1,
                                     max_candidates=25)
        assert len(results) > 0
        p = self._parent_hba(parent_methyl)
        for r in results:
            assert r["actual_delta"] == 1
            assert Lipinski.NumHAcceptors(Chem.MolFromSmiles(r["generated_smiles"])) - p == 1

    def test_minus1(self, parent_methyl):
        parent = "O=C(Nc1ccccc1)c1ccc(OC)cc1"  # para-methoxy (has HBA)
        results = propose_structures(parent, target="HBA", delta=-1,
                                     max_candidates=25)
        assert len(results) > 0
        p = self._parent_hba(parent)
        for r in results:
            assert r["actual_delta"] == -1
            assert Lipinski.NumHAcceptors(Chem.MolFromSmiles(r["generated_smiles"])) - p == -1


# ── RotB ─────────────────────────────────────────────────────────────────────

class TestRotB:
    def _parent_rotb(self, smi):
        return Descriptors.NumRotatableBonds(Chem.MolFromSmiles(smi))

    def test_plus1(self, parent_methyl):
        results = propose_structures(parent_methyl, target="RotB", delta=1,
                                     max_candidates=25)
        assert len(results) > 0
        p = self._parent_rotb(parent_methyl)
        for r in results:
            assert r["actual_delta"] == 1
            assert Descriptors.NumRotatableBonds(Chem.MolFromSmiles(r["generated_smiles"])) - p == 1

    def test_minus1(self, parent_ethyl):
        results = propose_structures(parent_ethyl, target="RotB", delta=-1,
                                     max_candidates=25)
        assert len(results) > 0
        p = self._parent_rotb(parent_ethyl)
        for r in results:
            assert r["actual_delta"] == -1
            assert Descriptors.NumRotatableBonds(Chem.MolFromSmiles(r["generated_smiles"])) - p == -1


# ── AR ───────────────────────────────────────────────────────────────────────

class TestAR:
    def test_plus1(self, parent_methyl):
        results = propose_structures(parent_methyl, target="AR", delta=1,
                                     max_candidates=25)
        assert len(results) > 0
        p = Descriptors.NumAromaticRings(Chem.MolFromSmiles(parent_methyl))
        for r in results:
            assert r["actual_delta"] == 1
            assert Descriptors.NumAromaticRings(Chem.MolFromSmiles(r["generated_smiles"])) - p == 1


# ── strict=True ──────────────────────────────────────────────────────────────

class TestStrictMode:
    def test_collateral_zero(self, parent_ethylbenzaldehyde):
        results = propose_structures(parent_ethylbenzaldehyde,
                                     target="RotB", delta=-1,
                                     strict=True, max_candidates=25)
        assert len(results) > 0
        parent_mol = Chem.MolFromSmiles(parent_ethylbenzaldehyde)
        p_hba = Lipinski.NumHAcceptors(parent_mol)
        p_hbd = Lipinski.NumHDonors(parent_mol)
        p_ar  = Descriptors.NumAromaticRings(parent_mol)
        for r in results:
            gen = Chem.MolFromSmiles(r["generated_smiles"])
            assert Lipinski.NumHAcceptors(gen) == p_hba
            assert Lipinski.NumHDonors(gen)    == p_hbd
            assert Descriptors.NumAromaticRings(gen) == p_ar
            assert r["collateral_change"] == 0

    def test_strict_subset_of_nonstrict(self, parent_methyl):
        strict    = propose_structures(parent_methyl, target="HBA", delta=1,
                                       strict=True,  max_candidates=50)
        nonstrict = propose_structures(parent_methyl, target="HBA", delta=1,
                                       strict=False, max_candidates=50)
        assert len(strict) <= len(nonstrict)
        for r in strict:
            assert r["collateral_change"] == 0


# ── Scaffold retention ────────────────────────────────────────────────────────

class TestScaffoldRetention:
    def test_scaffold_preserved_flag(self, parent_methyl):
        results = propose_structures(parent_methyl, target="HBA", delta=1,
                                     max_candidates=25)
        assert len(results) > 0
        for r in results:
            assert r["scaffold_preserved"] is True

    def test_has_substructure_match(self, parent_methyl):
        parent_mol   = Chem.MolFromSmiles(parent_methyl)
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(parent_mol)
        results = propose_structures(parent_methyl, target="HBA", delta=1,
                                     max_candidates=25)
        for r in results:
            gen_mol = Chem.MolFromSmiles(r["generated_smiles"])
            assert gen_mol.HasSubstructMatch(scaffold_mol)


# ── Duplicate removal ─────────────────────────────────────────────────────────

class TestNoDuplicates:
    def test_no_duplicate_smiles(self, parent_methyl):
        results = propose_structures(parent_methyl, target="HBA", delta=1,
                                     max_candidates=50)
        smiles = [r["generated_smiles"] for r in results]
        assert len(smiles) == len(set(smiles))


# ── remove_ar → NotImplementedError ──────────────────────────────────────────

class TestRemoveArUnsupported:
    def test_target_ar_minus1_raises(self, parent_methyl):
        with pytest.raises(NotImplementedError):
            propose_structures(parent_methyl, target="AR", delta=-1)

    def test_mode_remove_ar_raises(self, parent_methyl):
        with pytest.raises(NotImplementedError):
            propose_structures(parent_methyl, mode="remove_ar")


# ── Reproducibility ───────────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_same_results(self, parent_methyl):
        kw = dict(target="HBA", delta=1, shuffle_sites=True, random_seed=42)
        r1 = [r["generated_smiles"] for r in propose_structures(parent_methyl, **kw)]
        r2 = [r["generated_smiles"] for r in propose_structures(parent_methyl, **kw)]
        assert r1 == r2


# ── Result fields ─────────────────────────────────────────────────────────────

class TestResultFields:
    REQUIRED = {
        "parent_smiles", "generated_smiles",
        "target_descriptor", "requested_delta", "actual_delta",
        "site", "to_fragment",
        "parent_scaffold_smiles", "generated_murcko_scaffold",
        "scaffold_preserved", "collateral_change",
        "parent_features", "generated_features",
    }

    def test_all_keys_present(self, parent_methyl):
        results = propose_structures(parent_methyl, target="HBA", delta=1,
                                     max_candidates=5)
        assert len(results) > 0
        for r in results:
            assert not (self.REQUIRED - set(r)), \
                f"Missing keys: {self.REQUIRED - set(r)}"

    def test_no_internal_pool_index(self, parent_methyl):
        results = propose_structures(parent_methyl, target="HBA", delta=1,
                                     max_candidates=5)
        for r in results:
            assert "_pool_index" not in r

    def test_to_fragment_is_valid_smiles(self, parent_methyl):
        """
        to_fragment is either a direct pool entry or a sub-substituted variant
        of the parent R-group (with a pool entry grafted inside it).
        Either way it must be a valid, non-empty SMILES string.
        """
        from rdkit import Chem
        results = propose_structures(parent_methyl, target="HBA", delta=1,
                                     max_candidates=25)
        for r in results:
            assert isinstance(r["to_fragment"], str) and len(r["to_fragment"]) > 0
            mol = Chem.MolFromSmiles(r["to_fragment"])
            assert mol is not None, f"Invalid to_fragment SMILES: {r['to_fragment']}"


# ── extract_rgroups_for_smiles ────────────────────────────────────────────────

class TestExtractRgroups:
    def test_returns_list(self, parent_methyl):
        rgs = extract_rgroups_for_smiles(parent_methyl)
        assert rgs is not None and len(rgs) > 0

    def test_returns_none_for_invalid(self):
        assert extract_rgroups_for_smiles("NOT_VALID###") is None
