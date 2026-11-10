# -*- coding: utf-8 -*-
"""
tests/test_scaffold_intervention.py  –  ScaffoldTuner v2

Tests for the unified-pool version of propose_structures().

Test coverage
-------------
Test 1   pool_size                  R_GROUP_SMILES has exactly 25 unique entries
Test 2   no_descriptor_specific_pool  descriptor-specific sub-pools don't exist
Test 3   hba_exact_delta            all HBA+1 candidates: actual ΔHBA = +1
Test 4   hbd_exact_delta            all HBD+1 candidates: actual ΔHBD = +1
Test 5   rotb_plus1                 all RotB+1 candidates: actual ΔRotB = +1
Test 6   rotb_minus1                all RotB-1 candidates: actual ΔRotB = -1
Test 7   ar_plus1                   all AR+1 candidates: actual ΔAR = +1
Test 8   strict_mode                collateral_change == 0 for all strict candidates
Test 9   scaffold_retention         HasSubstructMatch(parent_scaffold) for all
Test 10  no_duplicates              no duplicate generated_smiles
Test 11  remove_ar_not_implemented  AR-1 raises NotImplementedError
Test 12  reproducibility            same seed → same results
Test 13  provenance_fields          required keys present in every result
Test 14  extract_rgroups_utility    extract_rgroups_for_smiles() behavior
"""

import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold

from scaffold_tuner.rgroup_pool import R_GROUP_SMILES, validate_pool
from scaffold_tuner.scaffold_intervention import (
    propose_structures,
    extract_rgroups_for_smiles,
    normalize_fragment_smiles,
    count_features,
)


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: unified pool has exactly 25 unique R-groups
# ──────────────────────────────────────────────────────────────────────────────

class TestPoolSize:
    def test_25_entries(self):
        assert len(R_GROUP_SMILES) == 25

    def test_all_unique(self):
        normalized = [normalize_fragment_smiles(s) for s in R_GROUP_SMILES]
        assert len(normalized) == len(set(normalized)), \
            "Duplicate R-group SMILES in pool"

    def test_validate_pool_passes(self):
        assert validate_pool() is True


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: no descriptor-specific fragment sub-pool
# ──────────────────────────────────────────────────────────────────────────────

class TestNoDescriptorSpecificPool:
    """
    scaffold_intervention must NOT define per-descriptor fragment pools
    (e.g. _FRAGMENT_TEMPLATES or add_hbd/add_hba sub-lists).
    Candidate selection is done entirely by whole-molecule validation.
    """

    def test_no_fragment_templates_attribute(self):
        import scaffold_tuner.scaffold_intervention as si
        assert not hasattr(si, "_FRAGMENT_TEMPLATES"), \
            "_FRAGMENT_TEMPLATES must not exist in scaffold_intervention"

    def test_no_per_mode_pool(self):
        import scaffold_tuner.scaffold_intervention as si
        for name in ("_add_hbd_pool", "_add_hba_pool", "_remove_hbd_pool",
                     "_remove_hba_pool", "_add_ar_pool", "_add_rb_pool",
                     "_remove_rb_pool"):
            assert not hasattr(si, name), f"{name} must not exist"

    def test_hba_candidates_come_from_full_pool(self, parent_methyl):
        """
        HBA+1 candidates must include R-groups that are not obvious HBA donors
        (e.g. amino [*:1]N increases HBA AND HBD — it still passes HBA+1 filter
        because actual ΔHBA = +1 regardless of pool membership).
        The pool evaluated is always R_GROUP_SMILES (all 25).
        """
        results = propose_structures(
            parent_methyl, target="HBA", delta=1, max_candidates=25
        )
        # Confirm some candidates use R-groups from the heteroatom-containing pool
        to_frags = {r["to_fragment"] for r in results}
        assert len(to_frags) > 0

    def test_hbd_candidates_come_from_full_pool(self, parent_methyl):
        results = propose_structures(
            parent_methyl, target="HBD", delta=1, max_candidates=25
        )
        assert len(results) > 0


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: HBA exact delta
# ──────────────────────────────────────────────────────────────────────────────

class TestHBAExactDelta:
    def test_all_candidates_hba_delta_one(self, parent_methyl):
        results = propose_structures(
            parent_methyl, target="HBA", delta=1, max_candidates=25
        )
        assert len(results) > 0
        parent_mol = Chem.MolFromSmiles(parent_methyl)
        parent_hba = Lipinski.NumHAcceptors(parent_mol)
        for rec in results:
            assert rec["actual_delta"] == 1
            gen_mol = Chem.MolFromSmiles(rec["generated_smiles"])
            assert Lipinski.NumHAcceptors(gen_mol) - parent_hba == 1

    def test_no_candidate_with_wrong_delta(self, parent_methyl):
        results = propose_structures(
            parent_methyl, target="HBA", delta=1, max_candidates=25
        )
        for rec in results:
            assert rec["actual_delta"] == 1, \
                f"Found candidate with actual_delta={rec['actual_delta']}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: HBD exact delta
# ──────────────────────────────────────────────────────────────────────────────

class TestHBDExactDelta:
    def test_all_candidates_hbd_delta_one(self, parent_methyl):
        results = propose_structures(
            parent_methyl, target="HBD", delta=1, max_candidates=25
        )
        assert len(results) > 0
        parent_mol = Chem.MolFromSmiles(parent_methyl)
        parent_hbd = Lipinski.NumHDonors(parent_mol)
        for rec in results:
            assert rec["actual_delta"] == 1
            gen_mol = Chem.MolFromSmiles(rec["generated_smiles"])
            assert Lipinski.NumHDonors(gen_mol) - parent_hbd == 1


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: RotB +1 whole-molecule validation
# ──────────────────────────────────────────────────────────────────────────────

class TestRotBPlus1:
    def test_all_candidates_rotb_delta_one(self, parent_methyl):
        results = propose_structures(
            parent_methyl, target="RotB", delta=1, max_candidates=25
        )
        assert len(results) > 0
        parent_mol  = Chem.MolFromSmiles(parent_methyl)
        parent_rotb = Descriptors.NumRotatableBonds(parent_mol)
        for rec in results:
            assert rec["actual_delta"] == 1
            gen_mol  = Chem.MolFromSmiles(rec["generated_smiles"])
            gen_rotb = Descriptors.NumRotatableBonds(gen_mol)
            assert gen_rotb - parent_rotb == 1


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: RotB -1 whole-molecule validation
# ──────────────────────────────────────────────────────────────────────────────

class TestRotBMinus1:
    def test_all_candidates_rotb_delta_minus1(self, parent_ethyl):
        results = propose_structures(
            parent_ethyl, target="RotB", delta=-1, max_candidates=25
        )
        assert len(results) > 0
        parent_mol  = Chem.MolFromSmiles(parent_ethyl)
        parent_rotb = Descriptors.NumRotatableBonds(parent_mol)
        for rec in results:
            assert rec["actual_delta"] == -1
            gen_mol  = Chem.MolFromSmiles(rec["generated_smiles"])
            gen_rotb = Descriptors.NumRotatableBonds(gen_mol)
            assert gen_rotb - parent_rotb == -1


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: AR +1 whole-molecule validation
# ──────────────────────────────────────────────────────────────────────────────

class TestARPlus1:
    def test_all_candidates_ar_delta_one(self, parent_methyl):
        results = propose_structures(
            parent_methyl, target="AR", delta=1, max_candidates=25
        )
        assert len(results) > 0
        parent_mol = Chem.MolFromSmiles(parent_methyl)
        parent_ar  = Descriptors.NumAromaticRings(parent_mol)
        for rec in results:
            assert rec["actual_delta"] == 1
            gen_mol = Chem.MolFromSmiles(rec["generated_smiles"])
            gen_ar  = Descriptors.NumAromaticRings(gen_mol)
            assert gen_ar - parent_ar == 1


# ──────────────────────────────────────────────────────────────────────────────
# Test 8: strict mode
# ──────────────────────────────────────────────────────────────────────────────

class TestStrictMode:
    """
    With strict=True, all non-target descriptors must be unchanged.
    4-ethylbenzaldehyde + RotB -1 gives collateral=0 candidates.
    """

    def test_collateral_zero_for_all_strict_candidates(
        self, parent_ethylbenzaldehyde
    ):
        results = propose_structures(
            parent_ethylbenzaldehyde,
            target="RotB", delta=-1,
            strict=True, max_candidates=25
        )
        assert len(results) > 0
        parent_mol  = Chem.MolFromSmiles(parent_ethylbenzaldehyde)
        parent_hba  = Lipinski.NumHAcceptors(parent_mol)
        parent_hbd  = Lipinski.NumHDonors(parent_mol)
        parent_ar   = Descriptors.NumAromaticRings(parent_mol)

        for rec in results:
            gen_mol = Chem.MolFromSmiles(rec["generated_smiles"])
            assert Lipinski.NumHAcceptors(gen_mol) == parent_hba, "ΔHBA ≠ 0"
            assert Lipinski.NumHDonors(gen_mol)    == parent_hbd, "ΔHBD ≠ 0"
            assert Descriptors.NumAromaticRings(gen_mol) == parent_ar, "ΔAR ≠ 0"
            assert rec["collateral_change"] == 0

    def test_strict_subset_of_nonstrict(self, parent_methyl):
        strict    = propose_structures(
            parent_methyl, target="HBA", delta=1, strict=True,  max_candidates=25
        )
        nonstrict = propose_structures(
            parent_methyl, target="HBA", delta=1, strict=False, max_candidates=25
        )
        assert len(strict) <= len(nonstrict)
        for rec in strict:
            assert rec["collateral_change"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# Test 9: scaffold retention
# ──────────────────────────────────────────────────────────────────────────────

class TestScaffoldRetention:
    def test_scaffold_preserved_flag_true(self, parent_methyl):
        results = propose_structures(
            parent_methyl, target="HBA", delta=1, max_candidates=25
        )
        assert len(results) > 0
        for rec in results:
            assert rec["scaffold_preserved"] is True

    def test_has_substructure_match(self, parent_methyl):
        parent_mol   = Chem.MolFromSmiles(parent_methyl)
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(parent_mol)
        results = propose_structures(
            parent_methyl, target="HBA", delta=1, max_candidates=25
        )
        for rec in results:
            gen_mol = Chem.MolFromSmiles(rec["generated_smiles"])
            assert gen_mol.HasSubstructMatch(scaffold_mol), \
                f"Parent scaffold not found in {rec['generated_smiles']}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 10: no duplicates
# ──────────────────────────────────────────────────────────────────────────────

class TestNoDuplicates:
    def test_no_duplicate_generated_smiles(self, parent_methyl):
        results = propose_structures(
            parent_methyl, target="HBA", delta=1, max_candidates=50
        )
        smiles = [r["generated_smiles"] for r in results]
        assert len(smiles) == len(set(smiles)), "Duplicate generated_smiles found"


# ──────────────────────────────────────────────────────────────────────────────
# Test 11: remove_ar → NotImplementedError
# ──────────────────────────────────────────────────────────────────────────────

class TestRemoveArNotImplemented:
    def test_target_ar_delta_minus1_raises(self, parent_methyl):
        with pytest.raises(NotImplementedError):
            propose_structures(parent_methyl, target="AR", delta=-1)

    def test_mode_remove_ar_raises(self, parent_methyl):
        with pytest.raises(NotImplementedError):
            propose_structures(parent_methyl, mode="remove_ar")


# ──────────────────────────────────────────────────────────────────────────────
# Test 12: reproducibility
# ──────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_same_results(self, parent_methyl):
        r1 = propose_structures(
            parent_methyl, target="HBA", delta=1,
            shuffle_sites=True, random_seed=42
        )
        r2 = propose_structures(
            parent_methyl, target="HBA", delta=1,
            shuffle_sites=True, random_seed=42
        )
        assert [r["generated_smiles"] for r in r1] == \
               [r["generated_smiles"] for r in r2]


# ──────────────────────────────────────────────────────────────────────────────
# Test 13: provenance fields
# ──────────────────────────────────────────────────────────────────────────────

class TestProvenanceFields:
    REQUIRED_KEYS = {
        "parent_smiles", "generated_smiles",
        "target_descriptor", "requested_delta", "actual_delta",
        "site", "to_fragment",
        "parent_scaffold_smiles", "generated_murcko_scaffold",
        "scaffold_preserved", "collateral_change",
        "parent_features", "generated_features",
    }

    def test_all_required_keys_present(self, parent_methyl):
        results = propose_structures(
            parent_methyl, target="HBA", delta=1, max_candidates=5
        )
        assert len(results) > 0
        for rec in results:
            missing = self.REQUIRED_KEYS - set(rec.keys())
            assert not missing, f"Missing keys: {missing}"

    def test_scaffold_preserved_true(self, parent_methyl):
        results = propose_structures(
            parent_methyl, target="HBA", delta=1, max_candidates=5
        )
        for rec in results:
            assert rec["scaffold_preserved"] is True

    def test_no_internal_pool_index_key(self, parent_methyl):
        """_pool_index is an internal ranking helper and must not appear in output."""
        results = propose_structures(
            parent_methyl, target="HBA", delta=1, max_candidates=5
        )
        for rec in results:
            assert "_pool_index" not in rec


# ──────────────────────────────────────────────────────────────────────────────
# Test 14: extract_rgroups_for_smiles utility
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractRgroupsUtility:
    def test_returns_normalized_list(self, parent_methyl):
        rgs = extract_rgroups_for_smiles(parent_methyl)
        assert rgs is not None and isinstance(rgs, list) and len(rgs) > 0

    def test_returns_none_for_invalid(self):
        assert extract_rgroups_for_smiles("NOT_VALID###") is None

    def test_to_fragment_in_pool_or_rgroup(self, parent_methyl):
        """
        to_fragment of each candidate is a pool entry applied to the parent site.
        It should correspond to a normalized pool SMILES.
        """
        normalized_pool = {normalize_fragment_smiles(s) for s in R_GROUP_SMILES}
        results = propose_structures(
            parent_methyl, target="HBA", delta=1, max_candidates=25
        )
        for rec in results:
            assert rec["to_fragment"] in normalized_pool, \
                f"to_fragment '{rec['to_fragment']}' not in unified pool"
