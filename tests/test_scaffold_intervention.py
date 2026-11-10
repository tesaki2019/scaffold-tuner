# -*- coding: utf-8 -*-
"""
tests/test_scaffold_intervention.py  –  ScaffoldTuner v2

Tests 1–10 for propose_structures().
"""

import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold

from scaffold_tuner.scaffold_intervention import (
    propose_structures,
    normalize_fragment_smiles,
)


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: exact matching is the default
# ──────────────────────────────────────────────────────────────────────────────

class TestExactMatchingDefault:
    """
    In match_mode="exact" (default), every candidate's from_fragment must
    be canonically identical to the parent R-group at the substituted site.
    """

    def test_match_type_is_exact(self, parent_methyl, minimal_library):
        results = propose_structures(
            parent_methyl,
            target="HBA", delta=1,
            transformation_library=minimal_library,
        )
        assert len(results) > 0, "Expected at least one candidate"
        for rec in results:
            assert rec["match_type"] == "exact", (
                f"match_type should be 'exact', got '{rec['match_type']}'"
            )

    def test_match_similarity_is_one(self, parent_methyl, minimal_library):
        results = propose_structures(
            parent_methyl,
            target="HBA", delta=1,
            transformation_library=minimal_library,
        )
        for rec in results:
            assert rec["match_similarity"] == 1.0, (
                f"Expected similarity 1.0 for exact match, "
                f"got {rec['match_similarity']}"
            )

    def test_from_fragment_equals_parent_rgroup(self, parent_methyl, minimal_library):
        """
        The normalized from_fragment must equal the normalized parent R-group
        at the substituted site.  We verify this by checking that the
        match_type is "exact" AND match_similarity is 1.0 (both guaranteed
        by _get_applicable_transforms when exact matching succeeds).
        """
        results = propose_structures(
            parent_methyl,
            target="HBA", delta=1,
            transformation_library=minimal_library,
        )
        for rec in results:
            assert rec["match_type"] == "exact"
            # Verify from_fragment is in the library rows with delta_HBA == 1
            lib_froms = set(
                minimal_library[minimal_library["delta_HBA"] == 1]["from_fragment"]
                .apply(normalize_fragment_smiles)
            )
            assert normalize_fragment_smiles(rec["from_fragment"]) in lib_froms


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: no silent similarity fallback in exact mode
# ──────────────────────────────────────────────────────────────────────────────

class TestNoSilentSimilarityFallback:
    """
    When match_mode="exact" and no library from_fragment matches any parent
    R-group for the requested task, the result must be empty.
    No automatic switch to similarity matching.
    """

    def test_returns_empty_when_no_exact_match(self, minimal_library):
        # 4-propylbenzanilide: R-group is [*:1]CCC (propyl).
        # The minimal library has no HBA+1 transformation with from_fragment=[*:1]CCC.
        parent_propyl = "O=C(Nc1ccccc1)c1ccc(CCC)cc1"
        results = propose_structures(
            parent_propyl,
            target="HBA", delta=1,
            transformation_library=minimal_library,
            match_mode="exact",
        )
        assert len(results) == 0, (
            f"Expected 0 candidates in exact mode with no matching "
            f"from_fragment, got {len(results)}"
        )

    def test_similarity_mode_can_find_when_exact_cannot(self, minimal_library):
        """Contrast: similarity mode may find candidates for the same parent."""
        parent_propyl = "O=C(Nc1ccccc1)c1ccc(CCC)cc1"
        exact_results = propose_structures(
            parent_propyl,
            target="HBA", delta=1,
            transformation_library=minimal_library,
            match_mode="exact",
        )
        sim_results = propose_structures(
            parent_propyl,
            target="HBA", delta=1,
            transformation_library=minimal_library,
            match_mode="similarity",
            similarity_threshold=0.1,
        )
        # exact must return 0; similarity may return >= 0
        assert len(exact_results) == 0
        for rec in sim_results:
            assert rec["match_type"] == "similarity"


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: exact descriptor delta – HBA
# ──────────────────────────────────────────────────────────────────────────────

class TestExactDeltaHBA:
    """All returned candidates: generated_HBA - parent_HBA == 1."""

    def test_delta_equals_one(self, parent_methyl, minimal_library):
        results = propose_structures(
            parent_methyl,
            target="HBA", delta=1,
            transformation_library=minimal_library,
        )
        assert len(results) > 0
        parent_mol  = Chem.MolFromSmiles(parent_methyl)
        parent_hba  = Lipinski.NumHAcceptors(parent_mol)

        for rec in results:
            assert rec["actual_delta"] == 1
            gen_mol = Chem.MolFromSmiles(rec["generated_smiles"])
            assert Lipinski.NumHAcceptors(gen_mol) - parent_hba == 1, (
                f"ΔHBA != 1 for {rec['generated_smiles']}"
            )

    def test_actual_delta_matches_requested(self, parent_methyl, minimal_library):
        results = propose_structures(
            parent_methyl,
            target="HBA", delta=1,
            transformation_library=minimal_library,
        )
        for rec in results:
            assert rec["actual_delta"] == rec["requested_delta"]


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: exact descriptor delta – HBD
# ──────────────────────────────────────────────────────────────────────────────

class TestExactDeltaHBD:
    """All returned candidates: generated_HBD - parent_HBD == 1."""

    def test_delta_equals_one(self, parent_methyl, minimal_library):
        results = propose_structures(
            parent_methyl,
            target="HBD", delta=1,
            transformation_library=minimal_library,
        )
        if not results:
            pytest.skip("No HBD+1 candidates for this parent/library")

        parent_mol  = Chem.MolFromSmiles(parent_methyl)
        parent_hbd  = Lipinski.NumHDonors(parent_mol)

        for rec in results:
            assert rec["actual_delta"] == 1
            gen_mol = Chem.MolFromSmiles(rec["generated_smiles"])
            assert Lipinski.NumHDonors(gen_mol) - parent_hbd == 1


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: RotB whole-molecule validation
# ──────────────────────────────────────────────────────────────────────────────

class TestRotBWholeMolecule:
    """
    RotB is validated on the full generated molecule.
    Any candidate whose whole-molecule actual_delta != requested_delta
    must be excluded even if library_delta matches.
    """

    def test_actual_delta_always_equals_requested(self, parent_methyl, minimal_library):
        results = propose_structures(
            parent_methyl,
            target="RotB", delta=1,
            transformation_library=minimal_library,
        )
        if not results:
            pytest.skip("No RotB+1 candidates for this parent/library")

        parent_mol   = Chem.MolFromSmiles(parent_methyl)
        parent_rotb  = Descriptors.NumRotatableBonds(parent_mol)

        for rec in results:
            assert rec["actual_delta"] == rec["requested_delta"], (
                f"actual_delta ({rec['actual_delta']}) != "
                f"requested_delta ({rec['requested_delta']})"
            )
            gen_mol  = Chem.MolFromSmiles(rec["generated_smiles"])
            gen_rotb = Descriptors.NumRotatableBonds(gen_mol)
            assert gen_rotb - parent_rotb == rec["requested_delta"]

    def test_library_delta_and_actual_delta_may_differ_but_actual_wins(
        self, parent_methyl, minimal_library
    ):
        """
        library_delta is stored for provenance only.
        actual_delta is the RDKit whole-molecule value and is what matters.
        For all returned candidates actual_delta == requested_delta.
        """
        results = propose_structures(
            parent_methyl,
            target="RotB", delta=1,
            transformation_library=minimal_library,
        )
        for rec in results:
            # actual_delta is always the ground truth
            assert rec["actual_delta"] == 1
            # library_delta is stored as-is (may equal or differ)
            assert "library_delta" in rec


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: scaffold retention
# ──────────────────────────────────────────────────────────────────────────────

class TestScaffoldRetention:
    """All candidates: generated_mol.HasSubstructMatch(parent_scaffold)."""

    def test_scaffold_preserved_flag(self, parent_methyl, minimal_library):
        results = propose_structures(
            parent_methyl,
            target="HBA", delta=1,
            transformation_library=minimal_library,
        )
        assert len(results) > 0
        for rec in results:
            assert rec["scaffold_preserved"] is True

    def test_substructure_match(self, parent_methyl, minimal_library):
        parent_mol   = Chem.MolFromSmiles(parent_methyl)
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(parent_mol)

        results = propose_structures(
            parent_methyl,
            target="HBA", delta=1,
            transformation_library=minimal_library,
        )
        assert len(results) > 0
        for rec in results:
            gen_mol = Chem.MolFromSmiles(rec["generated_smiles"])
            assert gen_mol.HasSubstructMatch(scaffold_mol), (
                f"Parent scaffold not found in {rec['generated_smiles']}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: strict mode
# ──────────────────────────────────────────────────────────────────────────────

class TestStrictMode:
    """
    With strict=True all non-target descriptors must be unchanged.

    Test molecule: 4-ethylbenzaldehyde (O=Cc1ccc(CC)cc1)
    Transformation: [*:1]CC -> [*:1]C  (RotB -1, collateral = 0)
    """

    def test_collateral_change_is_zero(
        self, parent_ethylbenzaldehyde, strict_library
    ):
        results = propose_structures(
            parent_ethylbenzaldehyde,
            target="RotB", delta=-1,
            transformation_library=strict_library,
            strict=True,
        )
        assert len(results) > 0, (
            "Expected at least one strict=True candidate for RotB -1"
        )
        parent_mol  = Chem.MolFromSmiles(parent_ethylbenzaldehyde)
        parent_hba  = Lipinski.NumHAcceptors(parent_mol)
        parent_hbd  = Lipinski.NumHDonors(parent_mol)
        parent_ar   = Descriptors.NumAromaticRings(parent_mol)

        for rec in results:
            gen_mol = Chem.MolFromSmiles(rec["generated_smiles"])
            assert Lipinski.NumHAcceptors(gen_mol) == parent_hba, "ΔHBA ≠ 0"
            assert Lipinski.NumHDonors(gen_mol)    == parent_hbd, "ΔHBD ≠ 0"
            assert Descriptors.NumAromaticRings(gen_mol) == parent_ar, "ΔAR ≠ 0"
            assert rec["collateral_change"] == 0, (
                f"collateral_change={rec['collateral_change']} in strict mode"
            )

    def test_strict_excludes_collateral_candidates(
        self, parent_methyl, minimal_library
    ):
        """
        Candidates with collateral changes are excluded by strict=True.
        The same call with strict=False may return more candidates.
        """
        strict  = propose_structures(
            parent_methyl,
            target="HBA", delta=1,
            transformation_library=minimal_library,
            strict=True,
        )
        nonstrict = propose_structures(
            parent_methyl,
            target="HBA", delta=1,
            transformation_library=minimal_library,
            strict=False,
        )
        # strict subset ≤ non-strict
        assert len(strict) <= len(nonstrict)
        # All strict results have collateral_change == 0
        for rec in strict:
            assert rec["collateral_change"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# Test 8: no duplicate SMILES
# ──────────────────────────────────────────────────────────────────────────────

class TestNoDuplicates:
    def test_no_duplicate_generated_smiles(self, parent_methyl, minimal_library):
        results = propose_structures(
            parent_methyl,
            target="HBA", delta=1,
            transformation_library=minimal_library,
            max_candidates=50,
        )
        smiles = [r["generated_smiles"] for r in results]
        assert len(smiles) == len(set(smiles)), "Duplicate generated_smiles"


# ──────────────────────────────────────────────────────────────────────────────
# Test 9: reserved future transformation remains unavailable in current backend
# ──────────────────────────────────────────────────────────────────────────────

class TestRemoveArUnsupported:
    def test_target_ar_delta_minus1_raises(self, parent_methyl, minimal_library):
        with pytest.raises(NotImplementedError, match="future transformation"):
            propose_structures(
                parent_methyl,
                target="AR", delta=-1,
                transformation_library=minimal_library,
            )

    def test_mode_remove_ar_raises(self, parent_methyl, minimal_library):
        with pytest.raises(NotImplementedError, match="future transformation"):
            propose_structures(
                parent_methyl,
                mode="remove_ar",
                transformation_library=minimal_library,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Test 10: provenance fields
# ──────────────────────────────────────────────────────────────────────────────

class TestProvenanceFields:
    REQUIRED_KEYS = {
        "parent_smiles", "generated_smiles",
        "target_descriptor", "requested_delta", "library_delta", "actual_delta",
        "site", "from_fragment", "to_fragment",
        "transformation_frequency", "transformation_source",
        "match_type", "match_similarity",
        "parent_scaffold_smiles", "generated_murcko_scaffold",
        "scaffold_preserved", "collateral_change",
        "parent_features", "generated_features",
    }

    def test_all_keys_present(self, parent_methyl, minimal_library):
        results = propose_structures(
            parent_methyl,
            target="HBA", delta=1,
            transformation_library=minimal_library,
        )
        assert len(results) > 0
        for rec in results:
            missing = self.REQUIRED_KEYS - set(rec.keys())
            assert not missing, f"Missing keys: {missing}"

    def test_frequency_and_source_types(self, parent_methyl, minimal_library):
        results = propose_structures(
            parent_methyl,
            target="HBA", delta=1,
            transformation_library=minimal_library,
        )
        for rec in results:
            assert isinstance(rec["transformation_frequency"], int)
            assert isinstance(rec["transformation_source"], str)

    def test_scaffold_info_present(self, parent_methyl, minimal_library):
        results = propose_structures(
            parent_methyl,
            target="HBA", delta=1,
            transformation_library=minimal_library,
        )
        for rec in results:
            assert rec["parent_scaffold_smiles"] != ""
            assert rec["scaffold_preserved"] is True


# ──────────────────────────────────────────────────────────────────────────────
# Test 11: reproducibility (bonus – required by spec)
# ──────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    """Same seed produces same results."""

    def test_same_seed_same_results(self, parent_methyl, minimal_library):
        r1 = propose_structures(
            parent_methyl,
            target="HBA", delta=1,
            transformation_library=minimal_library,
            shuffle_sites=True,
            random_seed=42,
        )
        r2 = propose_structures(
            parent_methyl,
            target="HBA", delta=1,
            transformation_library=minimal_library,
            shuffle_sites=True,
            random_seed=42,
        )
        smiles1 = [r["generated_smiles"] for r in r1]
        smiles2 = [r["generated_smiles"] for r in r2]
        assert smiles1 == smiles2, "Same seed should produce identical results"


# ──────────────────────────────────────────────────────────────────────────────
# Test 12: extract_rgroups_for_smiles – shared R-group utility
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractRgroupsForSmiles:
    """
    Test A: generation and evaluation use the same R-group extraction.

    extract_rgroups_for_smiles() is the shared public utility called by
    both propose_structures() (generation side) and compute_applicability.py
    (evaluation side).  Its output must be consistent with the R-groups
    used internally during generation.
    """

    def test_returns_list_of_normalized_smiles(self, parent_methyl):
        from scaffold_tuner.scaffold_intervention import (
            extract_rgroups_for_smiles,
            normalize_fragment_smiles,
        )
        rgs = extract_rgroups_for_smiles(parent_methyl)
        assert rgs is not None
        assert isinstance(rgs, list)
        assert len(rgs) > 0
        # Every item is a non-empty SMILES string
        for smi in rgs:
            assert isinstance(smi, str) and len(smi) > 0

    def test_returns_none_for_invalid_smiles(self):
        from scaffold_tuner.scaffold_intervention import extract_rgroups_for_smiles
        result = extract_rgroups_for_smiles("NOT_A_VALID_SMILES###")
        assert result is None

    def test_rgroups_match_generation_side(self, parent_methyl, minimal_library):
        """
        The R-groups extracted by extract_rgroups_for_smiles() must include
        the from_fragment of candidates returned by propose_structures().
        This confirms generation and evaluation share the same R-group definition.
        """
        from scaffold_tuner.scaffold_intervention import (
            extract_rgroups_for_smiles,
            normalize_fragment_smiles,
        )
        rgs = extract_rgroups_for_smiles(parent_methyl)
        assert rgs is not None

        results = propose_structures(
            parent_methyl,
            target="HBA", delta=1,
            transformation_library=minimal_library,
        )
        assert len(results) > 0

        # Each candidate's from_fragment (normalized) must appear in the
        # R-groups extracted by extract_rgroups_for_smiles()
        for rec in results:
            norm_from = normalize_fragment_smiles(rec["from_fragment"])
            assert norm_from in rgs, (
                f"from_fragment '{norm_from}' not found in "
                f"extract_rgroups_for_smiles() output: {rgs}"
            )
