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


# ── Tier 2: sub-substitution unit tests ──────────────────────────────────────

class TestSubstitution:
    """Unit tests for _rgroup_sub_substitutions().

    This function forms the backbone of Tier 2 candidate generation:
    instead of replacing an R-group wholesale (Tier 1), it grafts each
    pool entry onto every carbon *inside* the existing R-group that
    carries at least one implicit H.
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        from scaffold_tuner.scaffold_intervention import _rgroup_sub_substitutions
        self._fn = _rgroup_sub_substitutions

    # ── image2 regression: -NH2 grafted onto the CH2 of glycine-like chain ──

    def test_image2_nh2_substitution(self):
        """
        NCC(=O)N[*:1]  →  NC(N)C(=O)N[*:1]

        The CH2 gains an NH2 branch, raising HBD by 1.
        This is the sub-substitution that produces image2 from the parent
        NCC(=O)Nc1ccc(Cl)cc1C(=O)c1ccc[nH]1.
        """
        rg = "NCC(=O)N[*:1]"
        subs = self._fn(rg, R_GROUP_SMILES)
        expected = Chem.MolToSmiles(Chem.MolFromSmiles("NC(N)C(=O)N[*:1]"))
        got = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in subs}
        assert expected in got, (
            f"Expected NC(N)C(=O)N[*:1] in sub-substitutions of {rg}"
        )

    # ── image1 regression: pyrrolyl grafted onto the same CH2 ───────────────

    def test_image1_pyrrolyl_substitution(self):
        """
        NCC(=O)N[*:1]  →  NC(c1ccc[nH]1)C(=O)N[*:1]

        The CH2 gains a 3-pyrrolyl branch, raising both HBD and AR by 1.
        This is the sub-substitution that produces image1.
        """
        rg = "NCC(=O)N[*:1]"
        subs = self._fn(rg, R_GROUP_SMILES)
        expected = Chem.MolToSmiles(Chem.MolFromSmiles("NC(c1ccc[nH]1)C(=O)N[*:1]"))
        got = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in subs}
        assert expected in got, (
            f"Expected NC(c1ccc[nH]1)C(=O)N[*:1] in sub-substitutions of {rg}"
        )

    # ── all substitutable carbons are explored, not just the nearest one ─────

    def test_multiple_carbons_all_explored(self):
        """
        CCCC[*:1] has four substitutable carbons (atoms 0–3).
        Each of the 25 pool entries can be attached at any of them, so
        the number of distinct variants must exceed 25.
        """
        rg = "CCCC[*:1]"
        subs = self._fn(rg, R_GROUP_SMILES)
        # Four carbons × 25 pool entries → up to 100 variants (some may
        # coincide after canonicalization, but must be strictly > 25).
        assert len(subs) > 25, (
            f"Expected >25 variants for CCCC[*:1] (got {len(subs)}); "
            "not all substitutable carbons are being explored."
        )

    # ── attachment-point dummy map number is preserved ───────────────────────

    def test_attachment_map_number_preserved(self):
        """
        The [*:N] dummy in the input R-group must survive unchanged in
        every generated variant so the downstream _build_molecule() call
        can still identify the correct attachment site.
        """
        rg = "NC[*:2]"   # map_num = 2
        subs = self._fn(rg, R_GROUP_SMILES[:5])
        assert len(subs) > 0
        for s in subs:
            mol = Chem.MolFromSmiles(s)
            assert mol is not None
            maps = [a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
            assert maps == [2], (
                f"Expected dummy map number 2 in {s}, got {maps}"
            )

    # ── R-groups with no substitutable carbon produce no variants ────────────

    def test_no_carbon_produces_no_variants(self):
        """
        N[*:1] (amino group, no carbon) has no substitutable carbon,
        so sub-substitution must return an empty list.
        """
        subs = self._fn("N[*:1]", R_GROUP_SMILES)
        assert subs == [], (
            f"Expected [] for N[*:1], got {subs}"
        )

    # ── all returned SMILES are valid and distinct ───────────────────────────

    def test_results_are_valid_and_unique(self):
        """Every returned SMILES must parse without error and be unique."""
        rg = "NCC(=O)N[*:1]"
        subs = self._fn(rg, R_GROUP_SMILES)
        assert len(subs) > 0
        seen = set()
        for s in subs:
            mol = Chem.MolFromSmiles(s)
            assert mol is not None, f"Invalid SMILES returned: {s}"
            assert s not in seen, f"Duplicate returned: {s}"
            seen.add(s)


# ── Tier 2: end-to-end regression via propose_structures ─────────────────────

class TestTier2EndToEnd:
    """
    Confirm that sub-substitution variants reach the final output of
    propose_structures() and pass all existing validation checks
    (exact descriptor delta, scaffold retention, no duplicates).
    """

    PARENT = "NCC(=O)Nc1ccc(Cl)cc1C(=O)c1ccc[nH]1"
    IMAGE1 = "NC(C(=O)Nc1ccc(Cl)cc1C(=O)c1ccc[nH]1)c1ccc[nH]1"
    IMAGE2 = "NC(N)C(=O)Nc1ccc(Cl)cc1C(=O)c1ccc[nH]1"

    def _canon(self, smi):
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))

    def test_image2_in_hbd_plus1_results(self):
        """
        image2 (NC(N)C(=O)Nc…) must appear in HBD+1 candidates.
        It is generated by Tier 2 (NH2 grafted onto the CH2 of R1).
        """
        results = propose_structures(self.PARENT, target="HBD", delta=1,
                                     max_candidates=50)
        smiles = {r["generated_smiles"] for r in results}
        assert self._canon(self.IMAGE2) in smiles, (
            "image2 not found in HBD+1 candidates"
        )

    def test_image1_in_hbd_plus1_results(self):
        """
        image1 (NC(c1ccc[nH]1)C(=O)Nc…) must appear in HBD+1 candidates.
        It is generated by Tier 2 (3-pyrrolyl grafted onto the CH2 of R1).
        """
        results = propose_structures(self.PARENT, target="HBD", delta=1,
                                     max_candidates=50)
        smiles = {r["generated_smiles"] for r in results}
        assert self._canon(self.IMAGE1) in smiles, (
            "image1 not found in HBD+1 candidates"
        )

    def test_tier2_results_pass_descriptor_validation(self):
        """
        Every result, including those from Tier 2, must have
        actual_delta == requested_delta for the whole molecule.
        """
        from rdkit.Chem import Lipinski
        results = propose_structures(self.PARENT, target="HBD", delta=1,
                                     max_candidates=50)
        parent_hbd = Lipinski.NumHDonors(Chem.MolFromSmiles(self.PARENT))
        for r in results:
            gen_hbd = Lipinski.NumHDonors(Chem.MolFromSmiles(r["generated_smiles"]))
            assert gen_hbd - parent_hbd == 1, (
                f"Descriptor delta wrong for {r['generated_smiles']}"
            )

    def test_tier2_results_retain_scaffold(self):
        """
        Every result, including those from Tier 2, must contain the
        parent Murcko scaffold as a substructure.
        """
        from rdkit.Chem.Scaffolds import MurckoScaffold
        parent_mol   = Chem.MolFromSmiles(self.PARENT)
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(parent_mol)
        results = propose_structures(self.PARENT, target="HBD", delta=1,
                                     max_candidates=50)
        for r in results:
            gen_mol = Chem.MolFromSmiles(r["generated_smiles"])
            assert gen_mol.HasSubstructMatch(scaffold_mol), (
                f"Scaffold not retained in {r['generated_smiles']}"
            )

    def test_no_duplicates_with_tier2(self):
        """No duplicate SMILES in the combined Tier 1 + Tier 2 output."""
        results = propose_structures(self.PARENT, target="HBD", delta=1,
                                     max_candidates=50)
        smiles = [r["generated_smiles"] for r in results]
        assert len(smiles) == len(set(smiles))

    def test_tier2_identified_by_to_fragment(self):
        """
        Tier 2 candidates have a to_fragment that is NOT a direct pool
        entry (it is an expanded R-group with a pool fragment grafted in).
        Confirm that image1 and image2 are indeed Tier 2 (not Tier 1).
        """
        pool_canon = {
            Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in R_GROUP_SMILES
        }
        results = propose_structures(self.PARENT, target="HBD", delta=1,
                                     max_candidates=50)
        by_smi = {r["generated_smiles"]: r for r in results}

        for label, target in [("image1", self.IMAGE1), ("image2", self.IMAGE2)]:
            canon = self._canon(target)
            assert canon in by_smi, f"{label} not in results"
            frag = by_smi[canon]["to_fragment"]
            frag_canon = Chem.MolToSmiles(Chem.MolFromSmiles(frag))
            assert frag_canon not in pool_canon, (
                f"{label} to_fragment {frag} is a bare pool entry; "
                "expected a Tier 2 extended R-group"
            )


# ── extract_rgroups_for_smiles ────────────────────────────────────────────────

class TestExtractRgroups:
    def test_returns_list(self, parent_methyl):
        rgs = extract_rgroups_for_smiles(parent_methyl)
        assert rgs is not None and len(rgs) > 0

    def test_returns_none_for_invalid(self):
        assert extract_rgroups_for_smiles("NOT_VALID###") is None
