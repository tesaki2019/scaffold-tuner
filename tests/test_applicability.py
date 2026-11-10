# -*- coding: utf-8 -*-
"""
tests/test_applicability.py  –  ScaffoldTuner v2

Tests for compute_applicability():
    Test B1: denominator is n_total_parents (not n_extraction_success)
    Test B2: extraction failures are counted as non-applicable
    Test B3: secondary metric (among_extracted) is also computed correctly
"""

import pandas as pd
import pytest

from scripts.compute_applicability import compute_applicability


# ──────────────────────────────────────────────────────────────────────────────
# Minimal library fixture for applicability tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_library() -> pd.DataFrame:
    """Library with one HBA+1 transformation: [*:1]C → [*:1]OC."""
    rows = [("[*:1]C", "[*:1]OC", 0, 1, 0, 0, 0, 10, "test")]
    cols = ["from_fragment", "to_fragment",
            "delta_HBD", "delta_HBA", "delta_AR", "delta_RotB", "delta_HA",
            "frequency", "source"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["delta_HBD", "delta_HBA", "delta_AR", "delta_RotB",
              "delta_HA", "frequency"]:
        df[c] = df[c].astype(int)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Test B1: denominator is n_total_parents
# ──────────────────────────────────────────────────────────────────────────────

class TestApplicabilityDenominator:
    """
    When 1 of 3 parents fails R-group extraction, overall applicability
    must use n_total_parents=3 as denominator, NOT n_extraction_success=2.

    Setup:
        parent_methyl  : [*:1]C group → matches library → applicable
        parent_invalid : invalid SMILES → extraction fails → NOT applicable
        parent_propyl  : [*:1]CCC → no match in library → NOT applicable

    Expected for HBA +1:
        n_total_parents     = 3
        n_extraction_success= 2
        n_extraction_failed = 1
        n_applicable        = 1
        overall_applicability_percent     = 1/3 * 100 = 33.3%
        applicability_among_extracted_pct = 1/2 * 100 = 50.0%
    """

    _PARENT_METHYL  = "O=C(Nc1ccccc1)c1ccc(C)cc1"   # applicable
    _PARENT_INVALID = "NOT_VALID###"                   # extraction fails
    _PARENT_PROPYL  = "O=C(Nc1ccccc1)c1ccc(CCC)cc1"  # no library match

    def test_overall_uses_total_parents_as_denominator(self, small_library):
        parents = [self._PARENT_METHYL, self._PARENT_INVALID, self._PARENT_PROPYL]
        df = compute_applicability(small_library, parents)
        row = df[df["task"] == "HBA +1"].iloc[0]

        assert row["n_total_parents"]     == 3,   f"Expected 3, got {row['n_total_parents']}"
        assert row["n_extraction_success"]== 2,   f"Expected 2, got {row['n_extraction_success']}"
        assert row["n_extraction_failed"] == 1,   f"Expected 1, got {row['n_extraction_failed']}"
        assert row["n_applicable"]        == 1,   f"Expected 1, got {row['n_applicable']}"

    def test_overall_applicability_is_1_over_3(self, small_library):
        parents = [self._PARENT_METHYL, self._PARENT_INVALID, self._PARENT_PROPYL]
        df = compute_applicability(small_library, parents)
        row = df[df["task"] == "HBA +1"].iloc[0]

        expected_overall = round(1 / 3 * 100, 1)
        assert row["overall_applicability_percent"] == expected_overall, (
            f"Expected {expected_overall}%, got "
            f"{row['overall_applicability_percent']}%"
        )

    def test_overall_is_not_1_over_2(self, small_library):
        """Confirm it's NOT 1/2 = 50%, which would be wrong (using only extracted)."""
        parents = [self._PARENT_METHYL, self._PARENT_INVALID, self._PARENT_PROPYL]
        df = compute_applicability(small_library, parents)
        row = df[df["task"] == "HBA +1"].iloc[0]

        wrong_value = round(1 / 2 * 100, 1)   # 50.0 — must NOT be this
        assert row["overall_applicability_percent"] != wrong_value, (
            f"overall_applicability_percent is {wrong_value}% — "
            f"this means extraction failures are being excluded from denominator"
        )

    def test_among_extracted_is_1_over_2(self, small_library):
        """Secondary metric: 1/2 = 50.0%."""
        parents = [self._PARENT_METHYL, self._PARENT_INVALID, self._PARENT_PROPYL]
        df = compute_applicability(small_library, parents)
        row = df[df["task"] == "HBA +1"].iloc[0]

        expected_secondary = round(1 / 2 * 100, 1)
        assert row["applicability_among_extracted_percent"] == expected_secondary, (
            f"Expected {expected_secondary}%, got "
            f"{row['applicability_among_extracted_percent']}%"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test B2: all parents applicable
# ──────────────────────────────────────────────────────────────────────────────

class TestAllApplicable:
    def test_full_applicability(self, small_library):
        """When all parents have matching R-groups, overall = 100%."""
        parents = [
            "O=C(Nc1ccccc1)c1ccc(C)cc1",   # [*:1]C → match
            "O=Cc1ccc(C)cc1",               # also has a methyl group
        ]
        df = compute_applicability(small_library, parents)
        row = df[df["task"] == "HBA +1"].iloc[0]
        assert row["n_total_parents"] == 2
        assert row["overall_applicability_percent"] == 100.0


# ──────────────────────────────────────────────────────────────────────────────
# Test B3: result DataFrame structure
# ──────────────────────────────────────────────────────────────────────────────

class TestResultStructure:
    EXPECTED_COLS = {
        "task", "target", "delta",
        "n_total_parents", "n_extraction_success", "n_extraction_failed",
        "n_applicable",
        "overall_applicability_percent",
        "applicability_among_extracted_percent",
    }

    def test_all_columns_present(self, small_library):
        df = compute_applicability(small_library, ["O=C(Nc1ccccc1)c1ccc(C)cc1"])
        missing = self.EXPECTED_COLS - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_all_tasks_covered(self, small_library):
        df = compute_applicability(small_library, ["O=C(Nc1ccccc1)c1ccc(C)cc1"])
        expected_tasks = {"HBD +1", "HBD -1", "HBA +1", "HBA -1",
                          "RotB +1", "RotB -1", "AR +1"}
        actual_tasks = set(df["task"].tolist())
        assert actual_tasks == expected_tasks
