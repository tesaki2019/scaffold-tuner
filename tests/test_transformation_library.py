# -*- coding: utf-8 -*-
"""
tests/test_transformation_library.py  –  ScaffoldTuner v2

Tests A–G for load_transformation_library() and retrieve_transformations().
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from scaffold_tuner.scaffold_intervention import (
    load_transformation_library,
    retrieve_transformations,
    _LIB_CACHE,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_VALID_CSV = (
    "from_fragment,to_fragment,delta_HBD,delta_HBA,delta_AR,delta_RotB,"
    "frequency,source\n"
    "[*:1]C,[*:1]OC,0,1,0,1,50,test\n"
    "[*:1]C,[*:1]N,1,1,0,0,40,test\n"
    "[*:1]N,[*:1]C,-1,-1,0,0,35,test\n"
    "[*:1]C,[*:1]CC,0,0,0,1,30,test\n"
)


def _tmp_csv(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w", encoding="utf-8"
    )
    f.write(content)
    f.flush()
    f.close()
    return Path(f.name)


# ──────────────────────────────────────────────────────────────────────────────
# Test A: load valid CSV
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadValidCsv:
    def test_returns_dataframe(self):
        p = _tmp_csv(_VALID_CSV)
        try:
            _LIB_CACHE.pop(str(p), None)
            df = load_transformation_library(p)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 4
        finally:
            p.unlink(missing_ok=True)

    def test_required_columns_present(self):
        p = _tmp_csv(_VALID_CSV)
        try:
            _LIB_CACHE.pop(str(p), None)
            df = load_transformation_library(p)
            for col in ["from_fragment", "to_fragment",
                        "delta_HBD", "delta_HBA", "delta_AR", "delta_RotB"]:
                assert col in df.columns
        finally:
            p.unlink(missing_ok=True)

    def test_delta_columns_are_int(self):
        p = _tmp_csv(_VALID_CSV)
        try:
            _LIB_CACHE.pop(str(p), None)
            df = load_transformation_library(p)
            for col in ["delta_HBD", "delta_HBA", "delta_AR", "delta_RotB"]:
                assert df[col].dtype == int
        finally:
            p.unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Test B: load missing file
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadMissingFile:
    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_transformation_library("/nonexistent/library.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Test C: load missing required columns
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadMissingRequiredColumns:
    def test_raises_value_error(self):
        # CSV missing delta_HBA
        content = (
            "from_fragment,to_fragment,delta_HBD,delta_AR,delta_RotB\n"
            "[*:1]C,[*:1]OC,0,0,1\n"
        )
        p = _tmp_csv(content)
        try:
            _LIB_CACHE.pop(str(p), None)
            with pytest.raises(ValueError, match="missing required columns"):
                load_transformation_library(p)
        finally:
            p.unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Test D: optional columns filled with defaults
# ──────────────────────────────────────────────────────────────────────────────

class TestOptionalColumnDefaults:
    _MINIMAL = (
        "from_fragment,to_fragment,delta_HBD,delta_HBA,delta_AR,delta_RotB\n"
        "[*:1]C,[*:1]OC,0,1,0,1\n"
    )

    def test_delta_ha_default_zero(self):
        p = _tmp_csv(self._MINIMAL)
        try:
            _LIB_CACHE.pop(str(p), None)
            df = load_transformation_library(p)
            assert "delta_HA" in df.columns
            assert (df["delta_HA"] == 0).all()
        finally:
            p.unlink(missing_ok=True)

    def test_frequency_default_one(self):
        p = _tmp_csv(self._MINIMAL)
        try:
            _LIB_CACHE.pop(str(p), None)
            df = load_transformation_library(p)
            assert (df["frequency"] == 1).all()
        finally:
            p.unlink(missing_ok=True)

    def test_source_default_unknown(self):
        p = _tmp_csv(self._MINIMAL)
        try:
            _LIB_CACHE.pop(str(p), None)
            df = load_transformation_library(p)
            assert (df["source"] == "unknown").all()
        finally:
            p.unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Test E: retrieve exact delta
# ──────────────────────────────────────────────────────────────────────────────

class TestRetrieveExactDelta:
    def test_only_matching_rows_returned(self, minimal_library):
        result = retrieve_transformations(minimal_library, "HBA", 1)
        assert len(result) > 0
        assert (result["delta_HBA"] == 1).all()

    def test_no_non_matching_rows(self, minimal_library):
        result = retrieve_transformations(minimal_library, "HBA", 1)
        assert not (result["delta_HBA"] != 1).any()

    def test_negative_delta(self, minimal_library):
        result = retrieve_transformations(minimal_library, "HBD", -1)
        assert len(result) > 0
        assert (result["delta_HBD"] == -1).all()


# ──────────────────────────────────────────────────────────────────────────────
# Test F: retrieve empty when no match
# ──────────────────────────────────────────────────────────────────────────────

class TestRetrieveEmptyWhenNoMatch:
    def test_empty_for_nonexistent_delta(self, minimal_library):
        result = retrieve_transformations(minimal_library, "HBA", 99)
        assert len(result) == 0

    def test_empty_for_ar_minus1(self, minimal_library):
        result = retrieve_transformations(minimal_library, "AR", -1)
        assert len(result) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Test G: load caches result
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadCachesResult:
    def test_same_object_on_second_call(self):
        p = _tmp_csv(_VALID_CSV)
        try:
            _LIB_CACHE.pop(str(p), None)
            df1 = load_transformation_library(p)
            df2 = load_transformation_library(p)
            assert df1 is df2, "Second call should return the cached DataFrame"
        finally:
            p.unlink(missing_ok=True)
            _LIB_CACHE.pop(str(p), None)
