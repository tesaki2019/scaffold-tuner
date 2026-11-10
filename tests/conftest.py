# -*- coding: utf-8 -*-
"""
Shared fixtures for ScaffoldTuner v2 unit tests.

A minimal in-memory transformation library is defined here so that
tests run without any external file dependency.

Library design
--------------
All from_fragment / to_fragment use [*:1] as the attachment dummy.
Rows are chosen to exercise specific test scenarios:

  HBA +1:  [*:1]C  → [*:1]OC      (methyl → methoxy)
  HBD +1:  [*:1]C  → [*:1]N       (methyl → amino)
  HBD -1:  [*:1]N  → [*:1]C       (amino  → methyl)
  RotB +1: [*:1]C  → [*:1]CC      (methyl → ethyl)
  RotB -1: [*:1]CC → [*:1]C       (ethyl  → methyl)
  AR  +1:  [*:1]C  → [*:1]c1ccccc1 (methyl → phenyl)

Parent molecules
----------------
PARENT_METHYL = 4-methylbenzanilide   (R-group [*:1]C at para position)
PARENT_AMINO  = 4-aminobenzanilide    (R-group [*:1]N at para position)
PARENT_ETHYL  = 4-ethylbenzanilide    (R-group [*:1]CC at para position)
"""

import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Minimal transformation library
# ──────────────────────────────────────────────────────────────────────────────

_LIB_ROWS = [
    # from_fragment    to_fragment          dHBD dHBA dAR dRotB dHA freq source
    ("[*:1]C",  "[*:1]OC",           0,   1,   0,   1,   1,  50, "test"),
    ("[*:1]C",  "[*:1]N",            1,   1,   0,   0,   1,  40, "test"),
    ("[*:1]N",  "[*:1]C",           -1,  -1,   0,   0,  -1,  35, "test"),
    ("[*:1]C",  "[*:1]CC",           0,   0,   0,   1,   0,  30, "test"),
    ("[*:1]CC", "[*:1]C",            0,   0,   0,  -1,   0,  28, "test"),
    ("[*:1]C",  "[*:1]c1ccccc1",     0,   0,   1,   1,   0,  25, "test"),
]
# strict=True library: [*:1]CC -> [*:1]C gives RotB-1 with collateral=0
# when used on O=Cc1ccc(CC)cc1 (4-ethylbenzaldehyde)
_STRICT_LIB_ROWS = [
    ("[*:1]CC", "[*:1]C",  0, 0, 0, -1, 0, 28, "test"),
]
_LIB_COLS = [
    "from_fragment", "to_fragment",
    "delta_HBD", "delta_HBA", "delta_AR", "delta_RotB", "delta_HA",
    "frequency", "source",
]


@pytest.fixture(scope="session")
def minimal_library() -> pd.DataFrame:
    """Minimal in-memory transformation library used by all tests."""
    df = pd.DataFrame(_LIB_ROWS, columns=_LIB_COLS)
    for col in ["delta_HBD", "delta_HBA", "delta_AR", "delta_RotB",
                "delta_HA", "frequency"]:
        df[col] = df[col].astype(int)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Test parent molecules
# ──────────────────────────────────────────────────────────────────────────────

PARENT_METHYL = "O=C(Nc1ccccc1)c1ccc(C)cc1"    # para-methyl: R-group [*:1]C
PARENT_AMINO  = "O=C(Nc1ccccc1)c1ccc(N)cc1"    # para-amino:  R-group [*:1]N
PARENT_ETHYL  = "O=C(Nc1ccccc1)c1ccc(CC)cc1"   # para-ethyl:  R-group [*:1]CC


@pytest.fixture(scope="session")
def parent_methyl() -> str:
    return PARENT_METHYL


@pytest.fixture(scope="session")
def parent_amino() -> str:
    return PARENT_AMINO


@pytest.fixture(scope="session")
def parent_ethyl() -> str:
    return PARENT_ETHYL


# ── strict mode test fixtures ─────────────────────────────────────────────────

@pytest.fixture(scope="session")
def strict_library() -> pd.DataFrame:
    """
    Minimal library for strict=True tests.
    [*:1]CC -> [*:1]C gives RotB-1 with collateral=0 on 4-ethylbenzaldehyde.
    """
    df = pd.DataFrame(_STRICT_LIB_ROWS, columns=_LIB_COLS)
    for col in ["delta_HBD", "delta_HBA", "delta_AR", "delta_RotB",
                "delta_HA", "frequency"]:
        df[col] = df[col].astype(int)
    return df


@pytest.fixture(scope="session")
def parent_ethylbenzaldehyde() -> str:
    """4-ethylbenzaldehyde: R-group [*:1]CC; RotB-1 gives collateral=0."""
    return "O=Cc1ccc(CC)cc1"
