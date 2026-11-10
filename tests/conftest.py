# -*- coding: utf-8 -*-
"""
Shared fixtures for ScaffoldTuner v2 unit tests.

The unified R-group pool (rgroup_pool.R_GROUP_SMILES) is used directly
by propose_structures() — no transformation library is needed.

Test parent molecules
---------------------
PARENT_METHYL  : O=C(Nc1ccccc1)c1ccc(C)cc1   para-methyl benzanilide
PARENT_AMINO   : O=C(Nc1ccccc1)c1ccc(N)cc1   para-amino benzanilide
PARENT_ETHYL   : O=C(Nc1ccccc1)c1ccc(CC)cc1  para-ethyl benzanilide
PARENT_ETHYLBA : O=Cc1ccc(CC)cc1              4-ethylbenzaldehyde (strict=True tests)
"""

import pytest

PARENT_METHYL        = "O=C(Nc1ccccc1)c1ccc(C)cc1"
PARENT_AMINO         = "O=C(Nc1ccccc1)c1ccc(N)cc1"
PARENT_ETHYL         = "O=C(Nc1ccccc1)c1ccc(CC)cc1"
PARENT_ETHYLBENZALD  = "O=Cc1ccc(CC)cc1"


@pytest.fixture(scope="session")
def parent_methyl() -> str:
    return PARENT_METHYL

@pytest.fixture(scope="session")
def parent_amino() -> str:
    return PARENT_AMINO

@pytest.fixture(scope="session")
def parent_ethyl() -> str:
    return PARENT_ETHYL

@pytest.fixture(scope="session")
def parent_ethylbenzaldehyde() -> str:
    return PARENT_ETHYLBENZALD
