# -*- coding: utf-8 -*-
"""Shared test fixtures for ScaffoldTuner."""

import pytest

PARENT_METHYL       = "O=C(Nc1ccccc1)c1ccc(C)cc1"   # para-methyl benzanilide
PARENT_ETHYL        = "O=C(Nc1ccccc1)c1ccc(CC)cc1"  # para-ethyl benzanilide
PARENT_ETHYLBENZALD = "O=Cc1ccc(CC)cc1"              # 4-ethylbenzaldehyde

@pytest.fixture(scope="session")
def parent_methyl():
    return PARENT_METHYL

@pytest.fixture(scope="session")
def parent_ethyl():
    return PARENT_ETHYL

@pytest.fixture(scope="session")
def parent_ethylbenzaldehyde():
    return PARENT_ETHYLBENZALD
