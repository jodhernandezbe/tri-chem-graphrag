# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Tests for the extraction schema's manuscript id pattern (round 1: MS_01 to MS_10; round 2: MS_11 to MS_30).

Author: Jose D. Hernandez-Betancur
Date: 2026-11-04
"""

import importlib.util
import pathlib
import re
from types import ModuleType

import pytest

SCHEMA = pathlib.Path(__file__).resolve().parents[2] / "manuscript" / "scripts" / "extraction_schema.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("extraction_schema_under_test", SCHEMA)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not SCHEMA.exists(), reason="manuscript scripts not present")
class TestIdPattern:
    @pytest.mark.parametrize("value", ["MS_01", "MS_09", "MS_10", "MS_11", "MS_19", "MS_20", "MS_29", "MS_30"])
    def test_accepts_round1_and_round2_ids(self, value: str) -> None:
        assert re.fullmatch(_load().ID_PATTERN, value)

    @pytest.mark.parametrize("value", ["MS_00", "MS_31", "MS_100", "ms_05", "MS_5", "MS_1a"])
    def test_rejects_other_ids(self, value: str) -> None:
        assert re.fullmatch(_load().ID_PATTERN, value) is None

    def test_validator_uses_the_pattern(self) -> None:
        module = _load()
        assert module.Extraction.model_fields["manuscript_id"] is not None
        with pytest.raises(ValueError):
            module.Extraction._check_id.__func__(module.Extraction, "MS_31")  # type: ignore[attr-defined]
        assert module.Extraction._check_id.__func__(module.Extraction, "MS_12") == "MS_12"  # type: ignore[attr-defined]

    def test_round2_glob_is_separate_from_round1(self) -> None:
        module = _load()
        assert module.ROUND2_EXTRACTIONS != module.EXTRACTIONS and "extractions_round2" in module.ROUND2_EXTRACTIONS
