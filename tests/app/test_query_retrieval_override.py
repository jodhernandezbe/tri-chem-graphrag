# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Unit tests for the optional per-request retrieval override of the API."""

import asyncio
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.app.schemas import QueryRequest

RESULT: Dict[str, Any] = {
    "answer": "a",
    "records": [{"Tier": 1, "Channels": ["chemical"]}],
    "needs_clarification": False,
    "clarification": "",
}


def _call(retrieval: Optional[str]) -> MagicMock:
    from src.app.main import query

    pipeline = MagicMock()
    pipeline.run_structured.return_value = RESULT
    request = QueryRequest(query="q", retrieval=retrieval)  # type: ignore[arg-type]
    with patch("src.app.main.RAGPipeline", return_value=pipeline):
        response = asyncio.run(query(request))
    assert response.records[0]["Tier"] == 1
    return pipeline.run_structured


class TestRetrievalOverride:
    def test_omitted_defers_to_config(self) -> None:
        assert _call(None).call_args.kwargs["switches"] is None

    def test_legacy_is_passed_through(self) -> None:
        assert _call("legacy").call_args.kwargs["switches"] == {"retrieval": "legacy"}

    def test_unknown_value_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(query="q", retrieval="hybrid")  # type: ignore[arg-type]
