# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Pydantic request and response schemas for the GraphRAG API.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-13
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Payload for a RAG pipeline query."""

    query: str = Field(..., description="Natural language question to answer.")
    chemical_smiles: str = Field(
        default="",
        description=(
            "Optional SMILES string for chemical similarity retrieval. "
            "If omitted, the query planner extracts the chemical identifier from query."
        ),
    )


class QueryResponse(BaseModel):
    """Structured response from the RAG pipeline."""

    answer: str = Field(..., description="LLM-generated narrative answer.")
    records: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top-10 promising source reduction records sorted by similarity score.",
    )
