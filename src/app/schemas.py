# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Pydantic request and response schemas for the GraphRAG API.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-13
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Payload for a RAG pipeline query."""

    query: str = Field(..., description="Natural language question to answer.")
    chemical_smiles: str = Field(
        default="",
        description=(
            "Optional SMILES string for chemical similarity retrieval. "
            "If omitted, the extractor/planner derives the chemical identifier from query."
        ),
    )
    industry_sector: str = Field(
        default="",
        description="Optional industry/sector name for NAICS-title semantic retrieval.",
    )
    condition_of_use: str = Field(
        default="",
        description="Optional condition-of-use text for Activity semantic retrieval.",
    )
    retrieval: Optional[Literal["fused", "legacy"]] = Field(
        default=None,
        description="Optional retrieval override: 'fused' or 'legacy'. Omitted, the configured default (fused) applies.",
    )


class QueryResponse(BaseModel):
    """Structured response from the RAG pipeline."""

    answer: str = Field(..., description="LLM-generated narrative answer.")
    records: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top-10 promising source reduction records sorted by similarity score.",
    )
    needs_clarification: bool = Field(
        default=False,
        description="True when extraction/validation failed and the answer is a clarification request.",
    )
    clarification: str = Field(
        default="",
        description="Clarification message when needs_clarification is True; empty otherwise.",
    )
    matched_uses: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Fused mode: Activity names the use channel matched (name, score), best first.",
    )
    matched_sectors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Fused mode: NAICS titles the sector channel matched (code, title, score, qualifying), best first.",
    )
