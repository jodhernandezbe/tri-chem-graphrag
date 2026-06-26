# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Assembles multiple subgraph text blocks into a single context string.

Respects the ``MAX_CONTEXT_CHARS`` budget from config so the combined context
never exceeds what the LLM prompt can comfortably accommodate.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-09
"""

import logging
from typing import Any, Dict, List

from src.config_manager import ConfigManager

from .subgraph_formatter import format_subgraph

logger = logging.getLogger(__name__)

SEPARATOR = "\n" + "-" * 60 + "\n"


class ContextBuilder:
    """Converts a list of subgraph dicts into a single, budget-capped context string."""

    def __init__(self) -> None:
        self._max_chars: int = ConfigManager().config["RAG_GRAPH"]["RETRIEVAL"]["MAX_CONTEXT_CHARS"]

    def build(self, subgraphs: List[Dict[str, Any]], query_smiles: str = "") -> str:
        """Format and concatenate subgraphs up to the character budget.

        Args:
            subgraphs (List[Dict[str, Any]]): Subgraph dicts from ``GraphRetriever``.
                A ``similarity`` key, when present, is rendered in each block header so
                the LLM can treat close structural analogues as transferable.
            query_smiles (str): SMILES of the query chemical; when provided, a short
                header orients the LLM to the query and its similar chemicals.

        Returns:
            str: Combined context ready for prompt injection. Empty string if no
                valid subgraphs are provided.
        """
        blocks: List[str] = []
        total = 0
        for sg in subgraphs:
            block = format_subgraph(sg)
            if not block:
                continue
            if total + len(block) > self._max_chars:
                logger.debug(f"Context budget reached at {total} chars; {len(subgraphs) - len(blocks)} subgraph(s) dropped")
                break
            blocks.append(block)
            total += len(block)
        body = SEPARATOR.join(blocks)
        if query_smiles and body:
            header = (
                f"Query chemical SMILES: {query_smiles}\n"
                "The blocks below are the query chemical and its structurally most similar chemicals "
                "(each header shows the structural Similarity to the query; 1.0 marks the query chemical itself).\n"
            )
            return header + SEPARATOR + body
        return body
