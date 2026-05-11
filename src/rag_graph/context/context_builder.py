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

    def build(self, subgraphs: List[Dict[str, Any]]) -> str:
        """Format and concatenate subgraphs up to the character budget.

        Args:
            subgraphs (List[Dict[str, Any]]): Subgraph dicts from ``GraphRetriever``.

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
        return SEPARATOR.join(blocks)
