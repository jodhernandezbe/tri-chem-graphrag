# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Cleaning helpers for not-quite-JSON output emitted by small judge LLMs.

Small instruction-following models (e.g. granite3.3:8b used as the RAGAS judge) often
wrap JSON in markdown fences, add trailing commas, use smart quotes, or insert comments.
RAGAS' Pydantic output parser then fails, yielding NaN faithfulness / context-recall.
These helpers normalise such output before it reaches the parser.

Author: Jose D. Hernandez-Betancur
Date: 2026-06-25
"""

import re


def clean_llm_json(raw: str) -> str:
    """Strip markdown code fences and JS-style comments from an LLM JSON response.

    Args:
        raw (str): Raw string returned by the LLM.

    Returns:
        str: Fence- and comment-free JSON string.
    """
    cleaned = raw
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0]
    cleaned = re.sub(r"(?m)^\s*//[^\n]*\n?", "", cleaned)
    return cleaned.strip()


def fix_json_formatting(json_str: str) -> str:
    """Fix common JSON formatting issues produced by LLMs.

    Normalises smart/curly quotes, inserts missing commas between value/next-key pairs,
    and removes trailing commas before a closing brace or bracket.

    Args:
        json_str (str): Potentially malformed JSON string from an LLM.

    Returns:
        str: Corrected JSON string.
    """
    json_str = json_str.replace("“", '"').replace("”", '"').replace("’", "'")
    json_str = re.sub(
        r'((?:"(?:\\.|[^"\\])*")|(?:-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|true|false|null|\}|\])\s*(?="[^"]+"\s*:)',
        r"\1, ",
        json_str,
    )
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
    return json_str


def clean_json_text(raw: str) -> str:
    """Apply fence/comment stripping then formatting fixes in one call.

    Args:
        raw (str): Raw LLM output.

    Returns:
        str: Cleaned JSON string ready for ``model_validate_json`` / RAGAS parsing.
    """
    return fix_json_formatting(clean_llm_json(raw))
