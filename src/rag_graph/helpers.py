#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shared helper utilities for the workflow package.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-13

"""

import re


def clean_llm_json(raw: str) -> str:
    """Strip markdown code fences from an LLM JSON response.

    LLMs sometimes wrap JSON output in ```json ... ``` or ``` ... ```
    blocks. This function removes those fences so the result can be
    passed directly to `model_validate_json`.

    Args:
        raw (str): Raw string returned by the LLM.

    Returns:
        str: Clean JSON string ready for parsing.

    """
    cleaned = raw
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0]

    # Strip JavaScript-style single-line comments, not valid JSON but sometimes emitted by LLMs
    cleaned = re.sub(r"(?m)^\s*//[^\n]*\n?", "", cleaned)

    return cleaned.strip()


def fix_json_formatting(json_str: str) -> str:
    """Fix common JSON formatting issues produced by LLMs.

    Handles smart/curly quotes, missing commas between key-value pairs,
    and trailing commas before closing braces or brackets.

    Args:
        json_str (str): Potentially malformed JSON string from an LLM.

    Returns:
        str: Corrected JSON string.

    """
    # 1) Normalize curly/smart quotes
    json_str = json_str.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")

    # 2) Insert a missing comma between a completed value and the next key
    json_str = re.sub(
        r'((?:"(?:\\.|[^"\\])*")|(?:-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|true|false|null|\}|\])\s*(?="[^"]+"\s*:)',
        r"\1, ",
        json_str,
    )

    # 3) Remove trailing commas before } or ]
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    return json_str
