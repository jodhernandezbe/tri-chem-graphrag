# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""YAML-backed prompt manager for the GraphRAG generation layer.

Loads prompt templates from config/prompts/ and fills template variables
using Python's str.format_map so callers never touch raw YAML keys.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-09
"""

import logging
import pathlib
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

PROMPTS_DIR = pathlib.Path(__file__).parents[3] / "config" / "prompts"


class PromptManager:
    """Loads YAML prompt files and renders templates with caller-supplied variables."""

    def __new__(cls) -> "PromptManager":
        """Create or return the singleton instance."""
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _load(self, prompt_file: str) -> Dict[str, Any]:
        if prompt_file not in self._cache:
            path = PROMPTS_DIR / prompt_file
            try:
                with open(path, "r") as fh:
                    data = yaml.safe_load(fh)
                if not isinstance(data, dict):
                    raise ValueError(f"Prompt file {prompt_file} must be a YAML mapping")
                self._cache[prompt_file] = data
                logger.debug(f"Loaded prompt file: {path}")
            except FileNotFoundError:
                raise FileNotFoundError(f"Prompt file not found: {path}")
        return self._cache[prompt_file]

    def render(
        self,
        prompt_file: str,
        key: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Load a prompt template and fill in the provided variables.

        Args:
            prompt_file (str): YAML file name inside config/prompts/ (e.g. ``"rag_system.yaml"``).
            key (str): Top-level key inside the YAML file (e.g. ``"system_prompt"``).
            variables (Optional[Dict[str, Any]]): Template variables to substitute.

        Returns:
            str: The rendered prompt string.
        """
        data = self._load(prompt_file)
        template: str = data[key]
        return template.format_map(variables or {})

    def version(self, prompt_file: str) -> str:
        """Return the declared version string for a prompt file.

        Args:
            prompt_file (str): YAML file name inside config/prompts/.

        Returns:
            str: Version string, or ``"unknown"`` if not present.
        """
        return str(self._load(prompt_file).get("version", "unknown"))
