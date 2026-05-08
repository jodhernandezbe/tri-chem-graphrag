# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Year processing tracker for the TRI Knowledge Graph pipeline.

Persists a JSON record of which pipeline steps (extract, transform, load)
have been completed for each reporting year, preventing redundant processing.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-07
"""

import json
import logging
import pathlib
from datetime import datetime
from typing import Dict, List

CURRENT_DIR = pathlib.Path(__file__).parent
TRACKER_PATH = CURRENT_DIR.parent.parent / "ancillary" / "tri" / "processed_years.json"

logger = logging.getLogger(__name__)


class YearTracker:
    """Tracks completed pipeline steps per reporting year using a JSON file."""

    def __init__(self) -> None:
        self._data: Dict = self._load()

    def _load(self) -> Dict:
        if not TRACKER_PATH.exists():
            return {"processed": {}}
        with open(TRACKER_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _save(self) -> None:
        TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TRACKER_PATH, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2)

    def is_step_done(self, year: str, step: str) -> bool:
        """Return True if the given step was already completed for the year.

        Args:
            year (str): Reporting year, e.g. "2022".
            step (str): Pipeline step name: "extract", "transform", or "load".

        Returns:
            bool: True when the step is recorded as completed.
        """
        return step in self._data["processed"].get(year, {})

    def mark_step_done(self, year: str, step: str) -> None:
        """Record a completed step for the given year and persist to disk.

        Args:
            year (str): Reporting year, e.g. "2022".
            step (str): Pipeline step name: "extract", "transform", or "load".
        """
        self._data["processed"].setdefault(year, {})[step] = datetime.now().isoformat(timespec="seconds")
        self._save()
        logger.info(f"Marked '{step}' as done for year {year}")

    def pending_steps(self, year: str, steps: List[str]) -> List[str]:
        """Return the subset of steps not yet completed for the given year.

        Args:
            year (str): Reporting year, e.g. "2022".
            steps (List[str]): Ordered list of steps to check.

        Returns:
            List[str]: Steps still pending, preserving the original order.
        """
        return [s for s in steps if not self.is_step_done(year, s)]
