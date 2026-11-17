# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""TRI source-reduction activity families (F1 to F5) over both code series, and reduction bands.

The W series (pre-2005 reporting) and the S series (current) are mapped to the same five families, as in the
manuscript's family-level scoring. A copy of ``analysis/surrogate_and_p2_likelihood/xwalk.py`` (a test keeps them equal).
Reduction bands are the lower bounds that facilities report: R1 is 100 %, R2 at least 50 %, R3 at least 25 %.

Author: Jose D. Hernandez-Betancur
Date: 2026-11-04
"""

from typing import Dict, List, Optional, Tuple

UNKNOWN_FAMILY = "F0"

FAMILY_NAMES: Dict[str, str] = {
    "F1": "Material substitution/modification",
    "F2": "Product modification",
    "F3": "Process/equipment modification",
    "F4": "Inventory, material management, spill/leak prevention",
    "F5": "Operating practices and training",
}

_W_AND_S_CODES: Dict[str, str] = {
    "F1": "W41 W42 W43 W49 W53 W56 W61 W73 S01 S02 S03 S04 S05 S06",
    "F2": "W81 W82 W83 W84 W89 S11 S12 S13 S14",
    "F3": "W35 W50 W51 W52 W57 W58 W59 W60 W63 W64 W65 W66 W67 W68 W71 W72 W74 W75 W78 S21 S22 S23 S24 S25",
    "F4": "W21 W22 W23 W24 W25 W29 W31 W32 W33 W36 W39 W54 W55 S31 S32 S33 S34 S35",
    "F5": "W13 W14 W15 W19 S41 S42 S43 S44",
}
FAMILY_CODES: Dict[str, List[str]] = {fam: codes.split() for fam, codes in _W_AND_S_CODES.items()}

CODE_TO_FAMILY: Dict[str, str] = {code: fam for fam, codes in FAMILY_CODES.items() for code in codes}

# (lower bound in percent, band name), strongest first
BAND_THRESHOLDS: Tuple[Tuple[float, str], ...] = ((100.0, "R1"), (50.0, "R2"), (25.0, "R3"))
BAND_NAMES: Tuple[str, ...] = tuple(name for _, name in BAND_THRESHOLDS)


def family_of(code: str) -> str:
    """Return the activity family of a source-reduction code.

    Args:
        code (str): Activity code such as ``W58`` or ``S11``.

    Returns:
        str: ``F1`` to ``F5``, or ``F0`` when the code is unknown.
    """
    return CODE_TO_FAMILY.get(str(code).strip().upper(), UNKNOWN_FAMILY)


def band_of(percent: Optional[float]) -> Optional[str]:
    """Map a reported reduction percentage to its band.

    Args:
        percent (Optional[float]): Reported reduction in percent.

    Returns:
        Optional[str]: ``R1``, ``R2``, ``R3``, ``below`` (under 25 %), or ``None`` when ``percent`` is missing or not finite.
    """
    if percent is None or percent != percent or percent in (float("inf"), float("-inf")):
        return None
    return next((name for lower, name in BAND_THRESHOLDS if percent >= lower), "below")
