"""Source-reduction code to family crosswalk for the analysis scripts.

The single source of truth is ``src/rag_graph/retrieval/activity_families.py``; this module re-exports it under the
names the analysis scripts use (``FAM``, ``W``, ``S``, ``C2F``).
"""

import os
import sys
from typing import Dict, List

sys.path.insert(0, "/Users/josehernandez/Documents/tri_chem_graphrag")

from src.rag_graph.retrieval.activity_families import (  # noqa: E402
    CODE_TO_FAMILY,
    FAMILY_CODES,
    FAMILY_NAMES,
)

FAM: Dict[str, str] = dict(FAMILY_NAMES)
W: Dict[str, List[str]] = {f: [c for c in cs if c.startswith("W")] for f, cs in FAMILY_CODES.items()}
S: Dict[str, List[str]] = {f: [c for c in cs if c.startswith("S")] for f, cs in FAMILY_CODES.items()}
C2F: Dict[str, str] = dict(CODE_TO_FAMILY)
