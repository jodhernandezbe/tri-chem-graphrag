# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Optional retrieval behaviours: names, defaults, config lookup, and activity-code IDF.

Four switches change how the recommender ranks and selects evidence. Each is read from
``RAG_GRAPH.RETRIEVAL`` in ``config/config.yaml`` with the shipped behaviour as the default
(``retrieval`` ships as ``fused``; the other three apply to ``legacy`` only), and can be overridden per call:

    ranking:    ``similarity`` | ``specificity`` | ``consensus_specificity``
    cou_filter: ``hard`` | ``soft``
    analogues:  ``molecular`` | ``context``
    retrieval:  ``fused`` | ``legacy`` (fused ignores the other three; see ``fused_retriever``)

Author: Jose D. Hernandez-Betancur
Date: 2026-11-04
"""

import logging
import math
from typing import Dict, FrozenSet, Iterable, Mapping, Optional

from src.config_manager import ConfigManager

logger = logging.getLogger(__name__)

RANKING_SIMILARITY = "similarity"
RANKING_SPECIFICITY = "specificity"
RANKING_CONSENSUS = "consensus_specificity"
COU_HARD = "hard"
COU_SOFT = "soft"
ANALOGUES_MOLECULAR = "molecular"
ANALOGUES_CONTEXT = "context"
RETRIEVAL_LEGACY = "legacy"
RETRIEVAL_FUSED = "fused"

# switch name -> (config key, shipped default, allowed values)
SWITCHES: Dict[str, tuple] = {
    "ranking": ("RANKING", RANKING_SIMILARITY, (RANKING_SIMILARITY, RANKING_SPECIFICITY, RANKING_CONSENSUS)),
    "cou_filter": ("COU_FILTER", COU_HARD, (COU_HARD, COU_SOFT)),
    "analogues": ("ANALOGUES", ANALOGUES_MOLECULAR, (ANALOGUES_MOLECULAR, ANALOGUES_CONTEXT)),
    "retrieval": ("RETRIEVAL", RETRIEVAL_FUSED, (RETRIEVAL_FUSED, RETRIEVAL_LEGACY)),
}

# Fusion channels and the weight config key (weights default to 1.0 each: equal-weight reciprocal rank fusion)
FUSION_CHANNELS = ("chemical", "use", "sector", "prior")
WEIGHTS_KEY = "RETRIEVAL_WEIGHTS"
DEFAULT_CHANNEL_WEIGHT = 1.0

# Sector notice (fused): a matched NAICS title counts as the stated sector only at or above this similarity. One fixed value,
# not tuned: 0.85 sits in the gap between the best match for a sector absent from the graph (dry cleaning: 0.80, a laundry
# equipment maker) and the best match for sectors clearly present (0.8575 to 0.8664 for PVC, resin and synthetic rubber).
SECTOR_NOTICE_KEY = "SECTOR_NOTICE_MIN_SIMILARITY"
DEFAULT_SECTOR_NOTICE_THRESHOLD = 0.85

# Size of the molecular candidate pool that ``analogues=context`` re-ranks.
CONTEXT_POOL_SIZE = 50


def resolve_switch(name: str, override: Optional[str] = None) -> str:
    """Return a switch value: the per-call override, else the config value, else the shipped default.

    Args:
        name (str): One of ``ranking``, ``cou_filter``, ``analogues``, ``retrieval``.
        override (Optional[str]): Per-call value; ``None`` defers to config.

    Returns:
        str: The validated switch value.

    Raises:
        ValueError: If the switch name is unknown or the value is not allowed.
    """
    if name not in SWITCHES:
        raise ValueError(f"Unknown retrieval switch: '{name}'")
    key, default, allowed = SWITCHES[name]
    value = override
    if value is None:
        value = ConfigManager().config.get("RAG_GRAPH", {}).get("RETRIEVAL", {}).get(key, default)
    if value not in allowed:
        logger.error(f"Invalid value {value!r} for switch '{name}'; allowed: {allowed}")
        raise ValueError(f"Invalid value {value!r} for switch '{name}'; allowed: {allowed}")
    return str(value)


def parse_channel_weights(raw: Optional[Mapping[str, float]]) -> Dict[str, float]:
    """Validate channel weights and fill the unspecified channels with 1.0.

    Args:
        raw (Optional[Mapping[str, float]]): Channel name to weight; ``None`` or empty means all 1.0.

    Returns:
        Dict[str, float]: A weight for each of ``chemical``, ``use``, ``sector`` and ``prior``.

    Raises:
        ValueError: If a channel name is unknown or a weight is negative or not a number.
    """
    weights = {name: DEFAULT_CHANNEL_WEIGHT for name in FUSION_CHANNELS}
    for name, value in (raw or {}).items():
        if name not in weights:
            logger.error(f"Unknown fusion channel {name!r}; allowed: {FUSION_CHANNELS}")
            raise ValueError(f"Unknown fusion channel {name!r}; allowed: {FUSION_CHANNELS}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or math.isnan(value):
            logger.error(f"Invalid weight {value!r} for channel {name!r}: must be a number >= 0")
            raise ValueError(f"Invalid weight {value!r} for channel {name!r}: must be a number >= 0")
        weights[name] = float(value)
    return weights


def resolve_channel_weights(override: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
    """Return fusion channel weights: the per-call override, else ``RAG_GRAPH.RETRIEVAL.RETRIEVAL_WEIGHTS``, else all 1.0.

    Args:
        override (Optional[Mapping[str, float]]): Per-call weights; ``None`` defers to config.

    Returns:
        Dict[str, float]: A validated weight for each channel.
    """
    raw = override
    if raw is None:
        raw = ConfigManager().config.get("RAG_GRAPH", {}).get("RETRIEVAL", {}).get(WEIGHTS_KEY)
    return parse_channel_weights(raw)


def compute_code_idf(profiles: Mapping[str, Iterable[str]]) -> Dict[str, float]:
    """Compute IDF(c) = log(|C| / |{q in C : c in A_q}|) over per-chemical activity-code sets.

    Args:
        profiles (Mapping[str, Iterable[str]]): Chemical CAS to the activity codes it reports. Pass the
            index-split profiles when evaluating, so held-out reports cannot influence the weights.

    Returns:
        Dict[str, float]: Activity code to IDF; a code reported by every chemical gets 0.
    """
    sets: Dict[str, FrozenSet[str]] = {cas: frozenset(codes) for cas, codes in profiles.items()}
    doc_freq: Dict[str, int] = {}
    for codes in sets.values():
        for code in codes:
            doc_freq[code] = doc_freq.get(code, 0) + 1
    return {code: math.log(len(sets) / freq) for code, freq in doc_freq.items()}


def idf_from_counts(chemicals_per_code: Mapping[str, int], n_chemicals: int) -> Dict[str, float]:
    """Compute IDF(c) = log(|C| / df(c)) from per-code chemical counts.

    Args:
        chemicals_per_code (Mapping[str, int]): Activity code to the number of chemicals reporting it.
        n_chemicals (int): |C|, the number of chemicals reporting at least one code.

    Returns:
        Dict[str, float]: Activity code to IDF.
    """
    return {code: math.log(n_chemicals / freq) for code, freq in chemicals_per_code.items() if freq > 0 and n_chemicals > 0}


def resolve_sector_notice_threshold() -> float:
    """Return the sector-notice similarity threshold from ``RAG_GRAPH.RETRIEVAL.SECTOR_NOTICE_MIN_SIMILARITY``.

    Returns:
        float: Threshold in (0, 1]; the shipped default is 0.85.

    Raises:
        ValueError: If the configured value is not a number in (0, 1].
    """
    value = (
        ConfigManager().config.get("RAG_GRAPH", {}).get("RETRIEVAL", {}).get(SECTOR_NOTICE_KEY, DEFAULT_SECTOR_NOTICE_THRESHOLD)
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
        logger.error(f"Invalid {SECTOR_NOTICE_KEY} {value!r}: must be a number in (0, 1]")
        raise ValueError(f"Invalid {SECTOR_NOTICE_KEY} {value!r}: must be a number in (0, 1]")
    return float(value)
