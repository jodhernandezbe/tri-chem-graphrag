# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Read-only probability and efficiency estimates over fused retrieval records.

Given the ranked rows of the fused retriever, this layer reports, for each activity code and each activity family:
a probability, and a distribution over the reported reduction bands (R1 100 %, R2 at least 50 %, R3 at least 25 %).

    probability: the share of total fused score (reciprocal-rank-fusion mass) carried by the records of that code or
                 family. It sums to 1 over codes and over families. It is an evidence share, not a calibrated
                 probability; calibration is a separate, pre-registered step (``audit_logs/PREREG_value_demonstration.md``).
    bands:       among the records of that code or family that report a reduction, the score-weighted share in each band.

The layer never reorders, drops or edits rows, and calls no model, database or network.

Author: Jose D. Hernandez-Betancur
Date: 2026-11-04
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.rag_graph.retrieval.activity_families import BAND_NAMES, band_of, family_of
from src.rag_graph.retrieval.fused_retriever import is_catch_all_row


@dataclass(frozen=True)
class GroupEstimate:
    """Estimate for one code or one family.

    Attributes:
        key (str): Activity code or family id.
        name (str): Activity name (codes) or family id (families).
        family (str): Family of the code, or the family id itself.
        probability (float): Share of total fused score.
        n_records (int): Records supporting the group.
        n_with_band (int): Of those, records that report a reduction.
        band_probabilities (Dict[str, float]): Score-weighted share per band among records with a band.
        catch_all (bool): True for catch-all codes (names starting with "OTHER").
    """

    key: str
    name: str
    family: str
    probability: float
    n_records: int
    n_with_band: int
    band_probabilities: Dict[str, float]
    catch_all: bool = False


@dataclass(frozen=True)
class P2Estimate:
    """Code-level and family-level estimates for one query, each sorted by descending probability.

    Attributes:
        codes (List[GroupEstimate]): One entry per activity code.
        families (List[GroupEstimate]): One entry per family (``F0`` collects unknown codes).
        n_records (int): Rows used.
    """

    codes: List[GroupEstimate] = field(default_factory=list)
    families: List[GroupEstimate] = field(default_factory=list)
    n_records: int = 0


def _weights(rows: Sequence[Mapping[str, Any]]) -> List[float]:
    raw = [max(float(r.get("FusedScore") or 0.0), 0.0) for r in rows]
    return raw if sum(raw) > 0 else [1.0] * len(rows)


def _band_shares(rows: Sequence[Mapping[str, Any]], weights: Sequence[float]) -> Dict[str, float]:
    mass = {name: 0.0 for name in BAND_NAMES}
    for row, w in zip(rows, weights):
        band = band_of(row.get("AnnualReduction"))
        if band in mass:
            mass[str(band)] += w
    total = sum(mass.values())
    return {name: (m / total if total else 0.0) for name, m in mass.items()}


def _group(key: str, rows: Sequence[Mapping[str, Any]], weights: Sequence[float], total: float, family: str) -> GroupEstimate:
    names = [str(r.get("Reduction Activity", "")) for r in rows]
    with_band = [i for i, r in enumerate(rows) if band_of(r.get("AnnualReduction")) in BAND_NAMES]
    return GroupEstimate(
        key=key,
        name=names[0] if names else key,
        family=family,
        probability=sum(weights) / total,
        n_records=len(rows),
        n_with_band=len(with_band),
        band_probabilities=_band_shares(rows, weights),
        catch_all=bool(rows) and is_catch_all_row(rows[0]),
    )


def _estimates(
    buckets: Mapping[str, List[int]], rows: Sequence[Mapping[str, Any]], w: Sequence[float], by_family: bool
) -> List[GroupEstimate]:
    total = sum(w)
    out = []
    for key, idx in buckets.items():
        family = key if by_family else family_of(key)
        out.append(_group(key, [rows[i] for i in idx], [w[i] for i in idx], total, family))
    return sorted(out, key=lambda g: (-g.probability, g.key))


def estimate_p2(rows: Sequence[Mapping[str, Any]], limit: Optional[int] = None) -> P2Estimate:
    """Estimate code and family probabilities and reduction-band distributions from fused rows.

    Args:
        rows (Sequence[Mapping[str, Any]]): Rows from the fused retriever (``ReductionCode``, ``FusedScore``,
            ``AnnualReduction``); not modified.
        limit (Optional[int]): Use only the first ``limit`` rows (the fused order); ``None`` uses all.

    Returns:
        P2Estimate: Estimates sorted by descending probability; empty when there are no rows.
    """
    used = list(rows if limit is None else rows[:limit])
    if not used:
        return P2Estimate()
    w = _weights(used)
    codes: Dict[str, List[int]] = {}
    families: Dict[str, List[int]] = {}
    for i, row in enumerate(used):
        code = str(row.get("ReductionCode", "")).strip().upper()
        codes.setdefault(code, []).append(i)
        families.setdefault(family_of(code), []).append(i)
    return P2Estimate(_estimates(codes, used, w, False), _estimates(families, used, w, True), len(used))
