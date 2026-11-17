# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Sector and condition-of-use context similarity between chemicals (manuscript Section 4.3).

Each chemical has two count profiles over its reports: 4-digit NAICS subsectors and conditions of
use (Activity names). A profile is weighted as ``log1p(count) * idf(key)`` with
``idf = log((n + 1) / (df + 1)) + 1`` over the n chemicals, L2-normalised, and compared by cosine.
Context similarity is the mean of the sector and condition-of-use cosines, as in
``analysis/surrogate_and_p2_likelihood/anal_kg.py`` (``CTX = (SN + SU) / 2``).

Author: Jose D. Hernandez-Betancur
Date: 2026-11-04
"""

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

Counts = Mapping[str, float]
Profile = Tuple[Counts, Counts]

NAICS_DIGITS = 4


def _key_idf(per_chemical: Mapping[str, Counts]) -> Dict[str, float]:
    n_chem = len(per_chemical)
    doc_freq: Dict[str, int] = {}
    for counts in per_chemical.values():
        for key in counts:
            doc_freq[key] = doc_freq.get(key, 0) + 1
    return {key: math.log((n_chem + 1) / (freq + 1)) + 1.0 for key, freq in doc_freq.items()}


def _cosine(a: Counts, b: Counts, idf: Mapping[str, float], idf_unseen: float) -> float:
    """Cosine of two ``log1p(count) * idf`` vectors; 0 when either is empty."""
    va = {k: math.log1p(v) * idf.get(k, idf_unseen) for k, v in a.items() if v > 0}
    vb = {k: math.log1p(v) * idf.get(k, idf_unseen) for k, v in b.items() if v > 0}
    na, nb = math.sqrt(sum(x * x for x in va.values())), math.sqrt(sum(x * x for x in vb.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * vb[k] for k, x in va.items() if k in vb) / (na * nb)


class ContextProfiles:
    """Per-chemical sector and condition-of-use profiles with Section 4.3 context similarity."""

    def __init__(self, naics: Mapping[str, Counts], cou: Mapping[str, Counts]) -> None:
        """Build the profile store and its corpus IDFs.

        Args:
            naics (Mapping[str, Counts]): CAS to 4-digit NAICS report counts.
            cou (Mapping[str, Counts]): CAS to condition-of-use (Activity name) report counts.
        """
        self._naics, self._cou = naics, cou
        self._idf_naics, self._idf_cou = _key_idf(naics), _key_idf(cou)
        self._unseen_naics = math.log(len(naics) + 1) + 1.0
        self._unseen_cou = math.log(len(cou) + 1) + 1.0

    @classmethod
    def from_rows(cls, naics_rows: List[Dict[str, Any]], cou_rows: List[Dict[str, Any]]) -> "ContextProfiles":
        """Build a store from aggregate graph rows.

        Args:
            naics_rows (List[Dict[str, Any]]): Rows with ``cas``, ``key`` (NAICS prefix) and ``n``.
            cou_rows (List[Dict[str, Any]]): Rows with ``cas``, ``key`` (Activity name) and ``n``.

        Returns:
            ContextProfiles: The populated store.
        """
        naics: Dict[str, Dict[str, float]] = {}
        cou: Dict[str, Dict[str, float]] = {}
        for rows, target in ((naics_rows, naics), (cou_rows, cou)):
            for row in rows:
                if row.get("cas") and row.get("key"):
                    target.setdefault(row["cas"], {})[str(row["key"])] = float(row["n"])
        return cls(naics, cou)

    def profile(self, cas: str) -> Profile:
        """Return a chemical's (sector counts, condition-of-use counts); empty counts if unknown.

        Args:
            cas (str): CAS number.

        Returns:
            Profile: The two count mappings.
        """
        return self._naics.get(cas, {}), self._cou.get(cas, {})

    def similarity(self, a: Profile, b: Profile) -> float:
        """Context similarity: mean of the sector and condition-of-use cosines.

        Args:
            a (Profile): First profile.
            b (Profile): Second profile.

        Returns:
            float: Similarity in [0, 1].
        """
        sector = _cosine(a[0], b[0], self._idf_naics, self._unseen_naics)
        use = _cosine(a[1], b[1], self._idf_cou, self._unseen_cou)
        return (sector + use) / 2.0


def is_empty(profile: Profile) -> bool:
    """Return True when a profile has neither sector nor condition-of-use evidence.

    Args:
        profile (Profile): Profile to test.

    Returns:
        bool: Whether both count mappings are empty.
    """
    return not profile[0] and not profile[1]


def rerank_by_context(
    hits: List[Dict[str, Any]],
    store: ContextProfiles,
    query_profile: Profile,
    top_k: int,
    overrides: Optional[Mapping[str, Profile]] = None,
) -> List[Dict[str, Any]]:
    """Re-rank a molecular candidate pool by context similarity to the query and keep ``top_k``.

    Args:
        hits (List[Dict[str, Any]]): Vector hits (``node``, ``content``, ``score``), best molecular match first.
        store (ContextProfiles): Profiles of the candidates, built from the reports visible to the caller.
        query_profile (Profile): The query chemical's profile from visible reports only.
        top_k (int): Number of analogues to keep.
        overrides (Optional[Mapping[str, Profile]]): Profiles that replace the store's for given CAS numbers
            (used to hide held-out reports of the query chemical when it appears among its own analogues).

    Returns:
        List[Dict[str, Any]]: If the query has no profile, ``hits[:top_k]`` unchanged (molecular order).
            Otherwise the ``top_k`` hits by descending context similarity (ties by molecular score), each a
            copy carrying ``context_similarity``.
    """
    if is_empty(query_profile):
        return hits[:top_k]
    scored: List[Dict[str, Any]] = []
    for hit in hits:
        cas = hit.get("node", {}).get("cas_number")
        cand = (overrides or {}).get(cas) if cas else None
        cand = cand if cand is not None else store.profile(cas or "")
        scored.append({**hit, "context_similarity": store.similarity(query_profile, cand)})
    scored.sort(key=lambda h: (-h["context_similarity"], -float(h.get("score", 0.0))))
    return scored[:top_k]
