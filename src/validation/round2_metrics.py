# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Pure functions for the literature round 2 test (pre-registered in ``manuscript/audit_logs/PREREG_literature_round2.md``).

Definitions (prereg section 10):
    * The tool list is the layer's codes with probability above 0, in descending probability (ties by ascending code string).
    * The MostCommon list orders codes by the number of chemicals using them, descending (ties by ascending code string).
    * Catch-all codes (activity name starting with "OTHER", any case) are moved after the specific codes in both lists, as in the
      shipped answer, keeping the order inside each group.
    * Rank is 1-based. A code absent from the tool list is "not in list": it counts as a loss when MostCommon has it within the first
      ``BUDGET`` positions, and as a tie when neither list has it within those positions.

Author: Jose D. Hernandez-Betancur
Date: 2026-11-04
"""

import hashlib
import json
import logging
import os
from math import comb
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

BUDGET = 50
TOP_K = 10
WIN, LOSS, TIE = "win", "loss", "tie"
CATCH_ALL_PREFIX = "other"


def is_catch_all(name: str) -> bool:
    """Whether an activity name marks a catch-all code (starts with "OTHER", any case).

    Args:
        name (str): Activity name.

    Returns:
        bool: True for catch-all codes.
    """
    return name.strip().lower().startswith(CATCH_ALL_PREFIX)


def grouped_order(order: Sequence[str], names: Mapping[str, str]) -> List[str]:
    """Move catch-all codes after the specific codes, keeping the order inside each group.

    Args:
        order (Sequence[str]): Codes in rank order.
        names (Mapping[str, str]): Code to activity name.

    Returns:
        List[str]: Reordered codes.
    """
    catch = [c for c in order if is_catch_all(names.get(c, ""))]
    return [c for c in order if c not in catch] + catch


def tool_order(probs: Mapping[str, float], universe: Sequence[str]) -> List[str]:
    """Codes with probability above 0 in descending probability, ties by ascending code string.

    Args:
        probs (Mapping[str, float]): Layer probability per code.
        universe (Sequence[str]): Code vocabulary.

    Returns:
        List[str]: The tool list before grouping.
    """
    return [c for c in sorted(universe, key=lambda u: (-probs.get(u, 0.0), u)) if probs.get(c, 0.0) > 0]


def mostcommon_order(chemicals_per_code: Mapping[str, int], universe: Sequence[str]) -> List[str]:
    """Codes by descending number of chemicals using them; ties by ascending code string.

    Args:
        chemicals_per_code (Mapping[str, int]): Number of chemicals per code.
        universe (Sequence[str]): Code vocabulary.

    Returns:
        List[str]: The MostCommon list before grouping.
    """
    return sorted(universe, key=lambda u: (-chemicals_per_code.get(u, 0), u))


def rank_of(order: Sequence[str], code: str) -> Optional[int]:
    """1-based rank of a code, ``None`` when it is not in the list.

    Args:
        order (Sequence[str]): Ranked codes.
        code (str): Code to find.

    Returns:
        Optional[int]: Rank.
    """
    return list(order).index(code) + 1 if code in order else None


def compare_ranks(tool_rank: Optional[int], mc_rank: Optional[int], budget: int = BUDGET) -> str:
    """Outcome for one paper: win, loss or tie of the tool against MostCommon.

    Args:
        tool_rank (Optional[int]): Tool rank, ``None`` when not in the list.
        mc_rank (Optional[int]): MostCommon rank, ``None`` when not in the list.
        budget (int): Positions of each list that count (the tool list is the 50-record layer output).

    Returns:
        str: ``win`` (tool ranks better), ``loss`` or ``tie``.
    """
    mc = mc_rank if mc_rank is not None and mc_rank <= budget else None
    if tool_rank is None:
        return LOSS if mc is not None else TIE
    if mc is None:
        return WIN
    return WIN if tool_rank < mc else LOSS if tool_rank > mc else TIE


def count_outcomes(outcomes: Sequence[str]) -> Dict[str, int]:
    """Count wins, losses and ties.

    Args:
        outcomes (Sequence[str]): Outcomes from ``compare_ranks``.

    Returns:
        Dict[str, int]: ``wins``, ``losses`` and ``ties``.
    """
    return {"wins": outcomes.count(WIN), "losses": outcomes.count(LOSS), "ties": outcomes.count(TIE)}


def exact_sign_p(wins: int, losses: int) -> Optional[float]:
    """One-sided exact sign test p-value (ties excluded).

    Args:
        wins (int): Number of wins.
        losses (int): Number of losses.

    Returns:
        Optional[float]: P(X >= wins) for X ~ Binomial(wins + losses, 0.5), or ``None`` when there are no untied papers.
    """
    return binomial_upper_p(wins, wins + losses, 0.5) if wins + losses else None


def binomial_upper_p(k: int, n: int, p0: float) -> float:
    """Exact one-sided binomial p-value P(X >= k).

    Args:
        k (int): Observed successes.
        n (int): Trials.
        p0 (float): Success probability under the null.

    Returns:
        float: Upper-tail probability.
    """
    return float(sum(comb(n, i) * p0**i * (1 - p0) ** (n - i) for i in range(k, n + 1)))


def chance_p0(n_codes: int, k: int = TOP_K) -> float:
    """Chance of a given code being in a random top-k list: ``k / N`` with N the code vocabulary at run time.

    Args:
        n_codes (int): Number of distinct codes in the vocabulary.
        k (int): List length.

    Returns:
        float: Probability.
    """
    return k / n_codes


def paper_row(
    code: str, probs: Mapping[str, float], universe: Sequence[str], counts: Mapping[str, int], names: Mapping[str, str]
) -> Dict[str, Any]:
    """Ranks and outcome of one paper under the pre-registered definitions.

    Args:
        code (str): The paper's confirmed code.
        probs (Mapping[str, float]): Layer probability per code for the paper's query.
        universe (Sequence[str]): Code vocabulary.
        counts (Mapping[str, int]): Chemicals per code over all reports.
        names (Mapping[str, str]): Code to activity name.

    Returns:
        Dict[str, Any]: Tool and MostCommon ranks, top-k hits, and the outcome.
    """
    tool = rank_of(grouped_order(tool_order(probs, universe), names), code)
    mc = rank_of(grouped_order(mostcommon_order(counts, universe), names), code)
    return {
        "code": code,
        "tool_rank": tool,
        "mostcommon_rank": mc,
        "tool_hit_top10": tool is not None and tool <= TOP_K,
        "mostcommon_hit_top10": mc is not None and mc <= TOP_K,
        "outcome": compare_ranks(tool, mc),
    }


def summarize(rows: Sequence[Mapping[str, Any]], n_codes: int) -> Dict[str, Any]:
    """Primary sign test and top-10 hit tests for a set of paper rows.

    Args:
        rows (Sequence[Mapping[str, Any]]): Output of ``paper_row`` per paper.
        n_codes (int): Code vocabulary size at run time (recorded; sets the chance level).

    Returns:
        Dict[str, Any]: Counts, exact sign p, tool and MostCommon hit counts and their binomial p-values against chance.
    """
    c = count_outcomes([r["outcome"] for r in rows])
    n, p0 = len(rows), chance_p0(n_codes)
    hits, mc_hits = sum(bool(r["tool_hit_top10"]) for r in rows), sum(bool(r["mostcommon_hit_top10"]) for r in rows)
    return {
        "n_papers": n,
        **c,
        "sign_p_one_sided": exact_sign_p(c["wins"], c["losses"]),
        "n_codes": n_codes,
        "chance_p0": p0,
        "tool_hits_top10": hits,
        "tool_hits_p_vs_chance": binomial_upper_p(hits, n, p0) if n else None,
        "mostcommon_hits_top10": mc_hits,
        "mostcommon_hits_p_vs_chance": binomial_upper_p(mc_hits, n, p0) if n else None,
    }


def sha256_file(path: str) -> str:
    """SHA-256 of a file's bytes.

    Args:
        path (str): File path.

    Returns:
        str: Hex digest.
    """
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def check_freeze(manifest_path: str, papers_path: str) -> Tuple[Dict[str, Any], str]:
    """Refuse to run unless a freeze manifest exists and records the hash of the paper file about to be used.

    Args:
        manifest_path (str): Path of ``freeze_manifest.json``; it must hold ``papers_sha256``.
        papers_path (str): Path of the frozen paper file.

    Returns:
        Tuple[Dict[str, Any], str]: The manifest and the paper file's hash.

    Raises:
        FileNotFoundError: The manifest or the paper file is missing.
        ValueError: The manifest has no hash or it does not match the file.
    """
    for path in (manifest_path, papers_path):
        if not os.path.exists(path):
            logger.error(f"round 2 freeze: missing {path}")
            raise FileNotFoundError(path)
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    digest = sha256_file(papers_path)
    if manifest.get("papers_sha256") != digest:
        logger.error("round 2 freeze: paper file hash does not match the manifest")
        raise ValueError("paper file hash does not match freeze_manifest.json")
    return manifest, digest
