# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Fused contextual retrieval: reciprocal-rank fusion of chemical, use, sector and prior channels.

The scoring unit is a reduction record: one (report, chemical, activity code) with the report's declared uses
(Activity names), sector, year, state and annual reduction. Four channels each rank records:

    chemical: the 50 nearest chemicals (configured embedding, plus candidate categories); records inherit the
              chemical's similarity.
    use:      the condition-of-use text matched to the 10 nearest Activity names; records whose report declares
              one inherit the best match score.
    sector:   the sector text matched to the 10 nearest NAICS titles; records in those sectors inherit the score.
    prior:    records ranked by how many chemicals among the visible reports use their code.

Channels for inputs that were not supplied are skipped. Channels are fused with reciprocal rank fusion (k = 60,
weights from ``RETRIEVAL_WEIGHTS``, default equal); records with equal channel scores share a rank. No language model takes part in retrieval or ranking.

Author: Jose D. Hernandez-Betancur
Date: 2026-11-04
"""

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
)

from src.rag_graph.retrieval.switches import (
    resolve_channel_weights,
    resolve_sector_notice_threshold,
)

logger = logging.getLogger(__name__)

RRF_K = 60
CHEMICAL_POOL = 50
USE_TOP = 10
SECTOR_TOP = 10
IDENTICAL_SIMILARITY = 0.99999
CATCH_ALL_PREFIX = "other"
CH_CHEMICAL, CH_USE, CH_SECTOR, CH_PRIOR = "chemical", "use", "sector", "prior"
MATCHED_USE_TIERS = (1, 2)
QUERY_LABEL = "identical structure to the query"
ANALOGUE_LABEL = "ANALOGUE, not the query chemical"
TIER_LABELS: Dict[int, str] = {
    1: "same chemical, stated use",
    2: "analogue, stated use",
    3: "same chemical, other use",
    4: "analogue, stated sector",
    5: "analogue, other context",
    6: "other practice (not an analogue)",
}
CATCH_ALL_HEADING = "Catch-all activity codes (less specific):"
USE_UNMATCHED_NOTICE = (
    "Note: No records matched the stated condition of use ('{cou}') for the query chemical or its analogues; "
    "the records below are ranked by chemical similarity, sector and overall practice, and are not restricted to that use."
)
SECTOR_UNMATCHED_NOTICE = "Note: No records were found in the stated sector ('{sector}'); the records below come from other industries ({industries})."
SECTOR_SEPARATOR = " — "


@dataclass(frozen=True)
class ReductionRecord:
    """One reduction record: a report's documented activity code with its context.

    Attributes:
        report_id (str): Report node id.
        sort_key (str): Stable tie-break key (the report's document control number).
        cas (str): Chemical CAS number.
        chemical (str): Chemical name.
        code (str): Source-reduction activity code.
        name (str): Activity code name.
        uses (Tuple[str, ...]): Activity names (uses) the report declares.
        sector_code (str): Six-digit NAICS code, empty when unknown.
        sector_title (str): NAICS title.
        year (Any): Reporting year.
        state (Any): Facility state.
        annual_reduction (Optional[float]): Estimated annual reduction.
    """

    report_id: str
    sort_key: str
    cas: str
    chemical: str
    code: str
    name: str
    uses: Tuple[str, ...]
    sector_code: str
    sector_title: str
    year: Any
    state: Any
    annual_reduction: Optional[float]


@dataclass(frozen=True)
class FusedOptions:
    """Pre-registered variant settings.

    Attributes:
        prior (bool): Include the prior channel (F1, F3 yes; F2 no).
        demote_catch_all (bool): Move catch-all codes (names starting with "OTHER") to the end (F3).
        weights (Optional[Mapping[str, float]]): Channel weights for the fusion; ``None`` reads ``RETRIEVAL_WEIGHTS``
            from config (default all 1.0). A channel with weight 0 is skipped.
    """

    prior: bool = True
    demote_catch_all: bool = False
    weights: Optional[Mapping[str, float]] = None


F1 = FusedOptions()
F2 = FusedOptions(prior=False)
F3 = FusedOptions(demote_catch_all=True)


@dataclass
class FusedResult:
    """Outcome of one fused retrieval.

    Attributes:
        records (List[Dict[str, Any]]): De-duplicated records, best first, with ``Tier`` and ``Channels``.
        use_supplied (bool): A condition-of-use text was given.
        use_unmatched (bool): It was given but no returned record reached tier 1 or 2.
        retrieved_nodes (List[Dict[str, Any]]): Chemical-channel hits in the legacy ``node``/``score`` shape.
        matched_uses (List[Dict[str, Any]]): Activity names the use channel matched (``name``, ``score``), best first.
        matched_sectors (List[Dict[str, Any]]): NAICS titles the sector channel matched (``code``, ``title``, ``score``,
            ``qualifying``: score at or above the configured threshold), best first.
        sector_supplied (bool): A sector text was given.
        sector_unmatched (bool): It was given but no returned record sits in a qualifying matched sector.
        record_industries (List[str]): Distinct NAICS titles of the returned records, in rank order.
    """

    records: List[Dict[str, Any]]
    use_supplied: bool = False
    use_unmatched: bool = False
    retrieved_nodes: List[Dict[str, Any]] = field(default_factory=list)
    matched_uses: List[Dict[str, Any]] = field(default_factory=list)
    matched_sectors: List[Dict[str, Any]] = field(default_factory=list)
    sector_supplied: bool = False
    sector_unmatched: bool = False
    record_industries: List[str] = field(default_factory=list)


class FusedSource(Protocol):
    """What the fused retriever needs from the graph and vector indexes (read-only)."""

    def chemical_hits(self, smiles: str, exclude_cas: FrozenSet[str]) -> List[Tuple[str, float, str]]:
        """Return ``(cas, similarity, entry-point label)`` for the chemical pool, best first."""
        ...

    def activity_hits(self, text: str, k: int) -> Dict[str, float]:
        """Return Activity name to match score for the ``k`` nearest names."""
        ...

    def sector_hits(self, text: str, k: int) -> Dict[str, float]:
        """Return six-digit NAICS code to match score for the ``k`` nearest titles."""
        ...

    def reduction_records(self, report_split: Optional[str]) -> List[ReductionRecord]:
        """Return all reduction records under a split."""
        ...

    def sector_titles(self, codes: Iterable[str]) -> Dict[str, str]:
        """Return NAICS code to title for codes returned by ``sector_hits`` (missing codes may be absent)."""
        ...


def rank_positions(scores: Mapping[int, float]) -> Dict[int, int]:
    """Rank records by descending score; equal scores share the best rank.

    Args:
        scores (Mapping[int, float]): Record index to channel score.

    Returns:
        Dict[int, int]: Record index to 1-based rank (1 + number of strictly better records).
    """
    ordered = sorted(scores.values(), reverse=True)
    first: Dict[float, int] = {}
    for position, value in enumerate(ordered, 1):
        first.setdefault(value, position)
    return {i: first[s] for i, s in scores.items()}


def reciprocal_rank_fusion(
    channels: Mapping[str, Mapping[int, float]], k: int = RRF_K, weights: Optional[Mapping[str, float]] = None
) -> Dict[int, float]:
    """Fuse channel scores: ``sum over channels of w_c / (k + rank_c)``; weights default to 1.0 (equal).

    Args:
        channels (Mapping[str, Mapping[int, float]]): Channel name to record index to score.
        k (int): RRF constant.
        weights (Optional[Mapping[str, float]]): Channel name to weight; missing channels weigh 1.0, weight 0 skips a channel.

    Returns:
        Dict[int, float]: Record index to fused score (records in no channel are absent).
    """
    fused: Dict[int, float] = {}
    for name, scores in channels.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        if weight <= 0:
            continue
        for index, rank in rank_positions(scores).items():
            fused[index] = fused.get(index, 0.0) + weight / (k + rank)
    return fused


def chemical_channel(records: Sequence[ReductionRecord], similarity: Mapping[str, float]) -> Dict[int, float]:
    """Records inherit the similarity of their chemical.

    Args:
        records (Sequence[ReductionRecord]): Visible records.
        similarity (Mapping[str, float]): CAS to similarity for the chemical pool.

    Returns:
        Dict[int, float]: Record index to similarity.
    """
    return {i: similarity[r.cas] for i, r in enumerate(records) if r.cas in similarity}


def use_channel(records: Sequence[ReductionRecord], use_scores: Mapping[str, float]) -> Dict[int, float]:
    """Records declaring a matched use inherit the best match score.

    Args:
        records (Sequence[ReductionRecord]): Visible records.
        use_scores (Mapping[str, float]): Activity name to match score.

    Returns:
        Dict[int, float]: Record index to best matched-use score.
    """
    out: Dict[int, float] = {}
    for i, r in enumerate(records):
        hits = [use_scores[u] for u in r.uses if u in use_scores]
        if hits:
            out[i] = max(hits)
    return out


def sector_channel(records: Sequence[ReductionRecord], sector_scores: Mapping[str, float]) -> Dict[int, float]:
    """Records in a matched sector inherit its match score.

    Args:
        records (Sequence[ReductionRecord]): Visible records.
        sector_scores (Mapping[str, float]): Six-digit NAICS code to match score.

    Returns:
        Dict[int, float]: Record index to sector score.
    """
    return {i: sector_scores[r.sector_code] for i, r in enumerate(records) if r.sector_code in sector_scores}


def code_popularity(records: Sequence[ReductionRecord]) -> Dict[str, int]:
    """Count, per code, the distinct chemicals whose visible records use it.

    Args:
        records (Sequence[ReductionRecord]): Visible records.

    Returns:
        Dict[str, int]: Code to number of chemicals.
    """
    users: Dict[str, Set[str]] = {}
    for r in records:
        users.setdefault(r.code, set()).add(r.cas)
    return {code: len(cas) for code, cas in users.items()}


def prior_channel(records: Sequence[ReductionRecord]) -> Dict[int, float]:
    """Records score by how common their code is among the visible reports.

    Args:
        records (Sequence[ReductionRecord]): Visible records.

    Returns:
        Dict[int, float]: Record index to the code's chemical count.
    """
    popularity = code_popularity(records)
    return {i: float(popularity[r.code]) for i, r in enumerate(records)}


def assign_tier(similarity: Optional[float], use_hit: bool, sector_hit: bool) -> int:
    """Tier from which channels matched a record.

    1 same chemical and matched use; 2 analogue and matched use; 3 same chemical, other use; 4 analogue in the
    matched sector; 5 analogue, other context; 6 not an analogue (outside the chemical pool): prior, use or sector only.

    Args:
        similarity (Optional[float]): Chemical-channel similarity, ``None`` when the chemical is not in the pool.
        use_hit (bool): The record matched the stated use.
        sector_hit (bool): The record is in a matched sector.

    Returns:
        int: Tier 1 to 6.
    """
    if similarity is None:
        return 6
    same = similarity >= IDENTICAL_SIMILARITY
    if use_hit:
        return 1 if same else 2
    if same:
        return 3
    return 4 if sector_hit else 5


def is_catch_all(record: ReductionRecord) -> bool:
    """Whether a record's code is a catch-all (its name starts with "OTHER").

    Args:
        record (ReductionRecord): Record to test.

    Returns:
        bool: True for catch-all codes.
    """
    return record.name.strip().lower().startswith(CATCH_ALL_PREFIX)


def _format_sector(record: ReductionRecord) -> str:
    return f"{record.sector_code or 'N/A'}{SECTOR_SEPARATOR}{record.sector_title or 'N/A'}"


def to_row(
    record: ReductionRecord, tier: int, channels: List[str], similarity: Optional[float], via: str, score: float
) -> Dict[str, Any]:
    """Render a record in the ``tabulate_records`` shape plus ``Tier`` and ``Channels``.

    Args:
        record (ReductionRecord): The record.
        tier (int): Its tier.
        channels (List[str]): Channels that matched it.
        similarity (Optional[float]): Chemical similarity, ``None`` outside the pool.
        via (str): Candidate entry-point label, if the chemical was reached through a category.
        score (float): Fused score.

    Returns:
        Dict[str, Any]: Row for display, generation and evaluation.
    """
    return {
        "Chemical": record.chemical or "N/A",
        "CAS": record.cas or "N/A",
        "Similarity": round(similarity or 0.0, 4),
        "EntryPoint": via,
        "Sector": _format_sector(record),
        "Reduction Activity": record.name or record.code,
        "ReductionCode": record.code,
        "Year": record.year if record.year is not None else "N/A",
        "State": record.state if record.state is not None else "N/A",
        "Uses": list(record.uses),
        "AnnualReduction": record.annual_reduction,
        "Tier": tier,
        "Channels": channels,
        "FusedScore": round(score, 6),
    }


def dedupe_like_tabulate(rows: Iterable[Dict[str, Any]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Keep the first row per (chemical, activity, year, state), as ``tabulate_records`` does.

    Args:
        rows (Iterable[Dict[str, Any]]): Rows in rank order (may be lazy).
        limit (Optional[int]): Stop after this many unique rows.

    Returns:
        List[Dict[str, Any]]: Unique rows, order preserved.
    """
    seen: Set[Tuple[Any, ...]] = set()
    unique: List[Dict[str, Any]] = []
    for r in rows:
        key = (r["Chemical"], r["Reduction Activity"], r["Year"], r["State"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
            if limit is not None and len(unique) >= limit:
                break
    return unique


def is_catch_all_row(row: Mapping[str, Any]) -> bool:
    """Whether a fused row's activity name marks a catch-all code (starts with "OTHER", any case).

    Args:
        row (Mapping[str, Any]): Row from the fused retriever.

    Returns:
        bool: True for catch-all codes.
    """
    return str(row.get("Reduction Activity", "")).strip().lower().startswith(CATCH_ALL_PREFIX)


def split_catch_all(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Tuple[int, Mapping[str, Any]]], List[Tuple[int, Mapping[str, Any]]]]:
    """Split rows into specific and catch-all groups, each keeping fused order and its original 1-based rank.

    Presentation only: the input order, and so every score, is unchanged.

    Args:
        rows (Sequence[Mapping[str, Any]]): Rows in fused rank order.

    Returns:
        Tuple[List[Tuple[int, Mapping[str, Any]]], List[Tuple[int, Mapping[str, Any]]]]: (specific, catch-all) lists of
            (original rank, row).
    """
    ranked = list(enumerate(rows, start=1))
    specific = [(i, r) for i, r in ranked if not is_catch_all_row(r)]
    return specific, [(i, r) for i, r in ranked if is_catch_all_row(r)]


def fused_context_line(r: Mapping[str, Any], rank: Optional[int] = None) -> str:
    """One generation-context line for a fused record.

    Args:
        r (Mapping[str, Any]): Row from the fused retriever.
        rank (Optional[int]): Original fused rank to show; omitted when ``None``.

    Returns:
        str: Activity, chemical, tier, channels, uses, sector, year and state.
    """
    uses = "; ".join(r.get("Uses") or []) or "not recorded"
    rank_text = f"Rank {rank} | " if rank is not None else ""
    tag = QUERY_LABEL if float(r.get("Similarity", 0.0)) >= IDENTICAL_SIMILARITY else ANALOGUE_LABEL
    return (
        f"  - {rank_text}{r.get('Reduction Activity')} | {r.get('Chemical')} (CAS {r.get('CAS')}, similarity {r.get('Similarity')}, {tag}) | "
        f"Tier {r.get('Tier')}: {TIER_LABELS.get(int(r.get('Tier', 6)))} | Channels {','.join(r.get('Channels', []))} | "
        f"Uses {uses} | Sector {r.get('Sector')} | Year {r.get('Year')} | State {r.get('State')}"
    )


def fused_records_to_context(records: Sequence[Mapping[str, Any]], query_smiles: str = "") -> str:
    """Render fused records as the generation context: specific codes first, then catch-all codes under a heading.

    Each line shows the record's original fused rank; the order of the input is not changed.

    Args:
        records (Sequence[Mapping[str, Any]]): Rows from the fused retriever.
        query_smiles (str): SMILES of the query chemical, for the header.

    Returns:
        str: Context string; empty when there are no records.
    """
    if not records:
        return ""
    lines: List[str] = []
    if query_smiles:
        lines.append(f"QUERY STRUCTURE (SMILES): {query_smiles}")
        lines.append("QUERY CHEMICAL NAME: not provided. Never give the query chemical the name of any record below.")
    lines.append(
        "Source-reduction records ranked by a fusion of chemical similarity, stated use, stated sector and overall practice. "
        "Tier 1 is the same chemical under the stated use; 2 an analogue under the stated use; 3 the same chemical under another "
        "use; 4 an analogue in the stated sector; 5 an analogue in another context; 6 other practice. "
        "A record tagged ANALOGUE is a different chemical, NOT the query chemical; its activity is a transferable candidate."
    )
    lines.append("")
    specific, catch_all = split_catch_all(records)
    lines.extend(fused_context_line(r, i) for i, r in specific)
    if catch_all:
        lines.extend(["", CATCH_ALL_HEADING])
        lines.extend(fused_context_line(r, i) for i, r in catch_all)
    return "\n".join(lines).strip()


def match_entries(
    uses: Mapping[str, float], sectors: Mapping[str, float], titles: Mapping[str, str], threshold: float
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Describe what the use and sector channels matched, best first (ties by name or code).

    Args:
        uses (Mapping[str, float]): Activity name to match score.
        sectors (Mapping[str, float]): Six-digit NAICS code to match score.
        titles (Mapping[str, str]): NAICS code to title.
        threshold (float): Minimum sector similarity that counts as the stated sector being present.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: (matched uses, matched sectors with ``qualifying`` flags).
    """
    used = [{"name": n, "score": round(s, 4)} for n, s in sorted(uses.items(), key=lambda kv: (-kv[1], kv[0]))]
    sect = [
        {"code": c, "title": titles.get(c, ""), "score": round(s, 4), "qualifying": s >= threshold}
        for c, s in sorted(sectors.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return used, sect


def row_sector_code(row: Mapping[str, Any]) -> str:
    """NAICS code of a fused row (the part of ``Sector`` before the separator).

    Args:
        row (Mapping[str, Any]): Row from the fused retriever.

    Returns:
        str: Six-digit code, or ``N/A``.
    """
    return str(row.get("Sector", "")).split(SECTOR_SEPARATOR, 1)[0].strip()


def record_industries(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """Distinct NAICS titles of the returned rows, in rank order.

    Args:
        rows (Sequence[Mapping[str, Any]]): Rows from the fused retriever.

    Returns:
        List[str]: Titles without codes; rows with unknown sector are skipped.
    """
    titles = [str(r.get("Sector", "")).split(SECTOR_SEPARATOR, 1)[-1].strip() for r in rows]
    return list(dict.fromkeys(t for t in titles if t and t != "N/A"))


def sector_is_unmatched(rows: Sequence[Mapping[str, Any]], sectors: Sequence[Mapping[str, Any]]) -> bool:
    """Whether no returned row sits in a matched sector whose similarity reaches the threshold.

    Args:
        rows (Sequence[Mapping[str, Any]]): Rows from the fused retriever.
        sectors (Sequence[Mapping[str, Any]]): Matched sectors from ``match_entries``.

    Returns:
        bool: True when there are rows and none is in a qualifying sector.
    """
    qualifying = {str(s["code"]) for s in sectors if s["qualifying"]}
    return bool(rows) and not any(row_sector_code(r) in qualifying for r in rows)


def matches_header_lines(uses: Sequence[Mapping[str, Any]], sectors: Sequence[Mapping[str, Any]]) -> List[str]:
    """Transcript header lines listing what the use and sector channels matched.

    Args:
        uses (Sequence[Mapping[str, Any]]): Matched uses (``name``, ``score``).
        sectors (Sequence[Mapping[str, Any]]): Matched sectors (``code``, ``title``, ``score``, ``qualifying``).

    Returns:
        List[str]: Two lines; a channel that was not run shows an em dash.
    """
    use_text = "; ".join(f"{u['name']} ({u['score']})" for u in uses) or "—"
    sector_text = "; ".join(f"{s['code']} {s['title']} ({s['score']}{'*' if s['qualifying'] else ''})" for s in sectors) or "—"
    return [f"Matched uses     : {use_text}", f"Matched sectors  : {sector_text}  [* = at or above the sector threshold]"]


class FusedRetriever:
    """Runs the four channels over visible reduction records and fuses them."""

    def __init__(self, source: FusedSource) -> None:
        """Bind the data source.

        Args:
            source (FusedSource): Graph and vector access (read-only).
        """
        self._source = source

    def _channels(
        self,
        records: Sequence[ReductionRecord],
        chemical_smiles: str,
        sector_query: str,
        condition_of_use: str,
        exclude_cas: FrozenSet[str],
    ) -> Tuple[Dict[str, Dict[int, float]], Dict[str, float], Dict[str, str], Dict[str, float], Dict[str, float]]:
        channels: Dict[str, Dict[int, float]] = {}
        use_scores: Dict[str, float] = {}
        sector_scores: Dict[str, float] = {}
        similarity: Dict[str, float] = {}
        via: Dict[str, str] = {}
        if chemical_smiles:
            for cas, score, label in self._source.chemical_hits(chemical_smiles, exclude_cas):
                if cas not in similarity:
                    similarity[cas], via[cas] = score, label
            channels[CH_CHEMICAL] = chemical_channel(records, similarity)
        if condition_of_use.strip():
            use_scores = self._source.activity_hits(condition_of_use, USE_TOP)
            channels[CH_USE] = use_channel(records, use_scores)
        if sector_query.strip():
            sector_scores = self._source.sector_hits(sector_query, SECTOR_TOP)
            channels[CH_SECTOR] = sector_channel(records, sector_scores)
        return channels, similarity, via, use_scores, sector_scores

    def _sector_titles(self, records: Sequence[ReductionRecord], codes: Iterable[str]) -> Dict[str, str]:
        titles = {r.sector_code: r.sector_title for r in records if r.sector_title}
        lookup = getattr(self._source, "sector_titles", None)
        if lookup is not None:
            titles.update({c: t for c, t in lookup(codes).items() if t})
        return titles

    def retrieve(
        self,
        chemical_smiles: str = "",
        sector_query: str = "",
        condition_of_use: str = "",
        report_split: Optional[str] = None,
        options: FusedOptions = F1,
        exclude_report_ids: FrozenSet[str] = frozenset(),
        exclude_cas: FrozenSet[str] = frozenset(),
        limit: int = 10,
    ) -> FusedResult:
        """Fuse the channels and return the top ``limit`` de-duplicated records.

        Args:
            chemical_smiles (str): Query SMILES; the chemical channel is skipped when empty.
            sector_query (str): Sector text; the sector channel is skipped when empty.
            condition_of_use (str): Use text; the use channel is skipped when empty.
            report_split (Optional[str]): Read only reports with this split (``'index'`` when evaluating).
            options (FusedOptions): Variant settings.
            exclude_report_ids (FrozenSet[str]): Reports hidden from every channel (hold-out evaluation).
            exclude_cas (FrozenSet[str]): Chemicals left out of the chemical pool (development cross-validation).
            limit (int): Number of de-duplicated records to return.

        Returns:
            FusedResult: Records with ``Tier`` and ``Channels``, the use and sector notice flags, and the matched uses and sectors.
        """
        records = [r for r in self._source.reduction_records(report_split) if r.report_id not in exclude_report_ids]
        channels, similarity, via, use_scores, sector_scores = self._channels(
            records, chemical_smiles, sector_query, condition_of_use, exclude_cas
        )
        weights = resolve_channel_weights(options.weights)
        if options.prior and weights[CH_PRIOR] > 0:
            channels[CH_PRIOR] = prior_channel(records)
        rows = self._rows(records, channels, similarity, via, options, weights)
        out = dedupe_like_tabulate(rows, limit)
        supplied = bool(condition_of_use.strip())
        unmatched = supplied and not any(r["Tier"] in MATCHED_USE_TIERS for r in out)
        nodes = [{"node": {"cas_number": c}, "content": "", "score": s} for c, s in similarity.items()]
        uses, sectors = match_entries(
            use_scores, sector_scores, self._sector_titles(records, sector_scores), resolve_sector_notice_threshold()
        )
        sector_supplied = bool(sector_query.strip())
        return FusedResult(
            out,
            supplied,
            unmatched,
            nodes,
            uses,
            sectors,
            sector_supplied,
            sector_supplied and sector_is_unmatched(out, sectors),
            record_industries(out),
        )

    @staticmethod
    def _rows(
        records: Sequence[ReductionRecord],
        channels: Mapping[str, Mapping[int, float]],
        similarity: Mapping[str, float],
        via: Mapping[str, str],
        options: FusedOptions,
        weights: Mapping[str, float],
    ) -> Iterator[Dict[str, Any]]:
        fused = reciprocal_rank_fusion(channels, weights=weights)
        order = sorted(
            fused,
            key=lambda i: (
                -fused[i],
                -similarity.get(records[i].cas, 0.0),
                records[i].code,
                records[i].sort_key,
                records[i].report_id,
            ),
        )
        if options.demote_catch_all:
            order = [i for i in order if not is_catch_all(records[i])] + [i for i in order if is_catch_all(records[i])]
        for i in order:
            sim = similarity.get(records[i].cas)
            tier = assign_tier(sim, i in channels.get(CH_USE, {}), i in channels.get(CH_SECTOR, {}))
            names = [c for c in (CH_CHEMICAL, CH_USE, CH_SECTOR, CH_PRIOR) if i in channels.get(c, {})]
            yield to_row(records[i], tier, names, sim, via.get(records[i].cas, ""), fused[i])
