# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""LangGraph-based GraphRAG pipeline for TRI chemical knowledge graph queries.

Pipeline stages:
    extract       → LLM structured extraction (SMILES + sector + condition of use)
                    with deterministic RDKit SMILES validation, then conditional routing
    plan          → LLM extracts chemical identifier and sector from NL query
    retrieve      → parallel vector ANN search (chemical + sector) via Neo4jVector
    build_context → format subgraphs into a prompt-ready context string
    generate      → answer via ChatOllama + YAML-backed prompts

Author: Jose D. Hernandez-Betancur
Date: 2026-05-09
"""

import logging
import os
import re
from typing import Any, Dict, List, Mapping, Optional, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field
from rdkit import Chem
from typing_extensions import TypedDict

from src.config_manager import ConfigManager
from src.rag_graph.helpers import clean_llm_json, fix_json_formatting
from src.rag_graph.retrieval import RetrieverFactory
from src.rag_graph.retrieval.fused_retriever import (
    CATCH_ALL_HEADING,
    SECTOR_UNMATCHED_NOTICE,
    USE_UNMATCHED_NOTICE,
    fused_records_to_context,
    split_catch_all,
)
from src.rag_graph.retrieval.switches import (
    RANKING_SIMILARITY,
    RANKING_SPECIFICITY,
    RETRIEVAL_FUSED,
    resolve_switch,
)

from .prompt_manager import PromptManager

logger = logging.getLogger(__name__)


IDENTICAL_SIMILARITY = 0.99999
QUERY_TAG = " — IDENTICAL STRUCTURE TO THE QUERY"
ANALOGUE_TAG = " — ANALOGUE (not the query chemical)"
SECTOR_FALLBACK_NOTICE = (
    "Note: No qualifying records were found in the stated sector ('{sector}'); " "the records below come from other industries."
)
COU_FALLBACK_NOTICE = (
    "Note: No qualifying records were found under the stated condition of use ('{cou}'); "
    "the records below are not restricted by condition of use."
)


def _round_robin_by_chemical(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reorder similarity-sorted records by cycling one record per chemical.

    Args:
        rows (List[Dict[str, Any]]): Records pre-sorted by descending similarity.

    Returns:
        List[Dict[str, Any]]: Records interleaved across chemicals so structurally
            similar neighbours surface before any single chemical is exhausted.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r["CAS"], []).append(r)
    queues: List[List[Dict[str, Any]]] = list(groups.values())
    interleaved: List[Dict[str, Any]] = []
    while any(queues):
        for queue in queues:
            if queue:
                interleaved.append(queue.pop(0))
    return interleaved


def _format_sector(sg: Dict[str, Any], rep: Dict[str, Any]) -> str:
    """Format the NAICS sector label, preferring the report's values over the subgraph's.

    Args:
        sg (Dict[str, Any]): Subgraph dict (carries the sector on 'Chemical_Sector' paths).
        rep (Dict[str, Any]): Report dict (carries the sector on 'Chemical' paths).

    Returns:
        str: ``"<code> — <title>"``; ``N/A`` for any part found in neither.
    """
    code = rep.get("sector_code") or sg.get("sector_code") or sg.get("naics_code") or "N/A"
    title = rep.get("sector_title") or sg.get("sector_title") or sg.get("naics_title") or "N/A"
    return f"{code} — {title}"


def _rank_similarities(retrieved_nodes: List[Dict[str, Any]]) -> Dict[str, float]:
    """Map CAS to context similarity for analogues re-ranked by context (empty for molecular analogues)."""
    return {
        hit["node"]["cas_number"]: float(hit["context_similarity"])
        for hit in retrieved_nodes
        if hit.get("node", {}).get("cas_number") and "context_similarity" in hit
    }


def _reorder_by_specificity(
    rows: List[Dict[str, Any]],
    ranking: str,
    code_idf: Mapping[str, float],
    rank_sim: Mapping[str, float],
) -> List[Dict[str, Any]]:
    """Re-sort unique records by activity-code specificity (stable, so ties keep similarity order).

    ``specificity`` scores a record as similarity times IDF(code). ``consensus_specificity`` scores a
    code as the sum, over the distinct chemicals reporting it, of similarity times IDF(code), so a code
    reported by several neighbours rises; records are then ordered by their code's score.

    Args:
        rows (List[Dict[str, Any]]): Unique records, pre-sorted by descending similarity.
        ranking (str): ``specificity`` or ``consensus_specificity``.
        code_idf (Mapping[str, float]): IDF(c) per activity code (unseen codes get the maximum IDF).
        rank_sim (Mapping[str, float]): CAS to similarity overriding the molecular score (context analogues).

    Returns:
        List[Dict[str, Any]]: Records ordered by the chosen score, highest first.
    """
    fallback = max(code_idf.values()) if code_idf else 0.0
    sim = {r["CAS"]: rank_sim.get(r["CAS"], r["Similarity"]) for r in rows}
    idf = {r["ReductionCode"]: code_idf.get(r["ReductionCode"], fallback) for r in rows}
    if ranking == RANKING_SPECIFICITY:
        return sorted(rows, key=lambda r: sim[r["CAS"]] * idf[r["ReductionCode"]], reverse=True)
    support: Dict[Any, set] = {}
    for r in rows:
        support.setdefault(r["ReductionCode"], set()).add(r["CAS"])
    consensus = {c: sum(sim[cas] for cas in cas_set) * idf[c] for c, cas_set in support.items()}
    return sorted(rows, key=lambda r: consensus[r["ReductionCode"]], reverse=True)


def tabulate_records(
    retrieved_nodes: List[Dict[str, Any]],
    subgraphs: List[Dict[str, Any]],
    limit: int = 10,
    interleave: bool = False,
    ranking: str = RANKING_SIMILARITY,
    code_idf: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Rank and de-duplicate source-reduction records from retrieved subgraphs.

    Shared by the production pipeline (``RAGPipeline._tabulate``) and the offline
    recommendation evaluator. Records carry the raw ``ReductionCode`` (for evaluation)
    alongside the display fields.

    Args:
        retrieved_nodes (List[Dict[str, Any]]): Vector search hits with similarity scores.
        subgraphs (List[Dict[str, Any]]): Traversed subgraph dicts from retrieval.
        limit (int): Maximum number of unique records to return.
        interleave (bool): When True, round-robin across the retrieved chemicals so
            activities from structurally similar neighbours surface alongside the
            top chemical, instead of the top chemical saturating the list. When False
            (default, used by the evaluator) records are ranked purely by similarity.
            Ignored unless ``ranking`` is ``similarity``, because a specificity score replaces the interleave.
        ranking (str): ``similarity`` (default, shipped), ``specificity`` (similarity times the code's IDF) or
            ``consensus_specificity`` (per code, the sum over supporting neighbours of similarity times IDF).
            When analogues were re-ranked by context, "similarity" here is their context similarity.
        code_idf (Optional[Mapping[str, float]]): IDF per activity code; required unless ``ranking`` is
            ``similarity``. Compute it from index-split profiles when evaluating.

    Returns:
        List[Dict[str, Any]]: Unique source-reduction records.
    """
    scores: Dict[str, float] = {}
    via: Dict[str, str] = {}
    for hit in retrieved_nodes:
        node = hit.get("node", {})
        cas = node.get("cas_number")
        if cas:
            scores[cas] = max(scores.get(cas, 0.0), hit.get("score", 0.0))
            if node.get("via_candidate"):
                via[cas] = node["via_candidate"]

    resolve_switch("ranking", ranking)
    if ranking != RANKING_SIMILARITY and code_idf is None:
        raise ValueError(f"ranking='{ranking}' needs code_idf (IDF per activity code)")
    rank_sim = _rank_similarities(retrieved_nodes)

    rows: List[Dict[str, Any]] = []
    for sg in subgraphs:
        if sg.get("node_type", "") not in ("Chemical", "Chemical_Sector"):
            continue
        cas = sg.get("cas_number", "")
        name = sg.get("name", "N/A")
        similarity = scores.get(cas, 0.0)
        for rep in sg.get("reports", []):
            sector = _format_sector(sg, rep)
            code = rep.get("reduction_code")
            reduction = rep.get("reduction_name") or code
            if not reduction:
                continue
            rows.append(
                {
                    "Chemical": name,
                    "CAS": cas or "N/A",
                    "Similarity": round(similarity, 4),
                    "EntryPoint": via.get(cas, ""),
                    "Sector": sector,
                    "Reduction Activity": reduction,
                    "ReductionCode": code,
                    "Year": rep.get("year", "N/A"),
                    "State": rep.get("state", "N/A"),
                    **({"ContextSimilarity": round(rank_sim[cas], 4)} if cas in rank_sim else {}),
                }
            )

    if not rows:
        return []

    rows.sort(key=lambda r: rank_sim.get(r["CAS"], float(r["Similarity"])), reverse=True)
    seen: set = set()
    unique_rows: List[Dict[str, Any]] = []
    for r in rows:
        key = (r["Chemical"], r["Reduction Activity"], r["Year"], r["State"])
        if key not in seen:
            seen.add(key)
            unique_rows.append(r)

    if ranking != RANKING_SIMILARITY:
        unique_rows = _reorder_by_specificity(unique_rows, ranking, cast(Mapping[str, float], code_idf), rank_sim)
    elif interleave:
        unique_rows = _round_robin_by_chemical(unique_rows)
    return unique_rows[:limit]


def _table_row(r: Mapping[str, Any], rank: Optional[int]) -> str:
    line = f"| {r['Chemical']} | {r['CAS']} | {r['Similarity']} | {r['Sector']} | {r['Reduction Activity']} | {r['Year']} | {r['State']} |"
    if rank is None:
        return line
    return line + f" {r['Tier']} | {','.join(r['Channels'])} | {rank} |"


def records_table_lines(records: List[Dict[str, Any]]) -> List[str]:
    """Markdown table of the top records; fused records gain ``Tier``, ``Channels`` and ``Rank`` columns.

    Fused catch-all codes (names starting with "OTHER") follow the specific ones in a second table under a heading,
    keeping their original fused rank. The order of ``records`` itself is not changed.

    Args:
        records (List[Dict[str, Any]]): Records from ``tabulate_records`` or the fused retriever.

    Returns:
        List[str]: Title, header, separator and one line per record.
    """
    fused = bool(records) and "Tier" in records[0]
    header = "| Chemical | CAS | Similarity | Sector | Reduction Activity | Year | State |" + (
        " Tier | Channels | Rank |" if fused else ""
    )
    separator = "|---|---|---|---|---|---|---|" + ("---|---|---|" if fused else "")
    lines = ["\n**Top-10 Promising Source Reduction Records:**\n", header, separator]
    if not fused:
        return lines + [_table_row(r, None) for r in records]
    specific, catch_all = split_catch_all(records)
    lines.extend(_table_row(r, i) for i, r in specific)
    if catch_all:
        lines.extend(["", f"**{CATCH_ALL_HEADING}**\n", header, separator])
        lines.extend(_table_row(r, i) for i, r in catch_all)
    return lines


def records_to_context(records: List[Dict[str, Any]], query_smiles: str = "") -> str:
    """Render ranked source-reduction records into the generation context string.

    Grounding generation in the same records that the user is shown keeps the LLM
    narrative consistent with the recommendation table and prevents it from citing
    structurally similar chemicals that have no reduction records (a hallucination
    source when the raw subgraphs are used).

    Args:
        records (List[Dict[str, Any]]): Records from ``tabulate_records``.
        query_smiles (str): SMILES of the query chemical, for the context header.

    Returns:
        str: Context string grouped by chemical (descending similarity); empty if no records.
    """
    if not records:
        return ""
    if "Tier" in records[0]:
        return fused_records_to_context(records, query_smiles)
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((r.get("Chemical", "Unknown"), r.get("CAS", "N/A"), r.get("Similarity", 0.0)), []).append(r)
    lines: List[str] = []
    if query_smiles:
        lines.append(f"QUERY STRUCTURE (SMILES): {query_smiles}")
        lines.append("QUERY CHEMICAL NAME: not provided. Never give the query chemical the name of any block below.")
    lines.append(
        "Source-reduction activities reported for chemicals structurally similar to the query "
        "(Similarity = structural similarity to the query; 1.0 marks an identical structure). "
        "Every block tagged ANALOGUE is a different chemical, NOT the query chemical; its activities "
        "are transferable candidate recommendations."
    )
    lines.append("")
    for (name, cas, sim), recs in sorted(groups.items(), key=lambda kv: -float(kv[0][2])):
        tag = QUERY_TAG if float(sim) >= IDENTICAL_SIMILARITY else ANALOGUE_TAG
        entry = recs[0].get("EntryPoint", "")
        # A category (no SMILES) is reached through a member; the score is that member's.
        via = f" — category reached via member {entry}, whose structural similarity is {sim}" if entry else ""
        lines.append(f"[{name} | CAS {cas} | Similarity {sim}{tag}{via}]")
        for r in recs:
            lines.append(
                f"  - {r.get('Reduction Activity')} | Sector {r.get('Sector')} | "
                f"Year {r.get('Year')} | State {r.get('State')}"
            )
        lines.append("")
    return "\n".join(lines).strip()


def fix_query_chemical_mislabel(answer: str, records: List[Dict[str, Any]]) -> str:
    """Deterministically stop the answer from naming an analogue as the query chemical.

    Rewrites "the query chemical" to "the closest analogue" when, within the same clause, it is
    immediately followed by the name of a retrieved chemical whose structure is not identical to
    the query. Names of identical-structure records are left alone, and no name is ever invented.

    Args:
        answer (str): LLM-generated answer.
        records (List[Dict[str, Any]]): Records from ``tabulate_records``.

    Returns:
        str: The answer with mislabelled references corrected.
    """
    identical = {r.get("Chemical") for r in records if float(r.get("Similarity", 0.0)) >= IDENTICAL_SIMILARITY}
    analogues = {r["Chemical"] for r in records if r.get("Chemical") and r["Chemical"] not in identical}
    for name in sorted(analogues, key=len, reverse=True):
        pattern = re.compile(rf"\b(the )?query chemical(?=[^.\n]{{0,80}}?{re.escape(str(name))})", re.IGNORECASE)
        answer = pattern.sub("the closest analogue", answer)
    return answer


class QueryPlan(BaseModel):
    """Validated output of the LLM query planner."""

    model_config = ConfigDict(extra="ignore")

    chemical_query: Optional[str] = None
    sector_query: Optional[str] = None


class ChemicalExtraction(BaseModel):
    """Structured extraction of a chemical query and its optional metadata."""

    model_config = ConfigDict(extra="ignore")

    smiles: Optional[str] = Field(default=None, description="Chemical SMILES string, verbatim.")
    industry_sector: Optional[str] = Field(default=None, description="Free-text industry/sector name.")
    condition_of_use: Optional[str] = Field(default=None, description="Condition of use / activity context.")


class RAGState(TypedDict):
    """Shared state passed between LangGraph nodes."""

    query: str
    chemical_smiles: str
    chemical_query: str
    sector_query: str
    condition_of_use: str
    extracted: Dict[str, Any]
    needs_clarification: bool
    clarification: str
    retrieved_nodes: List[Dict[str, Any]]
    subgraphs: List[Dict[str, Any]]
    sector_fallback: bool
    cou_fallback: bool
    use_notice: bool
    sector_notice: bool
    record_industries: List[str]
    matched_uses: List[Dict[str, Any]]
    matched_sectors: List[Dict[str, Any]]
    switches: Dict[str, str]
    context: str
    answer: str
    records: List[Dict[str, Any]]


class RAGPipeline:
    """Compiled LangGraph GraphRAG pipeline (singleton)."""

    def __new__(cls) -> "RAGPipeline":
        """Create or return the singleton instance."""
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_graph"):
            return
        self._graph = self._build_graph()

    def _initial_state(
        self,
        query: str,
        chemical_smiles: str,
        industry_sector: str,
        condition_of_use: str,
        switches: Optional[Dict[str, str]] = None,
    ) -> RAGState:
        return {
            "query": query,
            "chemical_smiles": chemical_smiles,
            "chemical_query": "",
            "sector_query": industry_sector,
            "condition_of_use": condition_of_use,
            "extracted": {},
            "needs_clarification": False,
            "clarification": "",
            "retrieved_nodes": [],
            "subgraphs": [],
            "sector_fallback": False,
            "cou_fallback": False,
            "use_notice": False,
            "sector_notice": False,
            "record_industries": [],
            "matched_uses": [],
            "matched_sectors": [],
            "switches": dict(switches or {}),
            "context": "",
            "answer": "",
            "records": [],
        }

    def run(
        self,
        query: str,
        chemical_smiles: str = "",
        industry_sector: str = "",
        condition_of_use: str = "",
        switches: Optional[Dict[str, str]] = None,
    ) -> str:
        """Execute the full extract → (plan) → retrieve → context → generate pipeline.

        Args:
            query (str): Natural language question for the LLM.
            chemical_smiles (str): Optional SMILES string override. If omitted,
                the extractor/planner resolves a chemical identifier from ``query``.
            industry_sector (str): Optional industry/sector name override.
            condition_of_use (str): Optional condition-of-use override.
            switches (Optional[Dict[str, str]]): Per-call overrides for ``ranking``, ``cou_filter``,
                ``analogues`` and ``retrieval`` (``legacy`` or ``fused``), plus ``chemical_embedding`` (fused only:
                ``MorganFingerprint``, ``MolDeBERTa`` or ``Hybrid``); omitted keys use config (shipped defaults).

        Returns:
            str: The LLM-generated answer grounded in the knowledge graph context.
        """
        initial_state = self._initial_state(query, chemical_smiles, industry_sector, condition_of_use, switches)
        result = self._graph.invoke(initial_state)
        answer = str(result["answer"])
        records: List[Dict[str, Any]] = result.get("records", [])
        if not records:
            return answer
        return f"{answer}\n\n" + "\n".join(records_table_lines(records))

    def run_structured(
        self,
        query: str,
        chemical_smiles: str = "",
        industry_sector: str = "",
        condition_of_use: str = "",
        switches: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute the pipeline and return answer, records, and clarification state.

        Args:
            query (str): Natural language question for the LLM.
            chemical_smiles (str): Optional SMILES string override.
            industry_sector (str): Optional industry/sector name override.
            condition_of_use (str): Optional condition-of-use override.
            switches (Optional[Dict[str, str]]): Per-call overrides for ``ranking``, ``cou_filter``,
                ``analogues`` and ``retrieval`` (``legacy`` or ``fused``), plus ``chemical_embedding`` (fused only:
                ``MorganFingerprint``, ``MolDeBERTa`` or ``Hybrid``); omitted keys use config (shipped defaults).

        Returns:
            Dict[str, Any]: Keys ``answer`` (str), ``records`` (list of dicts),
                ``needs_clarification`` (bool), ``clarification`` (str), and, in fused mode, ``matched_uses`` and
                ``matched_sectors`` (what the use and sector channels matched).
        """
        initial_state = self._initial_state(query, chemical_smiles, industry_sector, condition_of_use, switches)
        result = self._graph.invoke(initial_state)
        return {
            "answer": str(result["answer"]),
            "records": result.get("records", []),
            "needs_clarification": bool(result.get("needs_clarification", False)),
            "clarification": str(result.get("clarification", "")),
            "matched_uses": result.get("matched_uses", []),
            "matched_sectors": result.get("matched_sectors", []),
        }

    def _extract(self, state: RAGState) -> Dict[str, Any]:
        cfg = ConfigManager().config["RAG_GRAPH"]["GENERATION"]
        llm = ChatOllama(
            model=cfg["MODEL"],
            base_url=os.getenv("OLLAMA_BASE_URL", cfg["BASE_URL"]),
            temperature=0.0,
        )
        pm = PromptManager()
        prompt_text = pm.render("rag_extract.yaml", "extract_prompt", {"input": state["query"]})
        try:
            structured = llm.with_structured_output(ChemicalExtraction)
            raw = structured.invoke([HumanMessage(content=prompt_text)])
            extracted = raw if isinstance(raw, ChemicalExtraction) else ChemicalExtraction.model_validate(raw)
        except Exception as exc:
            logger.warning(f"Structured extraction failed ({exc}); proceeding with empty extraction")
            extracted = ChemicalExtraction()

        smiles_candidate = state.get("chemical_smiles") or (extracted.smiles or "")
        updates: Dict[str, Any] = {
            "extracted": extracted.model_dump(),
            "sector_query": state.get("sector_query") or (extracted.industry_sector or ""),
            "condition_of_use": state.get("condition_of_use") or (extracted.condition_of_use or ""),
            "needs_clarification": False,
            "clarification": "",
            "chemical_smiles": "",
        }
        return self._validate_smiles(smiles_candidate, updates)

    def _validate_smiles(self, smiles_candidate: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        if not smiles_candidate:
            return updates
        mol = Chem.MolFromSmiles(smiles_candidate)
        if mol is None:
            updates["needs_clarification"] = True
            updates["clarification"] = (
                f"The provided SMILES string '{smiles_candidate}' is not a valid chemical structure. "
                "Please provide a valid SMILES string."
            )
            return updates
        updates["chemical_smiles"] = Chem.MolToSmiles(mol)
        return updates

    def _clarify(self, state: RAGState) -> Dict[str, Any]:
        return {"answer": state.get("clarification", "")}

    def _route_after_extract(self, state: RAGState) -> str:
        if state.get("needs_clarification"):
            return "clarify"
        if state.get("chemical_smiles"):
            return "retrieve"
        return "plan"

    def _plan(self, state: RAGState) -> Dict[str, Any]:
        cfg = ConfigManager().config["RAG_GRAPH"]["GENERATION"]
        llm = ChatOllama(
            model=cfg["MODEL"],
            base_url=os.getenv("OLLAMA_BASE_URL", cfg["BASE_URL"]),
            temperature=0.0,
            format="json",
        )
        pm = PromptManager()
        plan_text = pm.render("rag_plan.yaml", "plan_prompt", {"query": state["query"]})
        response = llm.invoke([HumanMessage(content=plan_text)])
        try:
            raw = clean_llm_json(str(response.content))
            raw = fix_json_formatting(raw)
            plan = QueryPlan.model_validate_json(raw)
        except Exception as exc:
            logger.warning(f"Query plan parsing failed ({exc}); proceeding with empty plan")
            plan = QueryPlan()

        chemical_query = ""
        if not state.get("chemical_smiles") and plan.chemical_query:
            chemical_query = plan.chemical_query

        return {
            "chemical_query": chemical_query,
            "sector_query": plan.sector_query or "",
        }

    def _retrieve(self, state: RAGState) -> Dict[str, Any]:
        factory = RetrieverFactory()

        chemical_smiles = state.get("chemical_smiles", "")
        if not chemical_smiles and state.get("chemical_query"):
            chemical_smiles = factory.get_cross_retriever().resolve_smiles(state["chemical_query"])

        switches = state.get("switches", {})
        if resolve_switch("retrieval", switches.get("retrieval")) == RETRIEVAL_FUSED:
            return self._retrieve_fused(factory, state, chemical_smiles)
        result = factory.retrieve_multi_result(
            chemical_smiles=chemical_smiles,
            sector_query=state.get("sector_query", ""),
            condition_of_use=state.get("condition_of_use", ""),
            include_candidates=True,
            cou_filter=switches.get("cou_filter"),
            analogues=switches.get("analogues"),
        )
        return {
            "chemical_smiles": chemical_smiles,
            "retrieved_nodes": result.retrieved_nodes,
            "subgraphs": result.subgraphs,
            "sector_fallback": result.sector_fallback,
            "cou_fallback": result.cou_fallback,
        }

    def _retrieve_fused(self, factory: RetrieverFactory, state: RAGState, chemical_smiles: str) -> Dict[str, Any]:
        result = factory.retrieve_fused(
            chemical_smiles=chemical_smiles,
            sector_query=state.get("sector_query", ""),
            condition_of_use=state.get("condition_of_use", ""),
            chemical_embedding=state.get("switches", {}).get("chemical_embedding"),
        )
        return {
            "chemical_smiles": chemical_smiles,
            "retrieved_nodes": result.retrieved_nodes,
            "records": result.records,
            "use_notice": result.use_unmatched,
            "sector_notice": result.sector_unmatched,
            "record_industries": result.record_industries,
            "matched_uses": result.matched_uses,
            "matched_sectors": result.matched_sectors,
        }

    def _tabulate(self, state: RAGState) -> Dict[str, Any]:
        if resolve_switch("retrieval", state.get("switches", {}).get("retrieval")) == RETRIEVAL_FUSED:
            return {"records": state.get("records", [])}
        ranking = resolve_switch("ranking", state.get("switches", {}).get("ranking"))
        extra: Dict[str, Any] = {}
        if ranking != RANKING_SIMILARITY:
            extra = {"ranking": ranking, "code_idf": RetrieverFactory().code_idf()}
        records = tabulate_records(
            state.get("retrieved_nodes", []),
            state.get("subgraphs", []),
            interleave=True,
            **extra,
        )
        return {"records": records}

    def _build_context(self, state: RAGState) -> Dict[str, Any]:
        context = records_to_context(state.get("records", []), state.get("chemical_smiles", ""))
        return {"context": context}

    def _no_records_message(self, state: RAGState) -> str:
        chemical = state.get("chemical_query") or state.get("chemical_smiles") or "the query chemical"
        scope = ""
        if state.get("sector_query"):
            scope += f" in the '{state['sector_query']}' sector"
        if state.get("condition_of_use"):
            scope += f" under the condition of use '{state['condition_of_use']}'"
        return (
            f"No source-reduction records were found for {chemical}{scope}. No facilities "
            "reported source-reduction activities for this chemical or its structurally similar "
            "analogues under the specified context in the TRI data (1987–2024)."
        )

    def _generate(self, state: RAGState) -> Dict[str, Any]:
        # With no retrieved records the context is empty; answer deterministically rather
        # than letting the LLM speculate about missing input.
        if not state.get("records"):
            return {"answer": self._no_records_message(state)}
        cfg = ConfigManager().config["RAG_GRAPH"]["GENERATION"]
        llm = ChatOllama(
            model=cfg["MODEL"],
            base_url=os.getenv("OLLAMA_BASE_URL", cfg["BASE_URL"]),
            temperature=float(cfg["TEMPERATURE"]),
        )
        pm = PromptManager()
        system_text = pm.render(
            "rag_system.yaml",
            "system_prompt",
            {
                "context": state["context"],
                "condition_of_use": state.get("condition_of_use") or "not specified",
            },
        )
        query_text = pm.render(
            "rag_query.yaml",
            "query_prompt",
            {"query": state["query"]},
        )
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=query_text),
        ]
        response = llm.invoke(messages)
        answer = fix_query_chemical_mislabel(str(response.content), state["records"])
        if state.get("sector_fallback") and state.get("sector_query"):
            notice = SECTOR_FALLBACK_NOTICE.format(sector=state["sector_query"])
            answer = f"{notice}\n\n{answer}"
        if state.get("cou_fallback") and state.get("condition_of_use"):
            notice = COU_FALLBACK_NOTICE.format(cou=state["condition_of_use"])
            answer = f"{notice}\n\n{answer}"
        if state.get("sector_notice") and state.get("sector_query"):
            industries = "; ".join(state.get("record_industries", []))
            answer = f"{SECTOR_UNMATCHED_NOTICE.format(sector=state['sector_query'], industries=industries)}\n\n{answer}"
        if state.get("use_notice") and state.get("condition_of_use"):
            answer = f"{USE_UNMATCHED_NOTICE.format(cou=state['condition_of_use'])}\n\n{answer}"
        return {"answer": answer}

    def _build_graph(self) -> Any:
        graph = StateGraph(RAGState)
        graph.add_node("extract", self._extract)
        graph.add_node("clarify", self._clarify)
        graph.add_node("plan", self._plan)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("build_context", self._build_context)
        graph.add_node("generate", self._generate)
        graph.add_node("tabulate", self._tabulate)
        graph.add_edge(START, "extract")
        graph.add_conditional_edges(
            "extract",
            self._route_after_extract,
            {"clarify": "clarify", "retrieve": "retrieve", "plan": "plan"},
        )
        graph.add_edge("clarify", END)
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "tabulate")
        graph.add_edge("tabulate", "build_context")
        graph.add_edge("build_context", "generate")
        graph.add_edge("generate", END)
        return graph.compile()
