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
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field
from rdkit import Chem
from typing_extensions import TypedDict

from src.config_manager import ConfigManager
from src.rag_graph.helpers import clean_llm_json, fix_json_formatting
from src.rag_graph.retrieval import RetrieverFactory

from .prompt_manager import PromptManager

logger = logging.getLogger(__name__)


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


def tabulate_records(
    retrieved_nodes: List[Dict[str, Any]],
    subgraphs: List[Dict[str, Any]],
    limit: int = 10,
    interleave: bool = False,
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

    rows: List[Dict[str, Any]] = []
    for sg in subgraphs:
        if sg.get("node_type", "") not in ("Chemical", "Chemical_Sector"):
            continue
        cas = sg.get("cas_number", "")
        name = sg.get("name", "N/A")
        similarity = scores.get(cas, 0.0)
        sector = (
            f"{sg.get('sector_code', sg.get('naics_code', 'N/A'))} — " f"{sg.get('sector_title', sg.get('naics_title', 'N/A'))}"
        )
        for rep in sg.get("reports", []):
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
                }
            )

    if not rows:
        return []

    rows.sort(key=lambda r: r["Similarity"], reverse=True)
    seen: set = set()
    unique_rows: List[Dict[str, Any]] = []
    for r in rows:
        key = (r["Chemical"], r["Reduction Activity"], r["Year"], r["State"])
        if key not in seen:
            seen.add(key)
            unique_rows.append(r)

    if interleave:
        unique_rows = _round_robin_by_chemical(unique_rows)
    return unique_rows[:limit]


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
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((r.get("Chemical", "Unknown"), r.get("CAS", "N/A"), r.get("Similarity", 0.0)), []).append(r)
    lines: List[str] = []
    if query_smiles:
        lines.append(f"Query chemical SMILES: {query_smiles}")
    lines.append(
        "Source-reduction activities reported for the query chemical and its structurally most similar "
        "chemicals (Similarity = structural similarity to the query; 1.0 marks the query chemical). "
        "Activities from close analogues are transferable candidate recommendations."
    )
    lines.append("")
    for (name, cas, sim), recs in sorted(groups.items(), key=lambda kv: -float(kv[0][2])):
        tag = " — QUERY CHEMICAL" if float(sim) >= 0.99999 else ""
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
    ) -> str:
        """Execute the full extract → (plan) → retrieve → context → generate pipeline.

        Args:
            query (str): Natural language question for the LLM.
            chemical_smiles (str): Optional SMILES string override. If omitted,
                the extractor/planner resolves a chemical identifier from ``query``.
            industry_sector (str): Optional industry/sector name override.
            condition_of_use (str): Optional condition-of-use override.

        Returns:
            str: The LLM-generated answer grounded in the knowledge graph context.
        """
        initial_state = self._initial_state(query, chemical_smiles, industry_sector, condition_of_use)
        result = self._graph.invoke(initial_state)
        answer = str(result["answer"])
        records: List[Dict[str, Any]] = result.get("records", [])
        if not records:
            return answer
        header = "| Chemical | CAS | Similarity | Sector | Reduction Activity | Year | State |"
        separator = "|---|---|---|---|---|---|---|"
        lines = ["\n**Top-10 Promising Source Reduction Records:**\n", header, separator]
        for r in records:
            lines.append(
                f"| {r['Chemical']} | {r['CAS']} | {r['Similarity']} "
                f"| {r['Sector']} | {r['Reduction Activity']} | {r['Year']} | {r['State']} |"
            )
        return f"{answer}\n\n" + "\n".join(lines)

    def run_structured(
        self,
        query: str,
        chemical_smiles: str = "",
        industry_sector: str = "",
        condition_of_use: str = "",
    ) -> Dict[str, Any]:
        """Execute the pipeline and return answer, records, and clarification state.

        Args:
            query (str): Natural language question for the LLM.
            chemical_smiles (str): Optional SMILES string override.
            industry_sector (str): Optional industry/sector name override.
            condition_of_use (str): Optional condition-of-use override.

        Returns:
            Dict[str, Any]: Keys ``answer`` (str), ``records`` (list of dicts),
                ``needs_clarification`` (bool), and ``clarification`` (str).
        """
        initial_state = self._initial_state(query, chemical_smiles, industry_sector, condition_of_use)
        result = self._graph.invoke(initial_state)
        return {
            "answer": str(result["answer"]),
            "records": result.get("records", []),
            "needs_clarification": bool(result.get("needs_clarification", False)),
            "clarification": str(result.get("clarification", "")),
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

        retrieved_nodes, subgraphs = factory.retrieve_multi(
            chemical_smiles=chemical_smiles,
            sector_query=state.get("sector_query", ""),
            condition_of_use=state.get("condition_of_use", ""),
            include_candidates=True,
        )
        return {
            "chemical_smiles": chemical_smiles,
            "retrieved_nodes": retrieved_nodes,
            "subgraphs": subgraphs,
        }

    def _tabulate(self, state: RAGState) -> Dict[str, Any]:
        records = tabulate_records(
            state.get("retrieved_nodes", []),
            state.get("subgraphs", []),
            interleave=True,
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
        return {"answer": str(response.content)}

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
