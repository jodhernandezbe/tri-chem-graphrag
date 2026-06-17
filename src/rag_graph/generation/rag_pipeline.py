# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""LangGraph-based GraphRAG pipeline for TRI chemical knowledge graph queries.

Pipeline stages:
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
from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict

from src.config_manager import ConfigManager
from src.rag_graph.context import ContextBuilder
from src.rag_graph.helpers import clean_llm_json, fix_json_formatting
from src.rag_graph.retrieval import RetrieverFactory

from .prompt_manager import PromptManager

logger = logging.getLogger(__name__)


def tabulate_records(
    retrieved_nodes: List[Dict[str, Any]],
    subgraphs: List[Dict[str, Any]],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Rank and de-duplicate source-reduction records from retrieved subgraphs.

    Shared by the production pipeline (``RAGPipeline._tabulate``) and the offline
    recommendation evaluator so both score the identical ranking. Records carry the
    raw ``ReductionCode`` (for evaluation) alongside the display fields.

    Args:
        retrieved_nodes (List[Dict[str, Any]]): Vector search hits with similarity scores.
        subgraphs (List[Dict[str, Any]]): Traversed subgraph dicts from retrieval.
        limit (int): Maximum number of unique records to return, ranked by similarity.

    Returns:
        List[Dict[str, Any]]: Unique source-reduction records sorted by similarity.
    """
    scores: Dict[str, float] = {}
    for hit in retrieved_nodes:
        node = hit.get("node", {})
        cas = node.get("cas_number")
        if cas:
            scores[cas] = max(scores.get(cas, 0.0), hit.get("score", 0.0))

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

    return unique_rows[:limit]


class QueryPlan(BaseModel):
    """Validated output of the LLM query planner."""

    model_config = ConfigDict(extra="ignore")

    chemical_query: Optional[str] = None
    sector_query: Optional[str] = None


class RAGState(TypedDict):
    """Shared state passed between LangGraph nodes."""

    query: str
    chemical_smiles: str
    chemical_query: str
    sector_query: str
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

    def run(
        self,
        query: str,
        chemical_smiles: str = "",
    ) -> str:
        """Execute the full plan → retrieve → context → generate pipeline.

        Args:
            query (str): Natural language question for the LLM.
            chemical_smiles (str): Optional SMILES string override. If omitted,
                the planner extracts a chemical identifier from ``query`` and
                resolves it to SMILES via the knowledge graph.

        Returns:
            str: The LLM-generated answer grounded in the knowledge graph context.
        """
        initial_state: RAGState = {
            "query": query,
            "chemical_smiles": chemical_smiles,
            "chemical_query": "",
            "sector_query": "",
            "retrieved_nodes": [],
            "subgraphs": [],
            "context": "",
            "answer": "",
            "records": [],
        }
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
    ) -> Dict[str, Any]:
        """Execute the pipeline and return answer and records separately.

        Args:
            query (str): Natural language question for the LLM.
            chemical_smiles (str): Optional SMILES string override.

        Returns:
            Dict[str, Any]: Keys ``answer`` (str) and ``records`` (list of dicts).
        """
        initial_state: RAGState = {
            "query": query,
            "chemical_smiles": chemical_smiles,
            "chemical_query": "",
            "sector_query": "",
            "retrieved_nodes": [],
            "subgraphs": [],
            "context": "",
            "answer": "",
            "records": [],
        }
        result = self._graph.invoke(initial_state)
        return {
            "answer": str(result["answer"]),
            "records": result.get("records", []),
        }

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
        )
        return {
            "chemical_smiles": chemical_smiles,
            "retrieved_nodes": retrieved_nodes,
            "subgraphs": subgraphs,
        }

    def _tabulate(self, state: RAGState) -> Dict[str, Any]:
        records = tabulate_records(state.get("retrieved_nodes", []), state.get("subgraphs", []))
        return {"records": records}

    def _build_context(self, state: RAGState) -> Dict[str, Any]:
        context = ContextBuilder().build(state["subgraphs"])
        return {"context": context}

    def _generate(self, state: RAGState) -> Dict[str, Any]:
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
            {"context": state["context"]},
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
        graph.add_node("plan", self._plan)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("build_context", self._build_context)
        graph.add_node("generate", self._generate)
        graph.add_node("tabulate", self._tabulate)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "build_context")
        graph.add_edge("build_context", "generate")
        graph.add_edge("generate", "tabulate")
        graph.add_edge("tabulate", END)
        return graph.compile()
