# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""LangGraph-based GraphRAG pipeline for TRI chemical knowledge graph queries.

Pipeline stages:
    retrieve      → vector ANN search via Neo4jVector
    traverse      → subgraph expansion via Neo4jGraph (Cypher)
    build_context → format subgraphs into a prompt-ready context string
    generate      → answer via ChatOllama + YAML-backed prompts

Author: Jose D. Hernandez-Betancur
Date: 2026-05-09
"""

import logging
import os
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.config_manager import ConfigManager
from src.rag_graph.context import ContextBuilder
from src.rag_graph.retrieval import RetrieverFactory

from .prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """Shared state passed between LangGraph nodes."""

    query: str
    retrieval_query: str
    embedding_type: str
    retrieved_nodes: List[Dict[str, Any]]
    subgraphs: List[Dict[str, Any]]
    context: str
    answer: str


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
        retrieval_query: str,
        embedding_type: str = "MolDeBERTa",
    ) -> str:
        """Execute the full retrieve → context → generate pipeline.

        Args:
            query (str): Natural language question for the LLM.
            retrieval_query (str): SMILES string or sector title text to embed
                for vector search (must match the ``embedding_type``).
            embedding_type (str): One of ``MorganFingerprint``, ``MolDeBERTa``,
                ``Hybrid``, or ``NAICSEmbeddings``.

        Returns:
            str: The LLM-generated answer grounded in the knowledge graph context.
        """
        initial_state: RAGState = {
            "query": query,
            "retrieval_query": retrieval_query,
            "embedding_type": embedding_type,
            "retrieved_nodes": [],
            "subgraphs": [],
            "context": "",
            "answer": "",
        }
        result = self._graph.invoke(initial_state)
        return str(result["answer"])

    def _retrieve(self, state: RAGState) -> Dict[str, Any]:
        factory = RetrieverFactory()
        retrieved_nodes, subgraphs = factory.retrieve(
            state["embedding_type"],
            state["retrieval_query"],
        )
        return {
            "retrieved_nodes": retrieved_nodes,
            "subgraphs": subgraphs,
        }

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
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("build_context", self._build_context)
        graph.add_node("generate", self._generate)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "build_context")
        graph.add_edge("build_context", "generate")
        graph.add_edge("generate", END)
        return graph.compile()
