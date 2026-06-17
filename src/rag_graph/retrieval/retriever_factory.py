# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Factory that wires embedding models to Neo4j vector indexes.

Builds and caches ``Neo4jVector`` instances (via LangChain) and ``GraphRetriever``
instances, keyed by embedding type, so models are loaded only once per process.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-09

"""

import logging
import os
from typing import Any, Dict, List, Mapping, Tuple, cast

from langchain_neo4j import Neo4jVector

from src.config_manager import ConfigManager
from src.rag_graph.embedding import (
    HybridMolecularEmbeddings,
    MolDeBERTaEmbeddings,
    MorganFingerprintEmbeddings,
    NAICSEmbeddings,
)

from .graph_retriever import CrossEntityRetriever, GraphRetriever

logger = logging.getLogger(__name__)

# Maps embedding_type → Neo4j vector index metadata.
# text_properties drives page_content; the key property (cas_number / naics_code)
# is intentionally excluded so it lands in doc.metadata for graph traversal.
INDEX_MAP: Dict[str, Dict[str, Any]] = {
    "MorganFingerprint": {
        "index_name": "chemical_morgan_fp_index",
        "node_label": "Chemical",
        "embedding_property": "morgan_fp_embedding",
        "text_properties": ["name"],
    },
    "MolDeBERTa": {
        "index_name": "chemical_moldeberta_index",
        "node_label": "Chemical",
        "embedding_property": "moldeberta_embedding",
        "text_properties": ["name"],
    },
    "Hybrid": {
        "index_name": "chemical_hybrid_index",
        "node_label": "Chemical",
        "embedding_property": "hybrid_embedding",
        "text_properties": ["name"],
    },
    "NAICSEmbeddings": {
        "index_name": "sector_naics_title_index",
        "node_label": "Sector",
        "embedding_property": "naics_title_embedding",
        "text_properties": ["naics_title"],
    },
}

# Node properties that hold embedding vectors — excluded from the node dict
# passed to the graph traverser to avoid sending megabyte-scale arrays around.
_EMBEDDING_PROPS = {
    "morgan_fp_embedding",
    "moldeberta_embedding",
    "hybrid_embedding",
    "naics_title_embedding",
}


def _node_from_metadata(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in metadata.items() if k not in _EMBEDDING_PROPS}


def _vector_hit(doc: Any, score: float) -> Dict[str, Any]:
    return {
        "node": _node_from_metadata(cast(Mapping[str, Any], doc.metadata)),
        "content": doc.page_content,
        "score": float(score),
    }


def _seed_nodes(retrieved_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [cast(Dict[str, Any], hit["node"]) for hit in retrieved_nodes]


class RetrieverFactory:
    """Singleton factory that builds and caches retriever pairs per embedding type."""

    def __new__(cls) -> "RetrieverFactory":
        """Create or return the singleton instance."""
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return

        neo4j_cfg = ConfigManager().config["DUMMY_NEO4J_CONFIG"]
        rag_cfg = ConfigManager().config["RAG_GRAPH"]
        emb_cfg: List[Dict[str, Any]] = rag_cfg["EMBEDDINGS"]
        self._url: str = os.getenv("NEO4J_URI", neo4j_cfg["NEO4J_URI"])
        self._user: str = os.getenv("NEO4J_USER", neo4j_cfg["NEO4J_USER"])
        self._password: str = os.getenv("NEO4J_PASSWORD", neo4j_cfg["NEO4J_PASSWORD"])
        self._top_k: int = rag_cfg["RETRIEVAL"]["TOP_K"]
        self._graph_limit: int = rag_cfg["RETRIEVAL"]["GRAPH_CONTEXT_LIMIT"]
        self._chem_embedding_type: str = rag_cfg["RETRIEVAL"].get("CHEMICAL_EMBEDDING_TYPE", "MolDeBERTa")
        self._emb_cfg = emb_cfg
        self._vector_stores: Dict[str, Neo4jVector] = {}
        self._graph_retrievers: Dict[str, GraphRetriever] = {}
        self._cross_retriever: CrossEntityRetriever | None = None
        self._initialized = True
        logger.info(f"RetrieverFactory using chemical embedding '{self._chem_embedding_type}' for retrieval")

    def _build_embedder(self, embedding_type: str) -> Any:
        for spec in self._emb_cfg:
            if spec["EMBEDDING_TYPE"] == embedding_type:
                props = spec["PROPERTIES"][0]
                if embedding_type == "MorganFingerprint":
                    return MorganFingerprintEmbeddings(
                        radius=props["RADIUS"],
                        n_bits=props["N_BITS"],
                    )
                if embedding_type == "MolDeBERTa":
                    return MolDeBERTaEmbeddings(
                        model_name=props["MODEL_NAME"],
                        max_length=int(props["MAX_LENGTH"]),
                    )
                if embedding_type == "Hybrid":
                    return HybridMolecularEmbeddings(
                        model_name=props["MODEL_NAME"],
                        max_length=int(props["MAX_LENGTH"]),
                    )
                if embedding_type == "NAICSEmbeddings":
                    return NAICSEmbeddings(
                        model_name=props["MODEL_NAME"],
                        max_length=int(props["MAX_LENGTH"]),
                    )
        raise ValueError(f"Unknown embedding type: '{embedding_type}'")

    def get_vector_store(self, embedding_type: str) -> Neo4jVector:
        """Return a cached ``Neo4jVector`` for the requested embedding type.

        Args:
            embedding_type (str): One of ``MorganFingerprint``, ``MolDeBERTa``,
                ``Hybrid``, or ``NAICSEmbeddings``.

        Returns:
            Neo4jVector: Configured LangChain vector store ready for similarity search.
        """
        if embedding_type not in self._vector_stores:
            if embedding_type not in INDEX_MAP:
                raise ValueError(f"Unknown embedding type: '{embedding_type}'")

            cfg = INDEX_MAP[embedding_type]
            embedder = self._build_embedder(embedding_type)
            self._vector_stores[embedding_type] = Neo4jVector.from_existing_index(
                embedding=embedder,
                url=self._url,
                username=self._user,
                password=self._password,
                index_name=cfg["index_name"],
                node_label=cfg["node_label"],
                embedding_node_property=cfg["embedding_property"],
                text_node_properties=cfg["text_properties"],
            )
            logger.info(f"Built Neo4jVector for embedding type '{embedding_type}'")
        return self._vector_stores[embedding_type]

    def get_graph_retriever(self, embedding_type: str) -> GraphRetriever:
        """Return a cached ``GraphRetriever`` for the node label of the embedding type.

        Args:
            embedding_type (str): One of the supported embedding type strings.

        Returns:
            GraphRetriever: Configured graph traversal retriever.
        """
        node_label: str = INDEX_MAP[embedding_type]["node_label"]
        if node_label not in self._graph_retrievers:
            self._graph_retrievers[node_label] = GraphRetriever(node_label, self._graph_limit)
            logger.info(f"Built GraphRetriever for node label '{node_label}'")
        return self._graph_retrievers[node_label]

    def get_cross_retriever(self) -> CrossEntityRetriever:
        """Return a cached ``CrossEntityRetriever``.

        Returns:
            CrossEntityRetriever: Configured for SMILES resolution and cross-entity traversal.
        """
        if self._cross_retriever is None:
            self._cross_retriever = CrossEntityRetriever(self._graph_limit)
            logger.info("Built CrossEntityRetriever")
        return self._cross_retriever

    def retrieve_multi(
        self,
        chemical_smiles: str = "",
        sector_query: str = "",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run parallel chemical and sector retrieval, intersecting via graph traversal.

        Chemical retrieval uses the configured chemical index (SMILES embedding;
        ``RAG_GRAPH.RETRIEVAL.CHEMICAL_EMBEDDING_TYPE``). Sector retrieval uses the
        NAICS title index (text embedding). When both return results, a cross-entity
        Cypher query finds reports at their intersection. Falls back to individual
        traversals if the intersection is empty.

        Args:
            chemical_smiles (str): SMILES string for chemical similarity search.
            sector_query (str): Sector description text for NAICS embedding search.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: ``(retrieved_nodes, subgraphs)``.
        """
        retrieved_nodes: List[Dict[str, Any]] = []
        cas_numbers: List[str] = []
        naics_codes: List[str] = []

        if chemical_smiles:
            vector_store = self.get_vector_store(self._chem_embedding_type)
            hits = vector_store.similarity_search_with_score(chemical_smiles, k=self._top_k)
            for doc, score in hits:
                hit = _vector_hit(doc, score)
                node = hit["node"]
                retrieved_nodes.append(hit)
                if "cas_number" in node:
                    cas_numbers.append(node["cas_number"])

        if sector_query:
            vector_store = self.get_vector_store("NAICSEmbeddings")
            hits = vector_store.similarity_search_with_score(sector_query, k=self._top_k)
            for doc, score in hits:
                hit = _vector_hit(doc, score)
                node = hit["node"]
                retrieved_nodes.append(hit)
                if "naics_code" in node:
                    naics_codes.append(node["naics_code"])

        cross = self.get_cross_retriever()
        if cas_numbers and naics_codes:
            subgraphs = cross.traverse_cross(cas_numbers, naics_codes)
            if not subgraphs:
                logger.warning("Cross-entity traversal empty; falling back to individual traversals")
                chem_nodes = _seed_nodes([r for r in retrieved_nodes if "cas_number" in r["node"]])
                sector_nodes = _seed_nodes([r for r in retrieved_nodes if "naics_code" in r["node"]])
                subgraphs = self.get_graph_retriever(self._chem_embedding_type).traverse_all(
                    chem_nodes
                ) + self.get_graph_retriever("NAICSEmbeddings").traverse_all(sector_nodes)
        elif cas_numbers:
            chem_nodes = _seed_nodes([r for r in retrieved_nodes if "cas_number" in r["node"]])
            subgraphs = self.get_graph_retriever(self._chem_embedding_type).traverse_all(chem_nodes)
        elif naics_codes:
            sector_nodes = _seed_nodes([r for r in retrieved_nodes if "naics_code" in r["node"]])
            subgraphs = self.get_graph_retriever("NAICSEmbeddings").traverse_all(sector_nodes)
        else:
            subgraphs = []

        return retrieved_nodes, subgraphs

    def retrieve(
        self,
        embedding_type: str,
        retrieval_query: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run vector search then graph traversal for the given query string.

        Args:
            embedding_type (str): Which embedding index to search.
            retrieval_query (str): SMILES string (chemical embeddings) or sector
                title text (NAICS embeddings) to embed and search.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: A pair of
                ``(retrieved_nodes, subgraphs)`` where ``retrieved_nodes`` are the
                raw vector search hits and ``subgraphs`` are the traversed results.
        """
        vector_store = self.get_vector_store(embedding_type)
        hits = vector_store.similarity_search_with_score(retrieval_query, k=self._top_k)
        retrieved_nodes = [_vector_hit(doc, score) for doc, score in hits]
        graph_ret = self.get_graph_retriever(embedding_type)
        subgraphs = graph_ret.traverse_all(_seed_nodes(retrieved_nodes))
        return retrieved_nodes, subgraphs
