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
    ActivityEmbeddings,
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
    "ActivityEmbeddings": {
        "index_name": "activity_name_index",
        "node_label": "Activity",
        "embedding_property": "activity_name_embedding",
        "text_properties": ["name"],
    },
}

# Candidate indices reuse the chemical embedders over the :Candidate label, keyed by the
# configured chemical embedding type, so a no-SMILES category can be reached via a member.
_CANDIDATE_INDEX_MAP: Dict[str, Dict[str, str]] = {
    "MorganFingerprint": {"index_name": "candidate_morgan_fp_index", "embedding_property": "morgan_fp_embedding"},
    "MolDeBERTa": {"index_name": "candidate_moldeberta_index", "embedding_property": "moldeberta_embedding"},
    "Hybrid": {"index_name": "candidate_hybrid_index", "embedding_property": "hybrid_embedding"},
}

# Node properties that hold embedding vectors — excluded from the node dict
# passed to the graph traverser to avoid sending megabyte-scale arrays around.
_EMBEDDING_PROPS = {
    "morgan_fp_embedding",
    "moldeberta_embedding",
    "hybrid_embedding",
    "naics_title_embedding",
    "activity_name_embedding",
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


def guarded_search(vector_store: Neo4jVector, query: str, k: int) -> List[Tuple[Any, float]]:
    """Embed and search, skipping queries that have no valid embedding.

    Chemical embeddings (e.g. Morgan fingerprints) yield an all-zero vector for any
    non-SMILES input. Neo4j rejects a zero-norm query vector, so such inputs are
    skipped gracefully — consistent with SMILES being the mandatory retrieval anchor
    (sector and condition-of-use are optional and embedded by their own text indices).

    Args:
        vector_store (Neo4jVector): The vector store to query.
        query (str): The text/SMILES to embed and search with.
        k (int): Number of nearest neighbours to retrieve.

    Returns:
        List[Tuple[Any, float]]: ``(document, score)`` pairs, empty if the query
            embeds to a zero-norm vector.
    """
    vector = vector_store.embedding.embed_query(query)
    if not any(vector):
        logger.warning(f"Query {query!r} has no valid embedding (zero-norm); returning no hits")
        return []
    return vector_store.similarity_search_with_score(query, k=k)


def dedup_categories_by_max_score(category_hits: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
    """Collapse candidate-derived category hits to one row per category.

    A category contains many member Candidates, several of which may match the query
    SMILES. Each match reaches the *same* category via ``<-[:CONTAINS]-``, so without
    deduplication a category is pulled in once per matching member and over-represented
    by sheer count. Keeping only the highest-scoring member makes a category as relevant
    as its structurally closest member, and retains *which* member was the entry point
    (for transparency — the category itself has no SMILES, so the score is the member's).

    Args:
        category_hits (List[Tuple[str, str, float]]): ``(category cas_number, entry-point
            member label, member score)`` triples from the candidate vector search.

    Returns:
        List[Tuple[str, str, float]]: One ``(category cas_number, entry-point member label,
            max score)`` per category, sorted by score descending.
    """
    best: Dict[str, Tuple[str, float]] = {}
    for category, member, score in category_hits:
        if category not in best or score > best[category][1]:
            best[category] = (member, score)
    return sorted(((cat, m, s) for cat, (m, s) in best.items()), key=lambda t: t[2], reverse=True)


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
                if embedding_type == "ActivityEmbeddings":
                    return ActivityEmbeddings(
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

    def get_candidate_vector_store(self, embedding_type: str) -> Neo4jVector:
        """Return a cached ``Neo4jVector`` over the ``Candidate`` index for a chemical embedding.

        Args:
            embedding_type (str): One of ``MorganFingerprint``, ``MolDeBERTa``, ``Hybrid``.

        Returns:
            Neo4jVector: Vector store over the matching ``candidate_*`` index, sharing the
                chemical embedder so query SMILES are embedded identically.
        """
        key = f"Candidate::{embedding_type}"
        if key not in self._vector_stores:
            cfg = _CANDIDATE_INDEX_MAP[embedding_type]
            self._vector_stores[key] = Neo4jVector.from_existing_index(
                embedding=self._build_embedder(embedding_type),
                url=self._url,
                username=self._user,
                password=self._password,
                index_name=cfg["index_name"],
                node_label="Candidate",
                embedding_node_property=cfg["embedding_property"],
                text_node_properties=["name"],
            )
            logger.info(f"Built Candidate Neo4jVector for embedding type '{embedding_type}'")
        return self._vector_stores[key]

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

    def _retrieve_activity_names(self, condition_of_use: str) -> List[str]:
        if not condition_of_use:
            return []
        vector_store = self.get_vector_store("ActivityEmbeddings")
        hits = vector_store.similarity_search_with_score(condition_of_use, k=self._top_k)
        names: List[str] = []
        for doc, _ in hits:
            content = str(doc.page_content)
            name = content.split(":", 1)[1].strip() if ":" in content else content.strip()
            if name:
                names.append(name)
        return names

    def _candidate_category_hits(self, chemical_smiles: str) -> List[Tuple[str, str, float]]:
        """Expand a query SMILES through the Candidate indices to relevant categories.

        Vector-searches the Candidate index, maps each matched member to its category
        Chemical via ``CONTAINS``, and collapses to one row per category at its highest
        member score (see ``dedup_categories_by_max_score``), keeping the entry-point member.

        Args:
            chemical_smiles (str): Query SMILES.

        Returns:
            List[Tuple[str, str, float]]: ``(category cas_number, entry-point member label,
                max member score)``, deduped per category.
        """
        if self._chem_embedding_type not in _CANDIDATE_INDEX_MAP:
            return []
        store = self.get_candidate_vector_store(self._chem_embedding_type)
        member_scores: List[Tuple[str, float]] = []
        for doc, score in guarded_search(store, chemical_smiles, self._top_k):
            cas = cast(Mapping[str, Any], doc.metadata).get("cas_number")
            if cas:
                member_scores.append((str(cas), float(score)))
        member_map = self.get_cross_retriever().candidate_categories([c for c, _ in member_scores])
        category_hits = [(member_map[c][0], f"{member_map[c][1]} (CAS {c})", s) for c, s in member_scores if c in member_map]
        return dedup_categories_by_max_score(category_hits)

    def retrieve_multi(
        self,
        chemical_smiles: str = "",
        sector_query: str = "",
        condition_of_use: str = "",
        include_candidates: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run parallel chemical and sector retrieval, intersecting via graph traversal.

        Chemical retrieval uses the configured chemical index (SMILES embedding;
        ``RAG_GRAPH.RETRIEVAL.CHEMICAL_EMBEDDING_TYPE``). Sector retrieval uses the
        NAICS title index (text embedding). When both return results, a cross-entity
        Cypher query finds reports at their intersection. Falls back to individual
        traversals if the intersection is empty. An optional condition-of-use string
        is semantically matched against the Activity index and used to restrict the
        graph traversal; an empty string disables that filter (default behavior).

        Args:
            chemical_smiles (str): SMILES string for chemical similarity search.
            sector_query (str): Sector description text for NAICS embedding search.
            condition_of_use (str): Optional condition-of-use text for Activity
                similarity search; restricts traversal to matching activities.
            include_candidates (bool): When True, also expand the query SMILES through the
                Candidate indices to reach no-SMILES category Chemicals via ``CONTAINS``
                (one row per category at its closest member's score). Default False keeps
                behavior identical for the validation harness.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: ``(retrieved_nodes, subgraphs)``.
        """
        retrieved_nodes: List[Dict[str, Any]] = []
        cas_numbers: List[str] = []
        naics_codes: List[str] = []
        activity_names: List[str] = self._retrieve_activity_names(condition_of_use)

        if chemical_smiles:
            vector_store = self.get_vector_store(self._chem_embedding_type)
            hits = guarded_search(vector_store, chemical_smiles, self._top_k)
            for doc, score in hits:
                hit = _vector_hit(doc, score)
                node = hit["node"]
                retrieved_nodes.append(hit)
                if "cas_number" in node:
                    cas_numbers.append(node["cas_number"])

        if include_candidates and chemical_smiles:
            for category_cas, entry_member, score in self._candidate_category_hits(chemical_smiles):
                if category_cas not in cas_numbers:
                    retrieved_nodes.append(
                        {"node": {"cas_number": category_cas, "via_candidate": entry_member}, "content": "", "score": score}
                    )
                    cas_numbers.append(category_cas)

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
            subgraphs = cross.traverse_cross(cas_numbers, naics_codes, activity_names)
            if not subgraphs:
                logger.warning("Cross-entity traversal empty; falling back to individual traversals")
                chem_nodes = _seed_nodes([r for r in retrieved_nodes if "cas_number" in r["node"]])
                sector_nodes = _seed_nodes([r for r in retrieved_nodes if "naics_code" in r["node"]])
                subgraphs = self.get_graph_retriever(self._chem_embedding_type).traverse_all(
                    chem_nodes, activity_names
                ) + self.get_graph_retriever("NAICSEmbeddings").traverse_all(sector_nodes, activity_names)
        elif cas_numbers:
            chem_nodes = _seed_nodes([r for r in retrieved_nodes if "cas_number" in r["node"]])
            subgraphs = self.get_graph_retriever(self._chem_embedding_type).traverse_all(chem_nodes, activity_names)
        elif naics_codes:
            sector_nodes = _seed_nodes([r for r in retrieved_nodes if "naics_code" in r["node"]])
            subgraphs = self.get_graph_retriever("NAICSEmbeddings").traverse_all(sector_nodes, activity_names)
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
        hits = guarded_search(vector_store, retrieval_query, self._top_k)
        retrieved_nodes = [_vector_hit(doc, score) for doc, score in hits]
        graph_ret = self.get_graph_retriever(embedding_type)
        subgraphs = graph_ret.traverse_all(_seed_nodes(retrieved_nodes))
        return retrieved_nodes, subgraphs
