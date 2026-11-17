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
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple, cast

from langchain_neo4j import Neo4jVector

from src.config_manager import ConfigManager
from src.rag_graph.embedding import (
    ActivityEmbeddings,
    HybridMolecularEmbeddings,
    MolDeBERTaEmbeddings,
    MorganFingerprintEmbeddings,
    NAICSEmbeddings,
)

from .context_profiles import ContextProfiles, Profile, rerank_by_context
from .fused_retriever import (
    CHEMICAL_POOL,
    F1,
    FusedOptions,
    FusedResult,
    FusedRetriever,
    ReductionRecord,
)
from .graph_retriever import CrossEntityRetriever, GraphRetriever
from .switches import (
    ANALOGUES_CONTEXT,
    CONTEXT_POOL_SIZE,
    COU_SOFT,
    idf_from_counts,
    resolve_switch,
)

logger = logging.getLogger(__name__)

CHEMICAL_EMBEDDING_TYPES = ("MorganFingerprint", "MolDeBERTa", "Hybrid")

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


# A vector hit at or above this score is the query structure itself (same constant as the pipeline's QUERY_TAG).
_IDENTICAL_SCORE = 0.99999


def has_records(subgraphs: List[Dict[str, Any]]) -> bool:
    """Return True when any chemical subgraph carries at least one reduction report.

    Args:
        subgraphs (List[Dict[str, Any]]): Traversed subgraph dicts.

    Returns:
        bool: Whether the traversal produced records that ``tabulate_records`` would use.
    """
    return any(sg.get("node_type") in ("Chemical", "Chemical_Sector") and sg.get("reports") for sg in subgraphs)


@dataclass
class RetrievalResult:
    """Outcome of ``RetrieverFactory.retrieve_multi_result``.

    Attributes:
        retrieved_nodes (List[Dict[str, Any]]): Vector hits (chemical, then sector).
        subgraphs (List[Dict[str, Any]]): Traversed subgraphs.
        sector_fallback (bool): A sector was given but the chemical-and-sector intersection was empty.
        cou_fallback (bool): ``cou_filter=soft`` and the condition-of-use filter returned no records,
            so the traversal was repeated without it.
    """

    retrieved_nodes: List[Dict[str, Any]]
    subgraphs: List[Dict[str, Any]]
    sector_fallback: bool = False
    cou_fallback: bool = False


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


def _activity_name(content: str) -> str:
    return content.split(":", 1)[1].strip() if ":" in content else content.strip()


def _naics_title(content: str) -> str:
    return content.split("naics_title:", 1)[1].strip() if "naics_title:" in content else ""


def _sector_key(code: Any) -> str:
    return str(int(float(code))) if code not in (None, "") else ""


def _to_record(row: Mapping[str, Any]) -> ReductionRecord:
    return ReductionRecord(
        report_id=str(row["report_id"]),
        sort_key=str(row.get("dcn") or ""),
        cas=str(row.get("cas") or ""),
        chemical=str(row.get("chemical") or ""),
        code=str(row["code"]),
        name=str(row.get("name") or ""),
        uses=tuple(sorted(str(u) for u in (row.get("uses") or []) if u)),
        sector_code=_sector_key(row.get("sector_code")),
        sector_title=str(row.get("sector_title") or ""),
        year=row.get("year"),
        state=row.get("state"),
        annual_reduction=row.get("annual_reduction"),
    )


class _FactorySource:
    """Adapts a ``RetrieverFactory`` to the ``FusedSource`` protocol (read-only)."""

    def __init__(
        self,
        factory: "RetrieverFactory",
        include_candidates: bool = True,
        embedding_type: Optional[str] = None,
    ) -> None:
        self._factory = factory
        self._include_candidates = include_candidates
        self._embedding_type = factory.resolve_chemical_embedding(embedding_type)
        self._cache: Dict[Any, Any] = {}
        self._sector_titles: Dict[str, str] = {}

    def chemical_hits(self, smiles: str, exclude_cas: FrozenSet[str]) -> List[Tuple[str, float, str]]:
        key = ("chem", self._embedding_type, smiles, exclude_cas)
        if key not in self._cache:
            self._cache[key] = self._chemical_hits(smiles, exclude_cas)
        return self._cache[key]

    def _chemical_hits(self, smiles: str, exclude_cas: FrozenSet[str]) -> List[Tuple[str, float, str]]:
        store = self._factory.get_vector_store(self._embedding_type)
        hits = guarded_search(store, smiles, CHEMICAL_POOL + len(exclude_cas))
        pool = [(str(d.metadata["cas_number"]), float(s), "") for d, s in hits if d.metadata.get("cas_number")]
        pool = [h for h in pool if h[0] not in exclude_cas][:CHEMICAL_POOL]
        if self._include_candidates:
            seen = {h[0] for h in pool}
            pool += [
                h
                for h in self._factory.candidate_hits(smiles, self._embedding_type)
                if h[0] not in seen and h[0] not in exclude_cas
            ]
        return pool

    def activity_hits(self, text: str, k: int) -> Dict[str, float]:
        store = self._factory.get_vector_store("ActivityEmbeddings")
        scored = [(_activity_name(str(d.page_content)), float(s)) for d, s in store.similarity_search_with_score(text, k=k)]
        return {name: score for name, score in scored if name}

    def sector_hits(self, text: str, k: int) -> Dict[str, float]:
        store = self._factory.get_vector_store("NAICSEmbeddings")
        hits = store.similarity_search_with_score(text, k=k)
        scored = [(_sector_key(d.metadata.get("naics_code")), float(s), _naics_title(str(d.page_content))) for d, s in hits]
        self._sector_titles.update({code: title for code, _, title in scored if code and title})
        return {code: score for code, score, _ in scored if code}

    def sector_titles(self, codes: Iterable[str]) -> Dict[str, str]:
        return {c: self._sector_titles[c] for c in codes if c in self._sector_titles}

    def reduction_records(self, report_split: Optional[str]) -> List[ReductionRecord]:
        if report_split not in self._cache:
            rows = self._factory.get_cross_retriever().reduction_records(report_split)
            self._cache[report_split] = [_to_record(r) for r in rows]
        return self._cache[report_split]


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

    def _candidate_category_hits(
        self,
        chemical_smiles: str,
        chemical_embedding: Optional[str] = None,
    ) -> List[Tuple[str, str, float]]:
        """Expand a query SMILES through the Candidate indices to relevant categories.

        Vector-searches the Candidate index, maps each matched member to its category
        Chemical via ``CONTAINS``, and collapses to one row per category at its highest
        member score (see ``dedup_categories_by_max_score``), keeping the entry-point member.

        Args:
            chemical_smiles (str): Query SMILES.
            chemical_embedding (Optional[str]): Chemical embedding type for this call; ``None`` uses the configured one.

        Returns:
            List[Tuple[str, str, float]]: ``(category cas_number, entry-point member label,
                max member score)``, deduped per category.
        """
        embedding = self.resolve_chemical_embedding(chemical_embedding)
        if embedding not in _CANDIDATE_INDEX_MAP:
            return []
        store = self.get_candidate_vector_store(embedding)
        member_scores: List[Tuple[str, float]] = []
        for doc, score in guarded_search(store, chemical_smiles, self._top_k):
            cas = cast(Mapping[str, Any], doc.metadata).get("cas_number")
            if cas:
                member_scores.append((str(cas), float(score)))
        member_map = self.get_cross_retriever().candidate_categories([c for c, _ in member_scores])
        category_hits = [(member_map[c][0], f"{member_map[c][1]} (CAS {c})", s) for c, s in member_scores if c in member_map]
        return dedup_categories_by_max_score(category_hits)

    @property
    def chemical_embedding_type(self) -> str:
        """The configured chemical embedding type (``RAG_GRAPH.RETRIEVAL.CHEMICAL_EMBEDDING_TYPE``)."""
        return self._chem_embedding_type

    def resolve_chemical_embedding(self, embedding_type: Optional[str] = None) -> str:
        """Return the chemical embedding type for one call: the override, else the configured one.

        Args:
            embedding_type (Optional[str]): Per-call override (``MorganFingerprint``, ``MolDeBERTa`` or ``Hybrid``).

        Returns:
            str: The validated embedding type.

        Raises:
            ValueError: If the override is not a chemical embedding type.
        """
        if embedding_type is None:
            return self._chem_embedding_type
        if embedding_type not in CHEMICAL_EMBEDDING_TYPES:
            logger.error(f"Invalid chemical embedding {embedding_type!r}; allowed: {CHEMICAL_EMBEDDING_TYPES}")
            raise ValueError(f"Invalid chemical embedding {embedding_type!r}; allowed: {CHEMICAL_EMBEDDING_TYPES}")
        return embedding_type

    def candidate_hits(self, chemical_smiles: str, chemical_embedding: Optional[str] = None) -> List[Tuple[str, float, str]]:
        """Candidate-category hits for a SMILES as ``(category cas, score, entry-point label)``.

        Args:
            chemical_smiles (str): Query SMILES.
            chemical_embedding (Optional[str]): Chemical embedding type for this call; ``None`` uses the configured one.

        Returns:
            List[Tuple[str, float, str]]: One row per category at its closest member's score.
        """
        hits = self._candidate_category_hits(chemical_smiles, chemical_embedding)
        return [(cat, score, member) for cat, member, score in hits]

    def retrieve_fused(
        self,
        chemical_smiles: str = "",
        sector_query: str = "",
        condition_of_use: str = "",
        report_split: Optional[str] = None,
        options: FusedOptions = F1,
        exclude_report_ids: FrozenSet[str] = frozenset(),
        exclude_cas: FrozenSet[str] = frozenset(),
        limit: int = 10,
        include_candidates: bool = True,
        chemical_embedding: Optional[str] = None,
    ) -> FusedResult:
        """Fused contextual retrieval (``retrieval=fused``); see ``fused_retriever`` for the channels.

        Args:
            chemical_smiles (str): Query SMILES; the chemical channel is skipped when empty.
            sector_query (str): Sector text; the sector channel is skipped when empty.
            condition_of_use (str): Use text; the use channel is skipped when empty.
            report_split (Optional[str]): Read only reports with this split (``'index'`` when evaluating).
            options (FusedOptions): Variant settings (F1, F2 or F3).
            exclude_report_ids (FrozenSet[str]): Reports hidden from every channel.
            exclude_cas (FrozenSet[str]): Chemicals left out of the chemical pool.
            limit (int): Number of de-duplicated records to return.
            include_candidates (bool): Expand the chemical pool through the Candidate indices.
            chemical_embedding (Optional[str]): Chemical embedding type for this call; ``None`` uses the configured one.

        Returns:
            FusedResult: Ranked records with ``Tier`` and ``Channels`` and the unmatched-use flag.
        """
        embedding = self.resolve_chemical_embedding(chemical_embedding)
        key = ("source", include_candidates, embedding)
        cache: Dict[Any, _FactorySource] = self.__dict__.setdefault("_fused_sources", {})
        if key not in cache:
            cache[key] = _FactorySource(self, include_candidates, embedding)
        retriever = FusedRetriever(cache[key])
        return retriever.retrieve(
            chemical_smiles, sector_query, condition_of_use, report_split, options, exclude_report_ids, exclude_cas, limit
        )

    def context_profiles(self, report_split: Optional[str] = None) -> ContextProfiles:
        """Return cached sector and condition-of-use profiles built from the reports visible under a split.

        Args:
            report_split (Optional[str]): Build from reports with this ``split`` only (``'index'`` when
                evaluating); ``None`` uses all reports.

        Returns:
            ContextProfiles: Profile store for context similarity.
        """
        cache: Dict[Optional[str], ContextProfiles] = self.__dict__.setdefault("_context_cache", {})
        if report_split not in cache:
            sector_rows, cou_rows = self.get_cross_retriever().context_rows(report_split)
            cache[report_split] = ContextProfiles.from_rows(sector_rows, cou_rows)
            logger.info(f"Built context profiles (report_split={report_split!r})")
        return cache[report_split]

    def code_idf(self, report_split: Optional[str] = None) -> Dict[str, float]:
        """Return cached IDF(c) = log(|C| / |{q : c in A_q}|) over the reports visible under a split.

        Args:
            report_split (Optional[str]): Use reports with this ``split`` only (``'index'`` when
                evaluating, so held-out reports cannot leak into the weights); ``None`` uses all.

        Returns:
            Dict[str, float]: Activity code to IDF.
        """
        cache: Dict[Optional[str], Dict[str, float]] = self.__dict__.setdefault("_idf_cache", {})
        if report_split not in cache:
            per_code, total = self.get_cross_retriever().code_profiles(report_split)
            cache[report_split] = idf_from_counts(per_code, total)
        return cache[report_split]

    def chemical_neighbours(
        self,
        chemical_smiles: str,
        analogues: Optional[str] = None,
        report_split: Optional[str] = None,
        query_cas: Optional[str] = None,
        query_profile: Optional[Profile] = None,
        profile_overrides: Optional[Mapping[str, Profile]] = None,
    ) -> List[Dict[str, Any]]:
        """Return the analogue chemicals (vector hits) for a SMILES.

        ``molecular`` returns the top ``TOP_K`` by the configured embedding. ``context`` takes the top
        ``CONTEXT_POOL_SIZE`` by the same embedding and keeps the ``TOP_K`` with the highest sector and
        condition-of-use similarity to the query, using only reports visible under ``report_split``. If the
        query has no visible profile the molecular order is kept.

        Args:
            chemical_smiles (str): Query SMILES.
            analogues (Optional[str]): ``molecular`` or ``context``; ``None`` reads config.
            report_split (Optional[str]): Reports visible to the profiles (``'index'`` when evaluating).
            query_cas (Optional[str]): Query chemical CAS; default is the identical-structure hit, if any.
            query_profile (Optional[Profile]): Explicit query profile (e.g. with hold-out reports removed).
            profile_overrides (Optional[Mapping[str, Profile]]): Replacement profiles for given CAS numbers.

        Returns:
            List[Dict[str, Any]]: Hits with ``node``, ``content``, ``score`` (molecular) and, for ``context``
                with a visible query profile, ``context_similarity``.
        """
        mode = resolve_switch("analogues", analogues)
        size = CONTEXT_POOL_SIZE if mode == ANALOGUES_CONTEXT else self._top_k
        vector_store = self.get_vector_store(self._chem_embedding_type)
        hits = [_vector_hit(doc, score) for doc, score in guarded_search(vector_store, chemical_smiles, size)]
        if mode != ANALOGUES_CONTEXT:
            return hits
        store = self.context_profiles(report_split)
        if query_profile is None:
            cas = query_cas or next((h["node"].get("cas_number") for h in hits if h["score"] >= _IDENTICAL_SCORE), None)
            query_profile = store.profile(cas) if cas else ({}, {})
        return rerank_by_context(hits, store, query_profile, self._top_k, profile_overrides)

    def _traverse(
        self,
        retrieved_nodes: List[Dict[str, Any]],
        activity_names: List[str],
        report_split: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Traverse from the retrieved chemical and sector seeds; return ``(subgraphs, sector_fallback)``."""
        cas_numbers = [r["node"]["cas_number"] for r in retrieved_nodes if "cas_number" in r["node"]]
        naics_codes = [r["node"]["naics_code"] for r in retrieved_nodes if "naics_code" in r["node"]]
        chem_nodes = _seed_nodes([r for r in retrieved_nodes if "cas_number" in r["node"]])
        sector_nodes = _seed_nodes([r for r in retrieved_nodes if "naics_code" in r["node"]])
        if cas_numbers and naics_codes:
            subgraphs = self.get_cross_retriever().traverse_cross(cas_numbers, naics_codes, activity_names, report_split)
            if subgraphs:
                return subgraphs, False
            logger.warning("Cross-entity traversal empty; falling back to individual traversals")
            return (
                self.get_graph_retriever(self._chem_embedding_type).traverse_all(chem_nodes, activity_names, report_split)
                + self.get_graph_retriever("NAICSEmbeddings").traverse_all(sector_nodes, activity_names, report_split),
                True,
            )
        if cas_numbers:
            return (
                self.get_graph_retriever(self._chem_embedding_type).traverse_all(chem_nodes, activity_names, report_split),
                False,
            )
        if naics_codes:
            return self.get_graph_retriever("NAICSEmbeddings").traverse_all(sector_nodes, activity_names, report_split), False
        return [], False

    def retrieve_multi_result(
        self,
        chemical_smiles: str = "",
        sector_query: str = "",
        condition_of_use: str = "",
        include_candidates: bool = False,
        report_split: Optional[str] = None,
        cou_filter: Optional[str] = None,
        analogues: Optional[str] = None,
        query_cas: Optional[str] = None,
    ) -> RetrievalResult:
        """Run parallel chemical and sector retrieval, intersecting via graph traversal.

        Chemical retrieval uses the configured chemical index (SMILES embedding;
        ``RAG_GRAPH.RETRIEVAL.CHEMICAL_EMBEDDING_TYPE``). Sector retrieval uses the
        NAICS title index (text embedding). When both return results, a cross-entity
        Cypher query finds reports at their intersection, falling back to individual
        traversals if it is empty. A condition-of-use string is matched against the Activity
        index and restricts the traversal (``cou_filter=hard``); with ``soft`` the traversal is
        repeated without that restriction if it returned no records, and ``cou_fallback`` is set.
        An empty condition of use disables the filter.

        Args:
            chemical_smiles (str): SMILES string for chemical similarity search.
            sector_query (str): Sector description text for NAICS embedding search.
            condition_of_use (str): Optional condition-of-use text for Activity similarity search.
            include_candidates (bool): Also expand the query SMILES through the Candidate indices to reach
                no-SMILES category Chemicals via ``CONTAINS`` (one row per category at its closest member's score).
            report_split (Optional[str]): Every traversal (and the context profiles) reads only reports whose
                ``split`` equals it (the evaluator passes ``'index'``). ``None`` reads all reports.
            cou_filter (Optional[str]): ``hard`` or ``soft``; ``None`` reads config (default ``hard``).
            analogues (Optional[str]): ``molecular`` or ``context``; ``None`` reads config (default ``molecular``).
            query_cas (Optional[str]): Query chemical CAS, used by ``analogues=context`` to find its profile.

        Returns:
            RetrievalResult: Retrieved nodes, subgraphs, and the sector and condition-of-use fallback flags.
        """
        cou_mode = resolve_switch("cou_filter", cou_filter)
        activity_names: List[str] = self._retrieve_activity_names(condition_of_use)
        retrieved_nodes: List[Dict[str, Any]] = []

        if chemical_smiles:
            retrieved_nodes.extend(self.chemical_neighbours(chemical_smiles, analogues, report_split, query_cas))

        if include_candidates and chemical_smiles:
            seen = {r["node"]["cas_number"] for r in retrieved_nodes if "cas_number" in r["node"]}
            for category_cas, entry_member, score in self._candidate_category_hits(chemical_smiles):
                if category_cas not in seen:
                    retrieved_nodes.append(
                        {"node": {"cas_number": category_cas, "via_candidate": entry_member}, "content": "", "score": score}
                    )
                    seen.add(category_cas)

        if sector_query:
            vector_store = self.get_vector_store("NAICSEmbeddings")
            for doc, score in vector_store.similarity_search_with_score(sector_query, k=self._top_k):
                retrieved_nodes.append(_vector_hit(doc, score))

        subgraphs, sector_fallback = self._traverse(retrieved_nodes, activity_names, report_split)
        cou_fallback = False
        if cou_mode == COU_SOFT and activity_names and not has_records(subgraphs):
            logger.warning("No records under the condition-of-use filter; repeating without it (cou_filter=soft)")
            subgraphs, sector_fallback = self._traverse(retrieved_nodes, [], report_split)
            cou_fallback = True
        return RetrievalResult(retrieved_nodes, subgraphs, sector_fallback, cou_fallback)

    def retrieve_multi(
        self,
        chemical_smiles: str = "",
        sector_query: str = "",
        condition_of_use: str = "",
        include_candidates: bool = False,
        report_split: Optional[str] = None,
        cou_filter: Optional[str] = None,
        analogues: Optional[str] = None,
        query_cas: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run ``retrieve_multi_result`` and return only ``(retrieved_nodes, subgraphs)``.

        Kept 2-tuple for the validation harness; see ``retrieve_multi_result`` for semantics.

        Args:
            chemical_smiles (str): SMILES string for chemical similarity search.
            sector_query (str): Sector description text for NAICS embedding search.
            condition_of_use (str): Optional condition-of-use text for Activity similarity search.
            include_candidates (bool): Also expand the query through the Candidate indices.
            report_split (Optional[str]): Read only reports with this split (e.g. ``'index'``); ``None`` reads all.
            cou_filter (Optional[str]): ``hard`` or ``soft``; ``None`` reads config.
            analogues (Optional[str]): ``molecular`` or ``context``; ``None`` reads config.
            query_cas (Optional[str]): Query chemical CAS for ``analogues=context``.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: ``(retrieved_nodes, subgraphs)``.
        """
        res = self.retrieve_multi_result(
            chemical_smiles, sector_query, condition_of_use, include_candidates, report_split, cou_filter, analogues, query_cas
        )
        return res.retrieved_nodes, res.subgraphs

    def retrieve_multi_detailed(
        self,
        chemical_smiles: str = "",
        sector_query: str = "",
        condition_of_use: str = "",
        include_candidates: bool = False,
        report_split: Optional[str] = None,
        cou_filter: Optional[str] = None,
        analogues: Optional[str] = None,
        query_cas: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
        """Run ``retrieve_multi_result`` and return ``(retrieved_nodes, subgraphs, sector_fallback)``.

        Args:
            chemical_smiles (str): SMILES string for chemical similarity search.
            sector_query (str): Sector description text for NAICS embedding search.
            condition_of_use (str): Optional condition-of-use text for Activity similarity search.
            include_candidates (bool): Also expand the query through the Candidate indices.
            report_split (Optional[str]): Read only reports with this split; ``None`` reads all.
            cou_filter (Optional[str]): ``hard`` or ``soft``; ``None`` reads config.
            analogues (Optional[str]): ``molecular`` or ``context``; ``None`` reads config.
            query_cas (Optional[str]): Query chemical CAS for ``analogues=context``.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]: ``(retrieved_nodes, subgraphs,
                sector_fallback)``; the soft condition-of-use flag is only on ``retrieve_multi_result``.
        """
        res = self.retrieve_multi_result(
            chemical_smiles, sector_query, condition_of_use, include_candidates, report_split, cou_filter, analogues, query_cas
        )
        return res.retrieved_nodes, res.subgraphs, res.sector_fallback

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
