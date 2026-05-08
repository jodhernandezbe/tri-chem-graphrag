# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Vector index management and embedding storage for the TRI Knowledge Graph.

Creates Neo4j vector indices on Chemical nodes (Morgan fingerprint, MolDeBERTa,
Hybrid) and Sector nodes (NAICS title), then populates the corresponding
embedding properties from transformed DataFrames.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-07
"""

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.config_manager import ConfigManager
from src.rag_graph.embedding import (
    HybridMolecularEmbeddings,
    MolDeBERTaEmbeddings,
    MorganFingerprintEmbeddings,
    NAICSEmbeddings,
)

from .base_loader import BATCH_SIZE, BaseLoader

logger = logging.getLogger(__name__)

# Embedding output dimensions
MORGAN_FP_DIMS: int = 2048  # MorganFingerprintEmbeddings(n_bits=2048) from config
MOLDEBERTA_DIMS: int = 768  # SaeedLab/MolDeBERTa-base-123M-mtr hidden size
HYBRID_DIMS: int = 1792  # HybridMolecularEmbeddings: Morgan(n_bits=1024) + MolDeBERTa(768)
NAICS_TITLE_DIMS: int = 384  # sentence-transformers/all-MiniLM-L6-v2

# Smaller batch for transformer inference to stay within GPU/CPU memory limits.
EMBEDDING_BATCH_SIZE: int = 100

_CHEMICAL_INDEX_SPECS: List[Tuple[str, str, int]] = [
    ("chemical_morgan_fp_index", "morgan_fp_embedding", MORGAN_FP_DIMS),
    ("chemical_moldeberta_index", "moldeberta_embedding", MOLDEBERTA_DIMS),
    ("chemical_hybrid_index", "hybrid_embedding", HYBRID_DIMS),
]
_SECTOR_INDEX_SPEC: Tuple[str, str, int] = ("sector_naics_title_index", "naics_title_embedding", NAICS_TITLE_DIMS)

_CONSTRAINTS: List[Tuple[str, str, str]] = [
    ("chemical_cas_unique", "Chemical", "cas_number"),
    ("facility_trifid_unique", "Facility", "trifid"),
    ("sector_naics_unique", "Sector", "naics_code"),
    ("report_dcn_unique", "Report", "document_control_number"),
    ("activity_name_unique", "Activity", "name"),
    ("reduction_code_unique", "Reduction", "code"),
]

_STORE_MORGAN_QUERY = f"""
UNWIND $rows AS row
CALL {{
    WITH row
    MATCH (c:Chemical {{cas_number: row.cas_number}})
    SET c.morgan_fp_embedding = row.embedding
}} IN TRANSACTIONS OF {BATCH_SIZE} ROWS
"""

_STORE_DEBERTA_QUERY = f"""
UNWIND $rows AS row
CALL {{
    WITH row
    MATCH (c:Chemical {{cas_number: row.cas_number}})
    SET c.moldeberta_embedding = row.embedding
}} IN TRANSACTIONS OF {BATCH_SIZE} ROWS
"""

_STORE_HYBRID_QUERY = f"""
UNWIND $rows AS row
CALL {{
    WITH row
    MATCH (c:Chemical {{cas_number: row.cas_number}})
    SET c.hybrid_embedding = row.embedding
}} IN TRANSACTIONS OF {BATCH_SIZE} ROWS
"""

_STORE_NAICS_QUERY = f"""
UNWIND $rows AS row
CALL {{
    WITH row
    MATCH (s:Sector {{naics_code: row.naics_code}})
    SET s.naics_title_embedding = row.embedding
}} IN TRANSACTIONS OF {BATCH_SIZE} ROWS
"""


class VectorIndexManager(BaseLoader):
    """Manages vector indices and embedding population for the TRI Knowledge Graph."""

    def __init__(self) -> None:
        super().__init__()
        rag_embeddings = ConfigManager().config["RAG_GRAPH"]["EMBEDDINGS"]
        self._morgan_params = self._load_morgan_params(rag_embeddings)
        self._moldeberta_params = self._load_transformer_params(rag_embeddings, "MolDeBERTa")
        self._naics_params = self._load_naics_params(rag_embeddings)

    def _load_morgan_params(self, cfg: List[Dict[str, Any]]) -> Dict[str, int]:
        for emb in cfg:
            if emb["EMBEDDING_TYPE"] == "MorganFingerprint":
                props = emb["PROPERTIES"][0]
                return {"radius": props["RADIUS"], "n_bits": props["N_BITS"]}
        return {"radius": 2, "n_bits": 2048}

    def _load_transformer_params(self, cfg: List[Dict[str, Any]], emb_type: str) -> Dict[str, Any]:
        for emb in cfg:
            if emb["EMBEDDING_TYPE"] == emb_type:
                props = emb["PROPERTIES"][0]
                return {"model_name": str(props["MODEL_NAME"]), "max_length": int(props.get("MAX_LENGTH", 512))}
        return {"model_name": "", "max_length": 512}

    def _load_naics_params(self, cfg: List[Dict[str, Any]]) -> Dict[str, Any]:
        for emb in cfg:
            if emb["EMBEDDING_TYPE"] == "NAICSEmbeddings":
                props = emb["PROPERTIES"][0]
                return {"model_name": props["MODEL_NAME"], "max_length": props.get("MAX_LENGTH", 512)}
        return {"model_name": "sentence-transformers/all-MiniLM-L6-v2", "max_length": 512}

    def _execute_schema(self, query: str) -> None:
        with self._connector.session() as session:
            session.run(query)  # type: ignore[arg-type]

    def _create_constraint(self, name: str, label: str, prop: str) -> None:
        query = f"CREATE CONSTRAINT {name} IF NOT EXISTS " f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
        self._execute_schema(query)
        logger.info(f"Ensured constraint: {name}")

    def _create_vector_index(self, index_name: str, label: str, prop: str, dims: int) -> None:
        query = (
            f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.{prop}) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dims}, "
            f"`vector.similarity_function`: 'cosine'}}}}"
        )
        self._execute_schema(query)
        logger.info(f"Ensured vector index: {index_name} ({dims} dims, cosine)")

    def _embed_and_store(
        self,
        records: List[Dict[str, Any]],
        model: Any,
        text_col: str,
        key_col: str,
        store_query: str,
    ) -> int:
        total = 0
        for i in range(0, len(records), EMBEDDING_BATCH_SIZE):
            batch = records[i : i + EMBEDDING_BATCH_SIZE]
            texts = [str(r.get(text_col) or "") for r in batch]
            embeddings = model.embed_documents(texts)
            store_records = [{key_col: r[key_col], "embedding": emb} for r, emb in zip(batch, embeddings)]
            self._bulk_write(store_query, store_records)
            total += len(batch)
            logger.debug(f"Embedding progress: {total}/{len(records)}")
        return total

    def _prepare_chemical_records(self, df: pd.DataFrame, require_smiles: bool) -> List[Dict[str, Any]]:
        subset = df[["cas_number", "canonical_smiles"]].drop_duplicates(subset=["cas_number"]).dropna(subset=["cas_number"])  # type: ignore[call-overload]
        if require_smiles:
            subset = subset.dropna(subset=["canonical_smiles"])  # type: ignore[call-overload]
        return [self._sanitize_record(r) for r in subset.to_dict("records")]

    def create_constraints(self) -> None:
        """Create uniqueness constraints for all node types (idempotent)."""
        for name, label, prop in _CONSTRAINTS:
            self._create_constraint(name, label, prop)

    def create_all_indices(self) -> None:
        """Create all vector indices for Chemical (×3) and Sector (×1) nodes (idempotent)."""
        for index_name, prop, dims in _CHEMICAL_INDEX_SPECS:
            self._create_vector_index(index_name, "Chemical", prop, dims)
        idx_name, prop, dims = _SECTOR_INDEX_SPEC
        self._create_vector_index(idx_name, "Sector", prop, dims)

    def setup_schema(self) -> None:
        """Create uniqueness constraints and all vector indices in one call (idempotent).

        Call this once before loading data to ensure the schema is in place.
        """
        self.create_constraints()
        self.create_all_indices()

    def embed_chemicals(self, df: pd.DataFrame) -> None:
        """Compute and store Morgan FP, MolDeBERTa, and Hybrid embeddings for Chemical nodes.

        Morgan fingerprints are computed for all chemicals; transformer-based
        embeddings are restricted to chemicals with a valid canonical_smiles.

        Args:
            df (pd.DataFrame): DataFrame with columns cas_number and canonical_smiles.
        """
        all_chems = self._prepare_chemical_records(df, require_smiles=False)
        smiles_chems = self._prepare_chemical_records(df, require_smiles=True)

        morgan = MorganFingerprintEmbeddings(**self._morgan_params)
        deberta = MolDeBERTaEmbeddings(**self._moldeberta_params)
        hybrid = HybridMolecularEmbeddings(**self._moldeberta_params)

        n_morgan = self._embed_and_store(all_chems, morgan, "canonical_smiles", "cas_number", _STORE_MORGAN_QUERY)
        n_deberta = self._embed_and_store(smiles_chems, deberta, "canonical_smiles", "cas_number", _STORE_DEBERTA_QUERY)
        n_hybrid = self._embed_and_store(smiles_chems, hybrid, "canonical_smiles", "cas_number", _STORE_HYBRID_QUERY)
        logger.info(f"Chemical embeddings stored — morgan: {n_morgan}, moldeberta: {n_deberta}, hybrid: {n_hybrid}")

    def embed_sectors(self, df: pd.DataFrame) -> None:
        """Compute and store NAICS title embeddings for Sector nodes.

        Args:
            df (pd.DataFrame): DataFrame with columns naics_code and naics_title.
        """
        subset = (
            df[["naics_code", "naics_title"]]
            .drop_duplicates(subset=["naics_code"])  # type: ignore[call-overload]
            .dropna(subset=["naics_code", "naics_title"])  # type: ignore[call-overload]
        )
        records = [self._sanitize_record(r) for r in subset.to_dict("records")]
        model = NAICSEmbeddings(**self._naics_params)
        count = self._embed_and_store(records, model, "naics_title", "naics_code", _STORE_NAICS_QUERY)
        logger.info(f"Stored NAICS title embeddings for {count} Sector nodes")
