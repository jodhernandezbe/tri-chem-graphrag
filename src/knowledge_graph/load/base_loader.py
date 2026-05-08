# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Base loader with shared Neo4j bulk write utilities.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-07
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from src.neo4j_connector import Neo4jConnector

# Number of rows Neo4j commits per periodic transaction inside CALL { } IN TRANSACTIONS.
BATCH_SIZE: int = 500

logger = logging.getLogger(__name__)


class BaseLoader:
    """Base class providing Neo4j bulk write utilities for all loaders."""

    def __init__(self) -> None:
        self._connector = Neo4jConnector()

    @staticmethod
    def _is_na(value: Any) -> bool:
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    def _sanitize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {k: (None if self._is_na(v) else v) for k, v in record.items()}

    def _to_records(
        self,
        df: pd.DataFrame,
        columns: List[str],
        key_cols: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if df.empty:
            return []
        available = [c for c in columns if c in df.columns]
        if not available:
            return []
        required = key_cols if key_cols else [available[0]]
        present_keys: List[Any] = [k for k in required if k in available]
        subset = df[available].drop_duplicates().dropna(subset=present_keys)  # type: ignore[call-overload]
        return [self._sanitize_record(r) for r in subset.to_dict("records")]

    def _bulk_write(self, query: str, records: List[Dict[str, Any]]) -> int:
        """Pass all records to Neo4j and execute as a bulk upsert.

        Uses CALL { } IN TRANSACTIONS internally (queries must include that
        clause). Requires an auto-commit (implicit) session — NOT execute_write.

        Args:
            query (str): Cypher query using UNWIND $rows + CALL { } IN TRANSACTIONS.
            records (List[Dict[str, Any]]): All records to write.

        Returns:
            int: Number of records submitted.
        """
        if not records:
            return 0
        with self._connector.session() as session:
            session.run(query, {"rows": records})  # type: ignore[arg-type]
        logger.debug(f"Bulk wrote {len(records)} records")
        return len(records)
