# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Edge loading module for the TRI Knowledge Graph.

Bulk-upserts all five relationship types (CLASSIFIED_AS, REPORTS_ON, FILED_BY,
RECORDS, DOCUMENTS) into Neo4j using CALL { } IN TRANSACTIONS for periodic commits.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-07
"""

import logging

import pandas as pd

from .base_loader import BATCH_SIZE, BaseLoader

logger = logging.getLogger(__name__)

_CLASSIFIED_AS_QUERY = f"""
UNWIND $rows AS row
CALL {{
    WITH row
    MATCH (f:Facility {{trifid: row.trifid}})
    MATCH (s:Sector {{naics_code: row.naics_code}})
    MERGE (f)-[:CLASSIFIED_AS]->(s)
}} IN TRANSACTIONS OF {BATCH_SIZE} ROWS
"""

_REPORTS_ON_QUERY = f"""
UNWIND $rows AS row
CALL {{
    WITH row
    MATCH (r:Report {{document_control_number: row.document_control_number}})
    MATCH (c:Chemical {{cas_number: row.cas_number}})
    MERGE (r)-[rel:REPORTS_ON]->(c)
    SET rel += {{maximum_amount: row.maximum_amount}}
}} IN TRANSACTIONS OF {BATCH_SIZE} ROWS
"""

_FILED_BY_QUERY = f"""
UNWIND $rows AS row
CALL {{
    WITH row
    MATCH (r:Report {{document_control_number: row.document_control_number}})
    MATCH (f:Facility {{trifid: row.trifid}})
    MERGE (r)-[:FILED_BY]->(f)
}} IN TRANSACTIONS OF {BATCH_SIZE} ROWS
"""

# Traverses FILED_BY to reach the Facility without requiring trifid in the 1b data.
_RECORDS_QUERY = f"""
UNWIND $rows AS row
CALL {{
    WITH row
    MATCH (r:Report {{document_control_number: row.document_control_number}})-[:FILED_BY]->(f:Facility)
    MATCH (a:Activity {{name: row.chemical_activity}})
    MERGE (f)-[:RECORDS]->(a)
}} IN TRANSACTIONS OF {BATCH_SIZE} ROWS
"""

_DOCUMENTS_QUERY = f"""
UNWIND $rows AS row
CALL {{
    WITH row
    MATCH (r:Report {{document_control_number: row.document_control_number}})
    MATCH (rd:Reduction {{code: row.source_reduction_code}})
    MERGE (r)-[rel:DOCUMENTS]->(rd)
    SET rel += {{annual_reduction: row.annual_reduction}}
}} IN TRANSACTIONS OF {BATCH_SIZE} ROWS
"""


class EdgeLoader(BaseLoader):
    """Bulk-upserts all Knowledge Graph relationships into Neo4j."""

    def load_classified_as(self, df: pd.DataFrame) -> int:
        """Bulk-upsert CLASSIFIED_AS edges (Facility → Sector).

        Args:
            df (pd.DataFrame): DataFrame with columns trifid, naics_code.

        Returns:
            int: Number of records submitted.
        """
        records = self._to_records(df, ["trifid", "naics_code"], key_cols=["trifid", "naics_code"])
        count = self._bulk_write(_CLASSIFIED_AS_QUERY, records)
        logger.info(f"Bulk-upserted {count} CLASSIFIED_AS relationships")
        return count

    def load_reports_on(self, df: pd.DataFrame) -> int:
        """Bulk-upsert REPORTS_ON edges (Report → Chemical).

        Args:
            df (pd.DataFrame): DataFrame with columns document_control_number,
                cas_number, maximum_amount.

        Returns:
            int: Number of records submitted.
        """
        records = self._to_records(
            df,
            ["document_control_number", "cas_number", "maximum_amount"],
            key_cols=["document_control_number", "cas_number"],
        )
        count = self._bulk_write(_REPORTS_ON_QUERY, records)
        logger.info(f"Bulk-upserted {count} REPORTS_ON relationships")
        return count

    def load_filed_by(self, df: pd.DataFrame) -> int:
        """Bulk-upsert FILED_BY edges (Report → Facility).

        Args:
            df (pd.DataFrame): DataFrame with columns document_control_number, trifid.

        Returns:
            int: Number of records submitted.
        """
        records = self._to_records(
            df,
            ["document_control_number", "trifid"],
            key_cols=["document_control_number", "trifid"],
        )
        count = self._bulk_write(_FILED_BY_QUERY, records)
        logger.info(f"Bulk-upserted {count} FILED_BY relationships")
        return count

    def load_records(self, df: pd.DataFrame) -> int:
        """Bulk-upsert RECORDS edges (Facility → Activity).

        Resolves the Facility via an existing FILED_BY relationship, so
        FILED_BY edges must be loaded before calling this method.

        Args:
            df (pd.DataFrame): DataFrame with columns document_control_number,
                chemical_activity (from 1b transformer).

        Returns:
            int: Number of records submitted.
        """
        records = self._to_records(
            df,
            ["document_control_number", "chemical_activity"],
            key_cols=["document_control_number", "chemical_activity"],
        )
        records = [r for r in records if r.get("chemical_activity")]
        count = self._bulk_write(_RECORDS_QUERY, records)
        logger.info(f"Bulk-upserted {count} RECORDS relationships")
        return count

    def load_documents(self, df: pd.DataFrame) -> int:
        """Bulk-upsert DOCUMENTS edges (Report → Reduction).

        Args:
            df (pd.DataFrame): DataFrame with columns document_control_number,
                source_reduction_code, annual_reduction (from 2a transformer).

        Returns:
            int: Number of records submitted.
        """
        records = self._to_records(
            df,
            ["document_control_number", "source_reduction_code", "annual_reduction"],
            key_cols=["document_control_number", "source_reduction_code"],
        )
        records = [r for r in records if r.get("source_reduction_code")]
        count = self._bulk_write(_DOCUMENTS_QUERY, records)
        logger.info(f"Bulk-upserted {count} DOCUMENTS relationships")
        return count
