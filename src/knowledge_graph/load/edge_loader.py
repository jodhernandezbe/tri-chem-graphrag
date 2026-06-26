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

# Per the canonical ontology (graph_model.svg), RECORDS originates from the Report,
# which co-locates the condition-of-use Activity with the report's chemical and reduction.
_RECORDS_QUERY = f"""
UNWIND $rows AS row
CALL {{
    WITH row
    MATCH (r:Report {{document_control_number: row.document_control_number}})
    MATCH (a:Activity {{name: row.chemical_activity}})
    MERGE (r)-[:RECORDS]->(a)
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

# Connects an existing TRI-reported category Chemical to its non-reported member
# Candidate, per the ontology (Chemical category → CONTAINS → Candidate).
_CONTAINS_QUERY = f"""
UNWIND $rows AS row
CALL {{
    WITH row
    MATCH (cat:Chemical {{cas_number: row.category_code}})
    MATCH (m:Candidate {{cas_number: row.cas_number}})
    MERGE (cat)-[:CONTAINS]->(m)
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
        """Bulk-upsert RECORDS edges (Report → Activity).

        Attaches the condition-of-use Activity to its originating Report, matching
        the canonical ontology so the activity shares a Report with the chemical
        and reduction it was filed alongside.

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

    def load_contains(self, df: pd.DataFrame) -> int:
        """Bulk-upsert CONTAINS edges (Chemical category → Candidate).

        Args:
            df (pd.DataFrame): DataFrame with columns category_code, cas_number.

        Returns:
            int: Number of records submitted.
        """
        records = self._to_records(df, ["category_code", "cas_number"], key_cols=["category_code", "cas_number"])
        count = self._bulk_write(_CONTAINS_QUERY, records)
        logger.info(f"Bulk-upserted {count} CONTAINS relationships")
        return count
