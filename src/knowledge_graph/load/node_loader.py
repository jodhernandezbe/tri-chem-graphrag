# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Node loading module for the TRI Knowledge Graph.

Bulk-upserts all six node types (Chemical, Facility, Sector, Report, Activity,
Reduction) into Neo4j using CALL { } IN TRANSACTIONS for periodic commits.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-07
"""

import logging

import pandas as pd

from .base_loader import BATCH_SIZE, BaseLoader

logger = logging.getLogger(__name__)

_CHEMICAL_QUERY = f"""
UNWIND $rows AS row
CALL {{
    WITH row
    MERGE (c:Chemical {{cas_number: row.cas_number}})
    SET c += {{name: row.chemical_name, canonical_smiles: row.canonical_smiles, tri_chem_id: row.tri_chem_id}}
}} IN TRANSACTIONS OF {BATCH_SIZE} ROWS
"""

_FACILITY_QUERY = f"""
UNWIND $rows AS row
CALL {{
    WITH row
    MERGE (f:Facility {{trifid: row.trifid}})
    SET f += {{state: row.facility_state}}
}} IN TRANSACTIONS OF {BATCH_SIZE} ROWS
"""

_SECTOR_QUERY = f"""
UNWIND $rows AS row
CALL {{
    WITH row
    MERGE (s:Sector {{naics_code: row.naics_code}})
    SET s += {{naics_title: row.naics_title}}
}} IN TRANSACTIONS OF {BATCH_SIZE} ROWS
"""

_REPORT_QUERY = f"""
UNWIND $rows AS row
CALL {{
    WITH row
    MERGE (r:Report {{document_control_number: row.document_control_number}})
    SET r += {{reporting_year: row.reporting_year}}
}} IN TRANSACTIONS OF {BATCH_SIZE} ROWS
"""

_ACTIVITY_QUERY = f"""
UNWIND $rows AS row
CALL {{
    WITH row
    MERGE (:Activity {{name: row.chemical_activity}})
}} IN TRANSACTIONS OF {BATCH_SIZE} ROWS
"""

_REDUCTION_QUERY = f"""
UNWIND $rows AS row
CALL {{
    WITH row
    MERGE (rd:Reduction {{code: row.source_reduction_code}})
    SET rd += {{name: row.source_reduction_description}}
}} IN TRANSACTIONS OF {BATCH_SIZE} ROWS
"""


class NodeLoader(BaseLoader):
    """Bulk-upserts all Knowledge Graph node types into Neo4j."""

    def load_chemicals(self, df: pd.DataFrame) -> int:
        """Bulk-upsert Chemical nodes.

        Args:
            df (pd.DataFrame): DataFrame with columns cas_number, chemical_name,
                canonical_smiles, tri_chem_id.

        Returns:
            int: Number of records submitted.
        """
        records = self._to_records(df, ["cas_number", "chemical_name", "canonical_smiles", "tri_chem_id"])
        count = self._bulk_write(_CHEMICAL_QUERY, records)
        logger.info(f"Bulk-upserted {count} Chemical nodes")
        return count

    def load_facilities(self, df: pd.DataFrame) -> int:
        """Bulk-upsert Facility nodes.

        Args:
            df (pd.DataFrame): DataFrame with columns trifid, facility_state.

        Returns:
            int: Number of records submitted.
        """
        records = self._to_records(df, ["trifid", "facility_state"])
        count = self._bulk_write(_FACILITY_QUERY, records)
        logger.info(f"Bulk-upserted {count} Facility nodes")
        return count

    def load_sectors(self, df: pd.DataFrame) -> int:
        """Bulk-upsert Sector nodes.

        Args:
            df (pd.DataFrame): DataFrame with columns naics_code, naics_title.

        Returns:
            int: Number of records submitted.
        """
        records = self._to_records(df, ["naics_code", "naics_title"])
        count = self._bulk_write(_SECTOR_QUERY, records)
        logger.info(f"Bulk-upserted {count} Sector nodes")
        return count

    def load_reports(self, df: pd.DataFrame) -> int:
        """Bulk-upsert Report nodes.

        Args:
            df (pd.DataFrame): DataFrame with columns document_control_number, reporting_year.

        Returns:
            int: Number of records submitted.
        """
        records = self._to_records(df, ["document_control_number", "reporting_year"])
        count = self._bulk_write(_REPORT_QUERY, records)
        logger.info(f"Bulk-upserted {count} Report nodes")
        return count

    def load_activities(self, df: pd.DataFrame) -> int:
        """Bulk-upsert Activity nodes.

        Args:
            df (pd.DataFrame): DataFrame with column chemical_activity (from 1b transformer).

        Returns:
            int: Number of records submitted.
        """
        records = self._to_records(df, ["chemical_activity"], key_cols=["chemical_activity"])
        records = [r for r in records if r.get("chemical_activity")]
        count = self._bulk_write(_ACTIVITY_QUERY, records)
        logger.info(f"Bulk-upserted {count} Activity nodes")
        return count

    def load_reductions(self, df: pd.DataFrame) -> int:
        """Bulk-upsert Reduction nodes.

        Args:
            df (pd.DataFrame): DataFrame with columns source_reduction_code,
                source_reduction_description (from 2a transformer).

        Returns:
            int: Number of records submitted.
        """
        records = self._to_records(
            df,
            ["source_reduction_code", "source_reduction_description"],
            key_cols=["source_reduction_code"],
        )
        records = [r for r in records if r.get("source_reduction_code")]
        count = self._bulk_write(_REDUCTION_QUERY, records)
        logger.info(f"Bulk-upserted {count} Reduction nodes")
        return count
