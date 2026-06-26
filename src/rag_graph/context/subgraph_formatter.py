# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Formats a single subgraph dict into a human-readable text block.

The output is plain text optimised for injection into an LLM prompt:
structured enough to be unambiguous, but compact enough to respect the
context-size budget managed by ContextBuilder.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-09
"""

from typing import Any, Dict, List


def _similarity_suffix(subgraph: Dict[str, Any]) -> str:
    sim = subgraph.get("similarity")
    if sim is None:
        return ""
    tag = " — QUERY CHEMICAL" if float(sim) >= 0.99999 else ""
    return f" | Similarity {sim}{tag}"


def _fmt_record(record: Dict[str, Any]) -> str:
    parts = []
    if record.get("year"):
        parts.append(f"Year {record['year']}")
    if record.get("facility"):
        parts.append(f"Facility {record['facility']}")
    if record.get("state"):
        parts.append(f"State {record['state']}")
    if record.get("chemical"):
        cas = f" ({record['cas_number']})" if record.get("cas_number") else ""
        parts.append(f"Chemical {record['chemical']}{cas}")
    if record.get("sector_code"):
        title = f" {record['sector_title']}" if record.get("sector_title") else ""
        parts.append(f"Sector {record['sector_code']}{title}")
    if record.get("reduction_code"):
        name = f" ({record['reduction_name']})" if record.get("reduction_name") else ""
        parts.append(f"Reduction {record['reduction_code']}{name}")
    return " | ".join(parts)


def _fmt_records(records: List[Dict[str, Any]], label: str) -> str:
    if not records:
        return f"  {label}: none recorded\n"
    lines = [f"  {label}:\n"]
    for rec in records:
        line = _fmt_record(rec)
        if line:
            lines.append(f"    - {line}\n")
    return "".join(lines)


def format_chemical_subgraph(subgraph: Dict[str, Any]) -> str:
    """Render a Chemical subgraph dict as a text block.

    Args:
        subgraph (Dict[str, Any]): Subgraph returned by ``GraphRetriever.traverse``.

    Returns:
        str: Human-readable text block for prompt injection.
    """
    name = subgraph.get("name", "Unknown")
    cas = subgraph.get("cas_number", "N/A")
    smiles = subgraph.get("smiles", "N/A")
    header = f"[Chemical] {name} | CAS {cas} | SMILES {smiles}{_similarity_suffix(subgraph)}\n"
    body = _fmt_records(subgraph.get("reports", []), "Reports")
    return header + body


def format_sector_subgraph(subgraph: Dict[str, Any]) -> str:
    """Render a Sector subgraph dict as a text block.

    Args:
        subgraph (Dict[str, Any]): Subgraph returned by ``GraphRetriever.traverse``.

    Returns:
        str: Human-readable text block for prompt injection.
    """
    code = subgraph.get("naics_code", "N/A")
    title = subgraph.get("naics_title", "Unknown")
    header = f"[Sector] {code} — {title}\n"
    body = _fmt_records(subgraph.get("context", []), "Activity")
    return header + body


def format_cross_entity_subgraph(subgraph: Dict[str, Any]) -> str:
    """Render a Chemical × Sector intersection subgraph dict as a text block.

    Args:
        subgraph (Dict[str, Any]): Subgraph returned by ``CrossEntityRetriever.traverse_cross``.

    Returns:
        str: Human-readable text block for prompt injection.
    """
    name = subgraph.get("name", "Unknown")
    cas = subgraph.get("cas_number", "N/A")
    smiles = subgraph.get("smiles", "N/A")
    sector_code = subgraph.get("sector_code", "N/A")
    sector_title = subgraph.get("sector_title", "Unknown")
    header = (
        f"[Chemical × Sector] {name} | CAS {cas} | SMILES {smiles} | "
        f"Sector {sector_code} — {sector_title}{_similarity_suffix(subgraph)}\n"
    )
    body = _fmt_records(subgraph.get("reports", []), "Reports")
    return header + body


FORMATTERS = {
    "Chemical": format_chemical_subgraph,
    "Sector": format_sector_subgraph,
    "Chemical_Sector": format_cross_entity_subgraph,
}


def format_subgraph(subgraph: Dict[str, Any]) -> str:
    """Dispatch to the appropriate formatter based on ``node_type``.

    Args:
        subgraph (Dict[str, Any]): Subgraph dict with a ``node_type`` key.

    Returns:
        str: Formatted text block, or empty string for unknown node types.
    """
    node_type: str = subgraph.get("node_type", "")
    formatter = FORMATTERS.get(node_type)
    if formatter is None:
        return ""
    return formatter(subgraph)
