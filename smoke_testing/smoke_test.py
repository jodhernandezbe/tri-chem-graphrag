# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Smoke test the GraphRAG ``/query`` endpoint with EPA risk-evaluation chemicals.

Posts each case in ``epa_risk_eval_cases.py`` to the running FastAPI server via
``requests`` and writes every answer to its own text file under
``data/smoke_tests_response/``. Every payload always carries a SMILES plus, in
most cases, either an industry sector or a condition of use.

Usage:
    python -m smoke_testing.smoke_test                 # defaults to localhost:8000
    GRAPHRAG_API_URL=http://localhost:8000 python -m smoke_testing.smoke_test
    python smoke_testing/smoke_test.py --base-url http://localhost:8000 --timeout 240

Author: Jose D. Hernandez-Betancur
Date: 2026-06-23
"""

import argparse
import os
import pathlib
import re
import sys
from typing import Any, Dict, List

import requests

from smoke_testing.epa_risk_eval_cases import EPA_RISK_EVAL_CASES

DEFAULT_BASE_URL = os.getenv("GRAPHRAG_API_URL", "http://localhost:8000")
RESPONSE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "smoke_tests_response"


def _slugify(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug or "case"


def _context_summary(payload: Dict[str, Any]) -> str:
    if payload.get("industry_sector"):
        return f"sector='{payload['industry_sector']}'"
    if payload.get("condition_of_use"):
        return f"condition_of_use='{payload['condition_of_use']}'"
    return "no extra context"


def _render_records(records: List[Dict[str, Any]]) -> List[str]:
    if not records:
        return ["RECORDS: (none)"]
    distinct = sorted(
        {(r.get("Chemical", "?"), r.get("CAS", "?"), r.get("Similarity", 0.0), r.get("EntryPoint", "")) for r in records},
        key=lambda x: -x[2],
    )
    lines = [f"DISTINCT SOURCE CHEMICALS ({len(distinct)}) — cross-chemical recommendations by SMILES similarity:"]
    for name, cas, sim, entry in distinct:
        via = f"  ← category reached via member {entry}" if entry else ""
        lines.append(f"  sim={sim:.4f}  {name} (CAS {cas}){via}")
    lines += ["", "RECORDS (ranked / interleaved):"]
    for r in records:
        entry = r.get("EntryPoint", "")
        via = f" [via {entry}]" if entry else ""
        lines.append(
            f"  [{r.get('Similarity')}] {r.get('Chemical')} (CAS {r.get('CAS')}){via} | "
            f"{r.get('Reduction Activity')} | {r.get('Sector')} | {r.get('Year')} | {r.get('State')}"
        )
    return lines


def _render_report(case: Dict[str, Any], body: Dict[str, Any]) -> str:
    payload: Dict[str, Any] = case["payload"]
    records: List[Dict[str, Any]] = body.get("records", [])
    lines = [
        f"CASE: {case['label']}",
        "=" * 80,
        f"SMILES           : {payload.get('chemical_smiles', '')}",
        f"Industry sector  : {payload.get('industry_sector', '') or '—'}",
        f"Condition of use : {payload.get('condition_of_use', '') or '—'}",
        f"EPA scenario     : {case.get('scenario', '') or '—'}",
        f"EPA reference    : {case.get('reference', '') or '—'}",
        f"Query            : {payload.get('query', '')}",
        f"needs_clarification: {body.get('needs_clarification', False)}",
        f"records returned : {len(records)}",
        "-" * 80,
        "ANSWER:",
        str(body.get("answer", "")).strip(),
        "-" * 80,
        *_render_records(records),
    ]
    clarification = str(body.get("clarification", "")).strip()
    if clarification:
        lines += ["-" * 80, "CLARIFICATION:", clarification]
    return "\n".join(lines) + "\n"


def _check_health(base_url: str) -> bool:
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def run(base_url: str, timeout: float) -> int:
    """Post every case, persist each answer, and print a pass/fail summary.

    Args:
        base_url (str): Base URL of the running GraphRAG API.
        timeout (float): Per-request timeout in seconds (generation can be slow).

    Returns:
        int: Process exit code — 0 if all cases behaved as expected, 1 otherwise.
    """
    if not _check_health(base_url):
        print(f"[ABORT] API not reachable at {base_url} (is the server running?).")
        return 1

    RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target   : {base_url}/query")
    print(f"Responses: {RESPONSE_DIR}")
    print(f"Cases    : {len(EPA_RISK_EVAL_CASES)}\n")

    failures = 0
    for i, case in enumerate(EPA_RISK_EVAL_CASES, start=1):
        label = str(case["label"])
        payload: Dict[str, Any] = case["payload"]
        expect_clarification = bool(case.get("expect_clarification", False))
        try:
            resp = requests.post(f"{base_url}/query", json=payload, timeout=timeout)
        except requests.RequestException as exc:
            failures += 1
            print(f"[{i:02d}] FAIL  {label} :: request error: {exc}")
            continue

        if resp.status_code != 200:
            failures += 1
            print(f"[{i:02d}] FAIL  {label} :: HTTP {resp.status_code}: {resp.text[:160]}")
            continue

        body: Dict[str, Any] = resp.json()
        out_path = RESPONSE_DIR / f"{i:02d}_{_slugify(label)}.txt"
        out_path.write_text(_render_report(case, body), encoding="utf-8")

        got_clarification = bool(body.get("needs_clarification", False))
        ok = got_clarification == expect_clarification
        status = "OK  " if ok else "FAIL"
        if not ok:
            failures += 1
        n_records = len(body.get("records", []))
        print(f"[{i:02d}] {status}  {label} :: {_context_summary(payload)} | records={n_records} → {out_path.name}")

    print(f"\nDone. {len(EPA_RISK_EVAL_CASES) - failures}/{len(EPA_RISK_EVAL_CASES)} cases as expected.")
    return 1 if failures else 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the GraphRAG /query endpoint.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL.")
    parser.add_argument("--timeout", type=float, default=240.0, help="Per-request timeout (s).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point: parse args and run the smoke test."""
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    sys.exit(run(args.base_url, args.timeout))


if __name__ == "__main__":
    main()
