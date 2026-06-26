# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Smoke-test cases drawn from EPA's TSCA chemical risk evaluations.

Each case always supplies a chemical ``SMILES`` and then exactly one optional
context field — either an ``industry_sector`` (free-text → NAICS-title semantic
search) or a ``condition_of_use`` (free-text → Activity-name semantic search).
The context value corresponds to a **challenging industrial scenario** associated
with that chemical's TSCA risk evaluation; each case carries the EPA reference
(``reference``) and a short ``scenario`` note. Scenario wording was checked against
the linked EPA pages (June 2026) and limited to what those pages support.

These context strings describe EPA-identified conditions of use / industrial
settings; they are resolved by semantic (embedding) similarity, not exact matching.

Author: Jose D. Hernandez-Betancur
Date: 2026-06-23
"""

from typing import Any, Dict, List

_QUESTION = "What source reduction activities have historically been effective for this chemical?"

# Each entry: a human label, the EPA-associated challenging scenario and reference,
# plus the JSON payload posted to POST /query. Invariant: every payload includes
# "chemical_smiles"; exactly one of "industry_sector" / "condition_of_use" is set
# (except the negative control).
EPA_RISK_EVAL_CASES: List[Dict[str, Any]] = [
    {
        "label": "Trichloroethylene (TCE) — vapor degreasing",
        "scenario": "Vapor degreasing, a primary industrial use; EPA found TCE presents unreasonable risk under its conditions of use.",
        "reference": "https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/risk-evaluation-trichloroethylene-tce-0",
        "payload": {
            "query": _QUESTION,
            "chemical_smiles": "ClC=C(Cl)Cl",
            "condition_of_use": "Vapor degreasing of fabricated metal parts",
        },
    },
    {
        "label": "Perchloroethylene (PCE) — dry cleaning",
        "scenario": "Commercial dry cleaning, a condition of use; 60 of 61 conditions of use contribute to the unreasonable-risk determination.",
        "reference": "https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/risk-evaluation-perchloroethylene-pce",
        "payload": {
            "query": _QUESTION,
            "chemical_smiles": "ClC(Cl)=C(Cl)Cl",
            "industry_sector": "Dry cleaning and laundry services",
        },
    },
    {
        "label": "Methylene chloride — paint and coating removal",
        "scenario": "Paint and coating removal, a condition of use EPA addressed; methylene chloride presents unreasonable risk under its conditions of use.",
        "reference": "https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/final-risk-evaluation-methylene-chloride",
        "payload": {
            "query": _QUESTION,
            "chemical_smiles": "ClCCl",
            "condition_of_use": "Commercial paint and coating removal",
        },
    },
    {
        "label": "1-Bromopropane (1-BP) — vapor degreasing / spray adhesives",
        "scenario": "Vapor degreasing and spray-adhesive use, industrial conditions of use; 1-BP presents unreasonable risk under its conditions of use.",
        "reference": "https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/risk-evaluation-1-bromopropane-1-bp",
        "payload": {
            "query": _QUESTION,
            "chemical_smiles": "CCCBr",
            "condition_of_use": "Vapor degreasing and spray adhesive application",
        },
    },
    {
        "label": "Carbon tetrachloride — industrial chemical manufacturing",
        "scenario": "Industrial processing/use in chemical manufacturing; 13 of 15 conditions of use present unreasonable risk to workers.",
        "reference": "https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/final-risk-evaluation-carbon-tetrachloride",
        "payload": {
            "query": _QUESTION,
            "chemical_smiles": "ClC(Cl)(Cl)Cl",
            "condition_of_use": "Industrial processing and use in chemical manufacturing",
        },
    },
    {
        "label": "1,4-Dioxane — byproduct in industrial processing",
        "scenario": "Worker exposure where 1,4-dioxane is generated as a byproduct; significantly contributes to the unreasonable-risk determination.",
        "reference": "https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/final-risk-evaluation-14-dioxane",
        "payload": {
            "query": _QUESTION,
            "chemical_smiles": "C1COCCO1",
            "condition_of_use": "Generated as a byproduct during industrial chemical processing",
        },
    },
    {
        "label": "N-Methylpyrrolidone (NMP) — industrial solvent use",
        "scenario": "Industrial/commercial use as a solvent (dermal exposure); NMP whole chemical presents unreasonable risk (29 of 37 conditions of use drive the determination).",
        "reference": "https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/risk-evaluation-n-methylpyrrolidone-nmp-0",
        "payload": {
            "query": _QUESTION,
            "chemical_smiles": "CN1CCCC1=O",
            "condition_of_use": "Industrial use as a solvent",
        },
    },
    {
        "label": "1,3-Butadiene — synthetic rubber / plastics manufacturing",
        "scenario": "Monomer used in polymerization for synthetic rubber, plastic, and resin manufacturing; industrial inhalation exposure presents unreasonable risk.",
        "reference": "https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/risk-evaluation-13-butadiene",
        "payload": {
            "query": _QUESTION,
            "chemical_smiles": "C=CC=C",
            "industry_sector": "Synthetic rubber and plastics material manufacturing",
        },
    },
    {
        "label": "Formaldehyde — manufacturing / resin production",
        "scenario": "Worker inhalation during manufacturing, processing, or use (e.g., resin production); formaldehyde presents unreasonable risk.",
        "reference": "https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/risk-evaluation-formaldehyde",
        "payload": {
            "query": _QUESTION,
            "chemical_smiles": "C=O",
            "industry_sector": "Synthetic resin and adhesive manufacturing",
        },
    },
    {
        "label": "TCEP — flame retardant in paint/coating manufacturing",
        "scenario": "Flame-retardant use in paint and coating manufacturing and polymers; seven conditions of use present unreasonable risk to workers.",
        "reference": "https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/risk-evaluation-tris2-chloroethyl-phosphate-tcep",
        "payload": {
            "query": _QUESTION,
            "chemical_smiles": "O=P(OCCCl)(OCCCl)OCCCl",
            "condition_of_use": "Flame-retardant additive in paint and coating manufacturing and polymers",
        },
    },
    {
        "label": "Vinyl chloride — PVC manufacturing",
        "scenario": "Manufacturing/processing as the monomer in polyvinyl chloride (PVC) production; within the risk-evaluation scope.",
        "reference": "https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/risk-evaluation-vinyl-chloride",
        "payload": {
            "query": _QUESTION,
            "chemical_smiles": "C=CCl",
            "industry_sector": "Polyvinyl chloride (PVC) and plastics resin manufacturing",
        },
    },
    {
        # Negative control: malformed SMILES must trip the RDKit gate → clarify route.
        "label": "NEGATIVE — invalid SMILES (expect clarification)",
        "scenario": "Input-validation control (deliberately malformed structure).",
        "reference": "",
        "payload": {
            "query": _QUESTION,
            "chemical_smiles": "ClC=C(Cl",
            "condition_of_use": "Vapor degreasing",
        },
        "expect_clarification": True,
    },
]
