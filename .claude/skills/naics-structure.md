# NAICS Structure Skill

Reference for working with North American Industry Classification System (NAICS) codes
in the TRI Chemical GraphRAG project.  Use this when writing Cypher queries, designing
multi-level sector analysis, or interpreting sector-conditioned evaluation results.

---

## NAICS Hierarchy

NAICS organises industries into five nested levels.  In Neo4j, every `Sector` node stores
the full **6-digit National Industry** code; higher levels are extracted via prefix truncation.

| Level | Digits | Name | Cypher extraction |
|-------|--------|------|-------------------|
| L1 | 2 | Sector | `substring(s.naics_code, 0, 2)` |
| L2 | 3 | Subsector | `substring(s.naics_code, 0, 3)` |
| L3 | 4 | Industry Group | `substring(s.naics_code, 0, 4)` |
| L4 | 5 | NAICS Industry | `substring(s.naics_code, 0, 5)` |
| L5 | 6 | National Industry | `s.naics_code` (stored directly) |

**Rule of thumb for this project:**
- L1 (2-digit): broadest grouping; use when sample size is small or for exploratory analysis
- L3 (4-digit, Industry Group): best balance of specificity and sample coverage in TRI data
- L5 (6-digit): most precise; appropriate when the chemical–sector pair has ≥ 5 reports

---

## TRI-Relevant NAICS Sectors

Most TRI chemicals appear in manufacturing (sector 31–33).  The subsectors below cover
the majority of source reduction activity records in the graph.

| 2-digit | Name | Key chemicals |
|---------|------|---------------|
| 31–33 | Manufacturing (broad) | Most TRI chemicals |
| 32 | Chemical, Petroleum, Plastics, Rubber Manufacturing | Benzene, toluene, styrene, lead |
| 325 | Chemical Manufacturing | Formaldehyde, methanol, acids |
| 324 | Petroleum and Coal Products | Benzene, xylene, hexane |
| 331 | Primary Metal Manufacturing | Lead, nickel, chromium, zinc |
| 332 | Fabricated Metal Products | Manganese, chromium, cadmium |
| 334 | Computer and Electronic Products | Mercury, lead, arsenic |
| 336 | Transportation Equipment | Lead, toluene, xylene |

---

## Cypher Patterns

### Get chemicals with their NAICS levels

```cypher
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:FILED_BY]->(f:Facility)-[:CLASSIFIED_AS]->(s:Sector)
WHERE r.reporting_year >= 2021
RETURN
  c.cas_number                   AS cas_number,
  s.naics_code                   AS naics_6digit,
  substring(s.naics_code, 0, 5)  AS naics_5digit,
  substring(s.naics_code, 0, 4)  AS naics_4digit,
  substring(s.naics_code, 0, 3)  AS naics_3digit,
  substring(s.naics_code, 0, 2)  AS naics_2digit
```

### Get source reduction activities conditioned on NAICS level L

```cypher
// Replace $L with the number of digits (2, 3, 4, 5, or 6)
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:DOCUMENTS]->(red:Reduction)
MATCH (r)-[:FILED_BY]->(f:Facility)-[:CLASSIFIED_AS]->(s:Sector)
WHERE r.reporting_year < 2021
  AND substring(s.naics_code, 0, $L) = $sector_prefix
RETURN
  c.cas_number                               AS cas_number,
  collect(DISTINCT red.code)                 AS activity_codes,
  collect(DISTINCT s.naics_code)             AS naics_codes
```

### Count chemicals per NAICS level (useful for choosing the right level)

```cypher
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:FILED_BY]->(f:Facility)-[:CLASSIFIED_AS]->(s:Sector)
WITH c, substring(s.naics_code, 0, 4) AS naics4, count(DISTINCT r) AS reports
WHERE reports >= 2
RETURN naics4, count(DISTINCT c) AS chemical_count
ORDER BY chemical_count DESC
LIMIT 20
```

### Build (NAICS-prefix × activity-code) pairs at a given level (used in operational anchor)

```cypher
// Used by AnchorBuilder.build_operational_anchors(naics_digits=4)
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:DOCUMENTS]->(red:Reduction)
MATCH (r)-[:FILED_BY]->(f:Facility)-[:CLASSIFIED_AS]->(s:Sector)
WHERE r.reporting_year < 2021
RETURN
  c.cas_number                                               AS cas_number,
  collect(DISTINCT substring(s.naics_code, 0, 4) + '|' + red.code) AS naics4_activity_pairs
```

### Sector-conditioned activity coverage (used in ActivityCoverageMetrics)

```cypher
// For a specific query chemical and NAICS prefix, find retrieved chemicals' activities
// within the same sector context
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:DOCUMENTS]->(red:Reduction)
MATCH (r)-[:FILED_BY]->(f:Facility)-[:CLASSIFIED_AS]->(s:Sector)
WHERE c.cas_number IN $retrieved_cas_numbers
  AND substring(s.naics_code, 0, $L) = $sector_prefix
RETURN
  c.cas_number      AS cas_number,
  collect(DISTINCT red.code) AS activity_codes
```

---

## Activity Code Families (TRI_SOURCE_REDUCTION_GROUPS)

Source reduction activity codes map to 5 families defined in `config.yaml`.
Use this mapping when computing **FamilyCoverage@k** (family-level activity coverage).

| Family ID | Family Name | Codes |
|-----------|-------------|-------|
| F1 | Material Substitutions and Modifications | S01, S02, S03, S04, S05, S06 |
| F2 | Product Modifications | S11, S12, S13, S14 |
| F3 | Process and Equipment Modifications | S21, S22, S23, S24, S25 |
| F4 | Inventory and Material Management | S31, S32, S33, S34, S35 |
| F5 | Operating Practices and Training | S41, S42, S43, S44 |

**Python mapping pattern:**

```python
from src.config_manager import ConfigManager

def build_code_to_family() -> dict[str, str]:
    groups = ConfigManager().config["TRI_SOURCE_REDUCTION_GROUPS"]
    return {
        code: f"F{i+1}"
        for i, group in enumerate(groups)
        for code in group["CODES"]
    }
```

---

## Choosing a NAICS Level for Analysis

Use this decision guide when writing evaluation code or interpreting results:

| Goal | Recommended level | Reason |
|------|------------------|--------|
| Maximum recall (broad context) | L1 — 2-digit | Every chemical in Manufacturing shares the prefix |
| Balanced specificity / coverage | L3 — 4-digit | Typical group size: 10–50 chemicals per prefix |
| Precise sector matching | L5 — 6-digit | Same as stored; use only when ≥ 5 reports per cell |
| Default in this project | L5 — 6-digit | Pre-registered; change via `--naics-digits` CLI flag |
| Sensitivity analysis | Run all L1, L3, L5 | Compare SectorCoverage@10 across levels |

**CLI flag:** `python -m src.validation.main --naics-digits 4` uses L3 (Industry Group) for
the operational anchor and sector coverage calculations.

---

## Known Data Facts (dev split 2014–2020)

- 167 chemicals have activity profiles in the dev split
- 49 unique TRI source reduction codes appear
- 1,115 unique (6-digit NAICS × activity code) pairs
- Most chemicals appear in 1–3 distinct 6-digit NAICS sectors
- Using 4-digit NAICS approximately triples the NAICS co-occurrence counts per pair
