"""Read-only: intrinsic embedding Spearman rho (same code path as src/validation/main.py step 'embedding') + graph counts."""

import json
import logging
import sys

sys.path.insert(0, "/Users/josehernandez/Documents/tri_chem_graphrag")
from dotenv import load_dotenv  # noqa: E402

load_dotenv("/Users/josehernandez/Documents/tri_chem_graphrag/.env")
from src.neo4j_connector import Neo4jConnector  # noqa: E402
from src.validation.anchor_builder import AnchorBuilder  # noqa: E402
from src.validation.embedding_metrics import EmbeddingMetrics  # noqa: E402

ab = AnchorBuilder()
st = ab.build_structural_anchors()
op = ab.build_operational_anchors()
em = EmbeddingMetrics(ab)
out = {}
for et in ("MorganFingerprint", "MolDeBERTa", "Hybrid"):
    r = em.compute(et, structural=st, operational=op, project=False)
    out[et] = dict(
        n=len(r.cas_numbers),
        rho_tanimoto=r.spearman_rho,
        p_tanimoto=r.spearman_pvalue,
        rho_jaccard_code=r.spearman_rho_operational,
        p_jaccard_code=r.spearman_pvalue_operational,
        rho_jaccard_family=r.spearman_rho_family,
    )
cn = Neo4jConnector()
with cn.session() as s:
    out["graph_nodes"] = {
        r["l"][0] if r["l"] else None: r["n"] for r in s.run("MATCH (n) RETURN labels(n) AS l, count(*) AS n")
    }
    out["graph_edges"] = {r["t"]: r["n"] for r in s.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n")}
print(json.dumps(out, indent=1, default=str))
json.dump(out, open("descriptor_rho.json", "w"), indent=1, default=str)
