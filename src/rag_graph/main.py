# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Entry point for the TRI Chemical GraphRAG query pipeline.

Exposes the LangGraph RAG pipeline via argparse so it can be invoked from the
command line. The caller supplies a natural-language question; the pipeline's
query planner automatically extracts the chemical identifier and sector from the
question. A SMILES string can be passed explicitly via --chemical-smiles to
bypass the planner's chemical extraction and SMILES resolution steps.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-09
"""

import argparse
import logging
import sys

from .generation.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRI Chemical GraphRAG — query the knowledge graph via natural language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query",
        required=True,
        metavar="TEXT",
        help="Natural language question to answer.",
    )
    parser.add_argument(
        "--chemical-smiles",
        required=False,
        dest="chemical_smiles",
        default="",
        metavar="SMILES",
        help=(
            "Optional SMILES string for chemical similarity retrieval. "
            "If omitted, the query planner extracts the chemical identifier from --query "
            "and resolves it to SMILES via the knowledge graph."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the GraphRAG pipeline and print the answer to stdout.

    Args:
        argv (list[str] | None): Argument list; defaults to ``sys.argv[1:]``.
    """
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    args = _parse_args(argv if argv is not None else sys.argv[1:])

    logger.info(f"Query          : {args.query}")
    if args.chemical_smiles:
        logger.info(f"Chemical SMILES: {args.chemical_smiles}")

    pipeline = RAGPipeline()
    answer = pipeline.run(
        query=args.query,
        chemical_smiles=args.chemical_smiles,
    )

    print("\n" + "=" * 60)
    print("Answer:")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()
