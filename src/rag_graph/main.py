# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Entry point for the TRI Chemical GraphRAG query pipeline.

Exposes the LangGraph RAG pipeline via argparse so it can be invoked from the
command line. The caller must supply a natural-language question, a retrieval
query appropriate for the chosen embedding type (SMILES for chemical embeddings
or a sector title for NAICS embeddings), and the desired embedding type.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-09

"""

import argparse
import logging
import sys

from .generation.rag_pipeline import RAGPipeline

SUPPORTED_EMBEDDING_TYPES = [
    "MorganFingerprint",
    "MolDeBERTa",
    "Hybrid",
    "NAICSEmbeddings",
]

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
        help="Natural language question to answer (passed to the LLM for generation).",
    )
    parser.add_argument(
        "--retrieval-query",
        required=True,
        dest="retrieval_query",
        metavar="TEXT",
        help=(
            "String to embed for vector search. "
            "Use a SMILES string for chemical embedding types (MorganFingerprint, MolDeBERTa, Hybrid) "
            "or a sector title for NAICSEmbeddings."
        ),
    )
    parser.add_argument(
        "--embedding-type",
        required=False,
        dest="embedding_type",
        default="MolDeBERTa",
        choices=SUPPORTED_EMBEDDING_TYPES,
        help="Embedding type that determines which Neo4j vector index is queried (default: MolDeBERTa).",
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

    logger.info(f"Embedding type : {args.embedding_type}")
    logger.info(f"Retrieval query: {args.retrieval_query}")
    logger.info(f"Query          : {args.query}")

    pipeline = RAGPipeline()
    answer = pipeline.run(
        query=args.query,
        retrieval_query=args.retrieval_query,
        embedding_type=args.embedding_type,
    )

    print("\n" + "=" * 60)
    print("Answer:")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()
