# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""FastAPI entry point for the TRI Chemical GraphRAG pipeline.

The RAG pipeline (and its embedding models) is loaded once at server startup
via the lifespan context manager and kept in memory for the lifetime of the
process. Subsequent requests reuse the cached models without reloading weights.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-13
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.rag_graph.generation.rag_pipeline import RAGPipeline

from .schemas import QueryRequest, QueryResponse

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the RAG pipeline (and embedding models) once at startup."""
    logger.info("Loading RAG pipeline and embedding models...")
    RAGPipeline()
    logger.info("RAG pipeline ready.")
    yield


app = FastAPI(
    title="TRI Chemical GraphRAG API",
    description="Query the TRI knowledge graph via natural language.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({"status": "ok"})


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Run the GraphRAG pipeline for a natural language question.

    Args:
        request (QueryRequest): Query payload with NL question and optional SMILES.

    Returns:
        QueryResponse: Narrative answer and top-10 source reduction records as a list of dicts.
    """
    try:
        result = RAGPipeline().run_structured(
            query=request.query,
            chemical_smiles=request.chemical_smiles,
            industry_sector=request.industry_sector,
            condition_of_use=request.condition_of_use,
        )
        return QueryResponse(
            answer=result["answer"],
            records=result["records"],
            needs_clarification=result["needs_clarification"],
            clarification=result["clarification"],
        )
    except Exception as exc:
        logger.error(f"Pipeline error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
