# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""End-to-end RAG quality evaluation using the RAGAS framework.

Loads the frozen synthetic QA benchmark, runs the shipped RAG pipeline (RAGPipeline.run_structured) for each
embedding type and the flat-vector ablation, and reports faithfulness, answer
relevancy, context precision, and context recall.

Requires RAGAS >= 0.2 and an Ollama instance for both generation and evaluation.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-19
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.config_manager import ConfigManager
from src.neo4j_connector import Neo4jConnector
from src.rag_graph.generation.prompt_manager import PromptManager
from src.rag_graph.generation.rag_pipeline import RAGPipeline
from src.rag_graph.retrieval import RetrieverFactory
from src.rag_graph.retrieval.retriever_factory import guarded_search
from src.validation.llm_json import clean_json_text

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK_PATH = Path(__file__).parents[2] / "data" / "qa_benchmark.json"

_ABLATION_CONDITIONS = ("MorganFingerprint", "MolDeBERTa", "Hybrid", "flat")

_SMILES_LOOKUP = """
MATCH (c:Chemical) WHERE c.cas_number = $cas
RETURN c.canonical_smiles AS smiles
LIMIT 1
"""

_SECTOR_TITLE_LOOKUP = """
MATCH (s:Sector) WHERE s.naics_code = $naics
RETURN s.naics_title AS title
LIMIT 1
"""


def _build_llm() -> ChatOllama:
    cfg = ConfigManager().config["RAG_GRAPH"]["GENERATION"]
    return ChatOllama(
        model=cfg["MODEL"],
        base_url=os.getenv("OLLAMA_BASE_URL", cfg["BASE_URL"]),
        temperature=float(cfg["TEMPERATURE"]),
    )


_OLLAMA_CLOUD_URL = "https://ollama.com"
_VALID_JUDGE_PROVIDERS = ("openai", "ollama_cloud", "ollama_local")


def _judge_model() -> str:
    """Return the judge model — env override ``RAGAS_JUDGE_MODEL`` wins over config."""
    rag = ConfigManager().config["RAG_GRAPH"]
    return os.getenv("RAGAS_JUDGE_MODEL") or rag.get("RAGAS", {}).get("JUDGE_MODEL", rag["GENERATION"]["MODEL"])


def _judge_provider() -> str:
    """Return the judge provider (openai | ollama_cloud | ollama_local).

    Env override ``RAGAS_JUDGE_PROVIDER`` wins over config so it can be passed on the CLI.
    """
    rag = ConfigManager().config["RAG_GRAPH"]
    return str(os.getenv("RAGAS_JUDGE_PROVIDER") or rag.get("RAGAS", {}).get("JUDGE_PROVIDER", "ollama_local")).lower()


def judge_descriptor() -> str:
    """Return a 'provider / model' string for the active judge (honors CLI/env overrides)."""
    return f"{_judge_provider()} / {_judge_model()}"


def ensure_judge_available() -> None:
    """Fail loudly before a RAGAS run if the configured judge cannot be used.

    RAGAS' faithfulness/recall prompts need a GPT-4-class judge (local 7-14B judges emit the
    wrong schema → NaN). The provider selects where that judge runs; the cloud providers need
    an API key. Raise a clear error up front rather than collecting samples (slow) and only
    then failing.
    """
    provider = _judge_provider()
    if provider not in _VALID_JUDGE_PROVIDERS:
        raise RuntimeError(f"Unknown RAG_GRAPH.RAGAS.JUDGE_PROVIDER '{provider}'. Use one of {_VALID_JUDGE_PROVIDERS}.")
    load_dotenv()
    required_key = {"openai": "OPENAI_API_KEY", "ollama_cloud": "OLLAMA_API_KEY"}.get(provider)
    if required_key and not os.getenv(required_key):
        raise RuntimeError(
            f"RAGAS judge provider is '{provider}' (model '{_judge_model()}'), but {required_key} is not set. "
            f"Add {required_key} to your .env (or environment) before running RAGAS."
        )


def _build_judge_llm() -> BaseChatModel:
    """Build the RAGAS judge LLM (separate from the generator) for the configured provider.

    - ``openai``: ChatOpenAI (e.g. gpt-4o-mini) — GPT-4-class, the reliable default.
    - ``ollama_cloud``: ChatOllama against Ollama Cloud (e.g. gpt-oss:120b-cloud) — a hosted
      open model, capable + fast, via ``OLLAMA_API_KEY``.
    - ``ollama_local``: ChatOllama against a local Ollama — note local 7-14B judges cannot
      score faithfulness/recall (context precision works); use only for quick/precision runs.
    """
    provider = _judge_provider()
    model = _judge_model()
    ensure_judge_available()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=0.0)
    if provider == "ollama_cloud":
        return ChatOllama(
            model=model,
            base_url=_OLLAMA_CLOUD_URL,
            temperature=0.0,
            client_kwargs={"headers": {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}},
        )
    cfg = ConfigManager().config["RAG_GRAPH"]["GENERATION"]
    return ChatOllama(
        model=model,
        base_url=os.getenv("OLLAMA_BASE_URL", cfg["BASE_URL"]),
        temperature=0.0,
    )


def _generate_answer(context: str, question: str, llm: ChatOllama) -> str:
    pm = PromptManager()
    system_text = pm.render("rag_system.yaml", "system_prompt", {"context": context, "condition_of_use": "not specified"})
    query_text = pm.render("rag_query.yaml", "query_prompt", {"query": question})
    response = llm.invoke([SystemMessage(content=system_text), HumanMessage(content=query_text)])
    return str(response.content)


_RECORD_FIELDS = ("Chemical", "CAS", "Similarity", "Sector", "Reduction Activity", "Year", "State")


def _record_strings(records: List[Dict[str, Any]]) -> List[str]:
    """Render each pipeline record, as shown to the user, as one context string for the RAGAS judge."""
    return [" | ".join(f"{field}: {r.get(field, '')}" for field in _RECORD_FIELDS) for r in records]


class RAGASEvaluator:
    """Runs RAGAS end-to-end evaluation for each embedding type and the flat ablation."""

    def __init__(
        self,
        benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
        connector: Optional[Neo4jConnector] = None,
        pipeline: Optional[RAGPipeline] = None,
    ) -> None:
        self._benchmark_path = benchmark_path
        self._connector = connector or Neo4jConnector()
        self._factory = RetrieverFactory()
        self._pipeline = pipeline

    def _load_benchmark(self) -> List[Dict[str, Any]]:
        if not self._benchmark_path.exists():
            raise FileNotFoundError(
                f"Frozen QA benchmark not found at {self._benchmark_path}. " "Run QAGenerator.generate() first."
            )
        return json.loads(self._benchmark_path.read_text())

    def _resolve_smiles(self, cas_number: str) -> str:
        with self._connector.session() as session:
            result = session.run(_SMILES_LOOKUP, {"cas": cas_number})  # type: ignore[arg-type]
            row = result.single()
            return str(row["smiles"]) if row and row["smiles"] else ""

    def _resolve_sector_title(self, naics_code: str) -> str:
        with self._connector.session() as session:
            result = session.run(_SECTOR_TITLE_LOOKUP, {"naics": naics_code})  # type: ignore[arg-type]
            row = result.single()
            return str(row["title"]) if row and row["title"] else ""

    def _retrieval_query(self, qa_pair: Dict[str, Any]) -> str:
        meta = qa_pair.get("metadata", {})
        if "cas_number" in meta:
            return self._resolve_smiles(meta["cas_number"])
        if "naics_code" in meta:
            return self._resolve_sector_title(meta["naics_code"])
        return qa_pair["question"]

    @staticmethod
    def _pipeline_inputs(qa_pair: Dict[str, Any], retrieval_key: str) -> Dict[str, str]:
        """Map the retrieval key onto the pipeline argument that carries it (SMILES, sector, or free text)."""
        meta = qa_pair.get("metadata", {})
        if "cas_number" in meta:
            return {"chemical_smiles": retrieval_key}
        if "naics_code" in meta:
            return {"industry_sector": retrieval_key}
        return {}

    def _get_pipeline(self) -> RAGPipeline:
        if self._pipeline is None:
            self._pipeline = RAGPipeline()
        return self._pipeline

    def _run_full(self, embedding_type: str, pipeline_inputs: Dict[str, str], question: str) -> tuple[str, List[str]]:
        """Run the shipped pipeline (default switches) with the chemical embedding overridden.

        Args:
            embedding_type (str): MorganFingerprint, MolDeBERTa or Hybrid.
            pipeline_inputs (Dict[str, str]): ``chemical_smiles`` or ``industry_sector`` carrying the retrieval key.
            question (str): Benchmark question, passed as the pipeline query.

        Returns:
            tuple[str, List[str]]: The pipeline answer and one context string per returned record.

        Raises:
            RuntimeError: If the pipeline asks for clarification instead of answering.
        """
        result = self._get_pipeline().run_structured(
            query=question,
            chemical_smiles=pipeline_inputs.get("chemical_smiles", ""),
            industry_sector=pipeline_inputs.get("industry_sector", ""),
            switches={"chemical_embedding": embedding_type},
        )
        if result["needs_clarification"]:
            raise RuntimeError(f"Pipeline requested clarification: {result['clarification']}")
        return result["answer"], _record_strings(result["records"])

    def _run_flat(self, embedding_type: str, retrieval_key: str, question: str, k: int = 5) -> tuple[str, List[str]]:
        vector_store = self._factory.get_vector_store(embedding_type)
        hits = guarded_search(vector_store, retrieval_key, k)
        chem_names = [doc.page_content for doc, _ in hits]
        context = "\n".join(chem_names)
        llm = _build_llm()
        answer = _generate_answer(context, question, llm)
        return answer, chem_names

    def _collect_samples(
        self,
        condition: str,
        pairs: List[Dict[str, Any]],
        max_pairs: int = 0,
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Collect RAGAS samples for a condition, tracking attempts, skips, and failures.

        Args:
            condition (str): Ablation condition name.
            pairs (List[Dict[str, Any]]): Frozen QA benchmark pairs.
            max_pairs (int): Cap on pairs to process (0 = all).

        Returns:
            tuple[List[Dict[str, Any]], Dict[str, int]]: Collected samples and a stats dict
                with keys n_attempted, n_failed, and n_skipped.
        """
        samples = []
        embed_type = "MolDeBERTa" if condition == "flat" else condition
        pool = pairs[:max_pairs] if max_pairs > 0 else pairs
        n_attempted = 0
        n_failed = 0
        n_skipped = 0
        for pair in pool:
            retrieval_key = self._retrieval_query(pair)
            question = pair["question"]
            if not retrieval_key:
                n_skipped += 1
                continue
            n_attempted += 1
            try:
                if condition == "flat":
                    answer, contexts = self._run_flat(embed_type, retrieval_key, question)
                else:
                    answer, contexts = self._run_full(condition, self._pipeline_inputs(pair, retrieval_key), question)
                samples.append(
                    {
                        "user_input": pair["question"],
                        "response": answer,
                        "retrieved_contexts": contexts,
                        "reference": pair["ground_truth"],
                    }
                )
            except Exception as exc:
                n_failed += 1
                logger.warning(f"Skipping QA pair — retrieval/generation failed: {exc}")
        stats = {"n_attempted": n_attempted, "n_failed": n_failed, "n_skipped": n_skipped}
        if n_attempted:
            logger.info(
                f"[{condition}] collected {len(samples)}/{n_attempted} samples "
                f"({n_failed} failed, {n_skipped} skipped for missing retrieval key)"
            )
        return samples, stats

    def evaluate(self, condition: str, max_pairs: int = 0) -> Dict[str, Any]:
        """Run RAGAS evaluation for one ablation condition.

        Args:
            condition (str): One of MorganFingerprint, MolDeBERTa, Hybrid, or flat.
                The flat condition uses MolDeBERTa for vector search but skips
                graph traversal so the LLM only sees raw chemical names.

        Returns:
            Dict[str, Any]: RAGAS metric scores (faithfulness, answer_relevancy,
                context_precision, context_recall) plus sample count.
        """
        try:
            from ragas import (  # type: ignore[import-untyped]
                EvaluationDataset,
                RunConfig,
                evaluate,
            )
            from ragas.llms import LangchainLLMWrapper  # type: ignore[import-untyped]
            from ragas.metrics import (  # type: ignore[import-untyped]
                ContextPrecision,
                ContextRecall,
                Faithfulness,
            )
        except ImportError as exc:
            raise ImportError(f"RAGAS >= 0.2 not installed: {exc}") from exc

        class _CleaningJudge(LangchainLLMWrapper):  # type: ignore[misc, valid-type]
            """Wrap the judge so each generation's text is cleaned before RAGAS parses it.

            Defense-in-depth alongside JSON mode: strips markdown fences, fixes trailing
            commas / smart quotes / missing commas that would otherwise yield NaN scores.
            """

            @staticmethod
            def _scrub(result: Any) -> Any:
                for gens in getattr(result, "generations", []):
                    for gen in gens:
                        if getattr(gen, "text", None):
                            gen.text = clean_json_text(gen.text)
                return result

            def generate_text(self, *args: Any, **kwargs: Any) -> Any:
                return self._scrub(super().generate_text(*args, **kwargs))

            async def agenerate_text(self, *args: Any, **kwargs: Any) -> Any:
                return self._scrub(await super().agenerate_text(*args, **kwargs))

        pairs = self._load_benchmark()
        samples, stats = self._collect_samples(condition, pairs, max_pairs=max_pairs)

        if not samples:
            logger.error(f"No valid samples collected for condition '{condition}'")
            return {"condition": condition, "n_samples": 0, **stats}

        judge_llm = _CleaningJudge(_build_judge_llm())  # type: ignore[arg-type]
        metrics = [Faithfulness(llm=judge_llm), ContextPrecision(llm=judge_llm), ContextRecall(llm=judge_llm)]
        dataset = EvaluationDataset.from_list(samples)
        ragas_cfg = ConfigManager().config["RAG_GRAPH"].get("RAGAS", {})
        run_config = RunConfig(
            timeout=int(ragas_cfg.get("JUDGE_TIMEOUT", 600)),
            max_workers=int(ragas_cfg.get("MAX_WORKERS", 1)),
        )
        scores = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)
        result = scores.to_pandas().mean(numeric_only=True).to_dict()
        result["condition"] = condition
        result["n_samples"] = len(samples)
        result.update(stats)
        logger.info(f"[{condition}] RAGAS scores: {result}")
        return result

    def evaluate_all(self, max_pairs: int = 0) -> List[Dict[str, Any]]:
        """Evaluate all ablation conditions and return one result dict per condition.

        Args:
            max_pairs (int): Cap on QA pairs per condition (0 = use all).

        Returns:
            List[Dict[str, Any]]: RAGAS metric scores for each condition.
        """
        ensure_judge_available()  # fail loudly before collecting samples if the judge is unusable
        return [self.evaluate(c, max_pairs=max_pairs) for c in _ABLATION_CONDITIONS]
