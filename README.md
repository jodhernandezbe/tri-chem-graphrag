# TRI Chemical GraphRAG

A Python application for TRI chemical knowledge graph retrieval-augmented generation using Neo4j.

## Requirements

- Python 3.12+
- Poetry for dependency management
- Docker and Docker Compose for containerized environments
- Neo4j database (provided via Docker)
- Ollama LLM models platform (provided via Docker)

## Quick Start

### Development Environment

For development with live code editing and dev dependencies:

```bash
docker-compose up neo4j ollama app-dev -d
```

### Production Environment

For production deployment:

```bash
docker-compose --profile prod up neo4j ollama app-prod -d
```

## Service URLs

Once the stack is running, the following interfaces are available from the host machine:

| Service | URL | Notes |
|---------|-----|-------|
| **Neo4j Browser** | http://localhost:7474 | Graph database UI; connect with `bolt://localhost:7687` |
| **Ollama API** | http://localhost:11435 | LLM inference; port remapped from container's 11434 |
| **FastAPI (app-dev)** | http://localhost:8000 | REST API; interactive docs at `/docs` and `/redoc` |

> **Inside the container** services are reached via Docker service names:
> `bolt://neo4j:7687`, `http://ollama:11434`, — these are set automatically via `docker-compose.yml` env vars.

### API Quick Test

```bash
# Health check
curl http://localhost:8000/health

# Query the pipeline
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Which chemicals in the chemical manufacturing sector had the most source reductions?", "chemical_smiles": ""}'
```

## Related Links

- [DEVELOPMENT.md](./docs/DEVELOPMENT.md)
- [PRODUCTION.md](./docs/PRODUCTION.md)

## Citation

If you use this software, please cite it. Citation metadata is in [`CITATION.cff`](./CITATION.cff): GitHub's
"Cite this repository" button reads it, and Zenodo uses it to describe each archived release (there is
deliberately no `.zenodo.json`). `tests/test_citation.py` keeps it consistent with `LICENSE` and `pyproject.toml`.

## License

Released under the [Apache License 2.0](./LICENSE).
