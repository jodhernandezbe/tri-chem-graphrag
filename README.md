# TRI Chemical GraphRAG

A Python application for TRI chemical knowledge graph retrieval-augmented generation using Neo4j.

## Requirements

- Python 3.12+
- Poetry for dependency management
- Docker and Docker Compose for containerized environments
- Neo4j database (provided via Docker)

## Quick Start

### Development Environment

For development with live code editing and dev dependencies:

```bash
docker-compose up neo4j app-dev -d
```

### Production Environment

For production deployment:

```bash
docker-compose --profile prod up neo4j app-prod -d
```

## Related Links

- [DEVELOPMENT.md](./docs/DEVELOPMENT.md)
- [PRODUCTION.md](./docs/PRODUCTION.md)
