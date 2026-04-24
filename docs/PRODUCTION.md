# Production Deployment

## Overview

The production environment uses Docker containers with minimal dependencies and optimized configuration for performance and security.

## Quick Start

```bash
docker-compose --profile prod up neo4j app-prod -d
```

## Production Configuration

### Neo4j Database

The Neo4j instance in production uses minimal setup with:
- APOC plugin enabled
- Basic authentication configured via `config/config.yaml`
- Data persistence through Docker volumes
- No exposed HTTP interface (Bolt protocol only)

### Python Application

The production container:
- Installs only main dependencies (excludes dev tools)
- Copies source code into container (no live mounting)
- Uses optimized Python image
- Runs with minimal resource footprint

## Database Access

- **Bolt Protocol**: bolt://localhost:7687
- **Credentials**: Configured in `config/config.yaml`
- **Web UI**: Not exposed in production for security

## Container Management

```bash
# Start production services
docker-compose --profile prod up -d

# View production logs
docker-compose logs app-prod
docker-compose logs neo4j

# Stop production services
docker-compose --profile prod down

# Update production deployment
docker-compose build app-prod
docker-compose --profile prod up -d
```

## Monitoring

```bash
# Check container status
docker-compose --profile prod ps

# View resource usage
docker stats

# Check Neo4j health
docker-compose exec neo4j cypher-shell "RETURN 1"
```

## Security Considerations

- Neo4j HTTP interface disabled
- Minimal container attack surface
- No development dependencies included
- Read-only container filesystem where possible
- Database credentials managed via configuration files