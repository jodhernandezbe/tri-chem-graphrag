# TRI Chemical GraphRAG Project Configuration

## Code Style & Standards
- **Type hints**: Required using `typing` module for all functions/methods/classes
- **Imports**: Absolute imports only, grouped (stdlib, third-party, local)
- **Line length**: Max 128 characters (black config)
- **Trailing commas**: Always use in function/method parameters and attributes
- **Linting**: `black`, `flake8`, `isort`, `pyright` (line-length: 128)

## Project Structure
- `src/knowledge_graph/`: Core KG modules (extract, transform, load)
- `src/rag_graph/`: RAG and embedding components
- `config/config.yaml`: Main configuration file
- `tests/`: Mirror src/ structure with test files
- `ancillary/tri/columns/`: TRI file column definitions
- `data/raw/`: Raw TRI datasets (CSV format)
- `graph_model.svg`: **CRITICAL** - Neo4j schema visualization

## Documentation Standards
- **Public functions/methods/classes**: Full docstrings with Args/Returns
- **Private methods**: No documentation unless complex logic
- **Module headers**: Include purpose, author, date

## Code Conventions
- **Functions**: Max 16 lines, single responsibility principle
- **Type annotations**: All parameters, returns, class attributes
- **Error handling**: Explicit exceptions with logging
- **Constants**: UPPER_CASE module-level
- **SOLID principles**: Applied throughout codebase
- **DRY principle**: Avoid code duplication, extract common patterns into utilities

## Architecture Patterns
- **Singleton**: Used for scrapers and data fetchers
- **Transformer pattern**: BaseTriTransformer inheritance
- **Configuration**: Centralized via ConfigManager
- **Dependency injection**: Constructor parameters

## Common Patterns
- Use `logger = logging.getLogger(__name__)` in each module
- Singleton pattern: `if not hasattr(cls, "instance")` check
- Config access via `ConfigManager().config`
- DataFrame operations: always `.copy()` before modifications
- Extract reusable logic into utility functions/modules

## Error Handling
- Use specific exceptions with descriptive messages
- Log errors before raising: `logger.error(f"Error: {e}")`
- Handle external API failures gracefully
- Validate inputs early in functions

## Performance & Concurrency
- **Batch operations**: Group Neo4j writes/reads in batches (500-1000 records)
- **Connection pooling**: Use Neo4j driver connection pooling for concurrent access
- **API semaphores**: Limit concurrent API calls (max 5 for NAICS, 3 for EPA)
- **Memory management**: Process large files in chunks, not all at once

## Security & Data Handling
- **Never commit actual TRI data** to git repository
- **Sanitize inputs** before Neo4j queries to prevent injection
- **Use parameterized queries** for all Neo4j operations
- **Validate file encodings** (handle ISO-8859-1 vs UTF-8 issues)
- **Secure credentials** via environment variables only

## Key Commands
```bash
# Development
pytest --verbose              # Run all tests
pytest tests/knowledge_graph/extract/  # Test specific modules
pytest tests/knowledge_graph/transform/
pyright                      # Type check
black src/ tests/            # Format code
pre-commit run --all-files   # Run all hooks

# Project
python -m src.knowledge_graph.main  # Run KG pipeline
```

## Testing Strategy
- Unit tests mirror src/ structure
- Test files prefixed with `test_`
- Mock external dependencies
- Test success and error scenarios

## Dependencies
- Core: pandas, requests, neo4j, beautifulsoup4
- ML/AI: transformers, torch, langchain, langgraph
- Dev: black, flake8, pytest, pyright

## Configuration
- Environment variables via `.env`
- YAML config for application settings
- Type validation with pyright strict mode
