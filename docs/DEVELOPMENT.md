# Development Guidelines

## Docker Development

### Development Environment

For development with live code editing and all dependencies (including dev dependencies):

```bash
docker-compose up neo4j app-dev -d
```

This will:
- Start Neo4j database with APOC plugin
- Mount entire project directory for bidirectional live editing
- Install all dependencies including development tools
- Keep containers running for interactive development

#### Working Inside the Container

Access the development container:
```bash
docker exec -it python-app-dev bash
```

**Important**: In the Docker environment, Poetry is configured with `virtualenvs.create false`, so:
- No need to run `poetry shell` (not available)
- Dependencies are installed globally in the container
- Run commands directly: `python -m src` or `poetry run python -m src`

### Development Workflow Options

You can choose between two development approaches:

1. **Local Development**: Use Poetry and pre-commit on your host machine (requires local Python 3.12+ and Poetry installation)
2. **Container Development**: Work inside the Docker container with all tools pre-installed
3. **Hybrid**: Use your host IDE for editing with container for running/testing

Both approaches work with the same project files due to bidirectional mounting.

### Database Access

- **Web UI**: http://localhost:7474
- **Bolt Protocol**: bolt://localhost:7687 (from host) or bolt://neo4j:7687 (from containers)
- **Credentials**: Configured in `config/config.yaml`

### Container Management

```bash
# View running containers
docker-compose ps

# View logs
docker-compose logs app-dev
docker-compose logs neo4j

# Stop services
docker-compose down

# Rebuild containers after dependency changes
docker-compose build app-dev
```

## Poetry

### Local Development (Host Machine)

When working directly on your host machine:

#### New Dependencies
```bash
poetry add package-name
poetry add --group dev package-name  # for dev dependencies
```

#### Environment Setup
```bash
poetry install                    # Install all dependencies
poetry shell                      # Activate virtual environment
poetry run python src/main.py     # Run commands in environment
```

### Docker Development

When working inside the Docker container:

#### New Dependencies
```bash
# Inside container (docker exec -it python-app-dev bash)
poetry add package-name           # Dependencies install globally
poetry install                   # Reinstall after changes

# Then rebuild container to persist changes
docker-compose build app-dev
```

#### Running Commands
```bash
# Inside container - no need for poetry shell
python -m src
poetry run python -m src          # Alternative syntax
```

## Pre-commit

### Changes

If there is any change in `.pre-commit-config.yaml`, the following command has to be run:

```
poetry run pre-commit autoupdate
```

### Pull

Each time you pull changes, run the following command to ensure your local environment is up-to-date:

```
poetry run pre-commit install
```

### Manually Run Hooks

To manually run all pre-commit hooks on all files in the repository, use the following command:

```
poetry run pre-commit run --all-files
```

Note: this is not required when you commit changes.

If you are running the above command or committing your changes, and one or more hooks like black or isort fail, stage their modifications to the git staging area by running `git add`. After that, you can run `commit` again.

## Installing pyright language server for IDE typecheck highlighting

Detailed instructions: [pyright](https://microsoft.github.io/pyright/#/installation)

[Pycharm](https://github.com/InSyncWithFoo/pyright-langserver-for-pycharm)

VSCode: search for `Pylance` on marketplace

Add path to executable to plugin:

```shell
which pyright-langserver
```

Insert that path to plugin config in your IDE as path to executable

## ETL Pipeline

### Running the pipeline

```bash
# All steps for a single year
python -m src.knowledge_graph.main --years 2022

# Multiple years
python -m src.knowledge_graph.main --years 2022 2023 2024

# Specific steps only
python -m src.knowledge_graph.main --years 2022 --steps extract transform
python -m src.knowledge_graph.main --years 2022 --steps load
```

### Available steps

| Step | Description |
|------|-------------|
| `extract` | Downloads raw TRI CSV files from the EPA website |
| `transform` | Cleans, enriches (NAICS titles, canonical SMILES), and saves processed CSVs |
| `load` | Bulk-upserts nodes and relationships into Neo4j and populates vector embeddings |

### Year tracking

Completed steps are recorded in `ancillary/tri/processed_years.json`. Re-running
the same year and step is a no-op unless the entry is removed from that file.

### Testing with 2022 data

```bash
# Start Neo4j first
docker-compose up neo4j -d

# Run the full pipeline for 2022
python -m src.knowledge_graph.main --years 2022
```

### Transformer pairing fix and graph repairs (2026-11-04)

**Defect.** `Tri2aTransformer` (`src/knowledge_graph/transform/tri_2a_transformer.py`) matched every column
containing `est_annual_reduction_code`, which also matched the `..._est_annual_reduction_code_description`
text columns. The estimate list was therefore interleaved with descriptions, and the loader paired
`codes[i]` with `annual_reductions[i]` on that list. Activity 2 received a description, activity 3 received
activity 2's estimate, and activity 4 received a description. Activities 2 and 4 were dropped, and activity 3
could be loaded with the wrong estimate.

**Fix.** Each activity is paired with its own `<ordinal>_source_reduction_activity_code_est_annual_reduction_code`
column, and only R1 to R3 are kept. The `_description` columns are excluded from the reduction columns.

**Graph after repair.** The 2014 to 2024 graph was repaired in place to match a clean build. The first four
counts equal the raw-data counts from `data/raw/tri_2a_*.csv`:

- 3,041 activities (`DOCUMENTS` edges)
- 2,932 reports, 1,742 facilities
- 194 chemicals, of which 166 have a valid structure
- 2,043 `Candidate` nodes and 2,162 `CONTAINS` edges

Before the repair the graph held 2,940 / 2,912 / 1,736 / 192 for the first four counts, 2,040 Candidates and
2,159 CONTAINS edges.

**One-off repair scripts** in `manuscript/scripts/`. Both are dry run by default, pass `--apply` to write, and
write an audit JSON:

- `patch_graph_transformer_fix.py` diffs the graph against the corrected transformer output and adds, removes
  or corrects `DOCUMENTS` edges, and adds the missing Report, Facility, Sector, Chemical and Candidate nodes.
  Audit: `manuscript/audit_logs/transformer_patch_2026-11-04.json`.
- `delete_orphan_2022_nodes.py` removes the RY2022 Report nodes left without a `DOCUMENTS` edge, and the nodes
  stranded by them. A clean build does not create these. Audit: `manuscript/audit_logs/orphan_cleanup_2026-11-04.json`.

Database dumps taken before each run are in `manuscript/audit_logs/neo4j_backup/`, under
`2026-11-04_pre_orphan_cleanup/` and `2026-11-04_pre_transformer_patch/`. Diagnosis: `manuscript/audit_logs/count_gaps_2026-11-04.md`.

## Retrieval Configuration

Retrieval behaviour is set in `config/config.yaml` under `RAG_GRAPH.RETRIEVAL`. The shipped values are the
defaults. Each can be overridden per call through the `switches` argument of `RAGPipeline.run` and
`run_structured`, or through the `retrieval` field of the API request. Invalid values raise `ValueError`.

### Fused retrieval (shipped default, `RETRIEVAL: fused`)

Fused retrieval scores reduction records (one report, chemical and activity code) and fuses four channels by
weighted reciprocal rank fusion, score = Σ w / (60 + rank). Channels for inputs that were not supplied are skipped.

| Channel | Ranks records by | Weight |
|---------|------------------|-------:|
| chemical | Similarity of the 50 nearest chemicals in the configured embedding (`CHEMICAL_EMBEDDING_TYPE`, Hybrid) | 1.0 |
| use | The stated condition of use matched to the 10 nearest Activity names | 1.0 |
| sector | The stated sector matched to the 10 nearest NAICS titles | 1.0 |
| prior | How many chemicals among the visible reports use the record's code | 0.5 |

The prior weight (`RETRIEVAL_WEIGHTS`) was chosen by cross-validation inside the index split
(`manuscript/audit_logs/prior_weight_evaluation.json`). No language model takes part in retrieval or ranking.

Each record is labelled with an evidence tier:

1. same chemical, stated use
2. analogue, stated use
3. same chemical, other use
4. analogue, stated sector
5. analogue, other context
6. other practice (not an analogue)

Evidence is never dropped silently. When no record matches the stated use, the answer starts with a fixed notice
and the records are ranked by chemical similarity, sector and overall practice. The legacy sector and
condition-of-use fallbacks also add fixed notices. Catch-all activity codes (those beginning "other") are listed after
the specific codes under a fixed heading. This changes presentation only, not the scores or the order of the
underlying ranking.

### Legacy retrieval (`RETRIEVAL: legacy`)

Set `RETRIEVAL: legacy`, or pass `retrieval=legacy` for one call, to restore the previous similarity-ranked
traversal (vector ANN, graph traversal, cross-entity intersection). It is kept to reproduce earlier answers. The
other switches below apply to legacy only.

### Other switches

| Switch | Values | Default | Effect |
|--------|--------|---------|--------|
| `ranking` | `similarity`, `specificity`, `consensus_specificity` | `similarity` | Orders records in `tabulate_records`. The non-similarity options weight activity codes by IDF. |
| `cou_filter` | `hard`, `soft` | `hard` | `hard` returns only records matching the condition of use. `soft` repeats the traversal without that filter when nothing matches, and the answer says so. |
| `analogues` | `molecular`, `context` | `molecular` | `context` re-ranks the top 50 molecular neighbours by sector and use similarity. |
| `chemical_embedding` | `MorganFingerprint`, `MolDeBERTa`, `Hybrid` | `Hybrid` (`CHEMICAL_EMBEDDING_TYPE`) | Chemical index for SMILES similarity. The RAGAS conditions set it per run. |
| `report_split` | `index`, `test`, or none | none (all reports) | Per-call argument of the retriever rather than a config switch. Evaluators pass `index` so that withheld reports are hidden. |

Switch values are defined in `src/rag_graph/retrieval/switches.py`.

## Documentation Style

The project follows the Google style to document the code. The pre-commit hooks are configured to check this style.