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

## Documentation Style

The project follows the Google style to document the code. The pre-commit hooks are configured to check this style.