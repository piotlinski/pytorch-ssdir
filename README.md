# SSDIR

Single Shot Multi-Box Detect Infer & Repeat implementation in PyTorch

## Development

Requirements:

- Install `pre-commit` (https://pre-commit.com/#install)
- Install `poetry` (https://python-poetry.org/docs/#installation)
- Execute `pre-commit install`
- Use `poetry` to handle requirements
  - Execute `poetry add <package_name>` to add new library
  - Execute `poetry install` to create virtualenv and install packages

## Manual

Use `make` to run commands

- `make help` - show help
- `make format` - format code
- `make test` - run tests
  - `args="--lf" make test` - run pytest tests with different arguments
- `make shell` - run poetry shell
- `make build` - build docker image with SSDIR package
