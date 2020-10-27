DOCKER_RUN := docker run -u `id -u $(USER)`:`id -g $(USER)` --rm -v $(shell pwd):/app

help: ## Show this help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

format: ## Run pre-commit hooks to format code
	 pre-commit run --all-files

PYPI_USERNAME ?= trasee_rd
PYPI_PASSWORD ?=
build.dev: ## Build docker development image
	docker build --build-arg PYPI_USERNAME=${PYPI_USERNAME} --build-arg PYPI_PASSWORD=${PYPI_PASSWORD} -f dockerfiles/Dockerfile.dev -t piotrekzie100/dev:ssdir-dev .

build.prod: ## Build docker production image
	docker build --build-arg PYPI_USERNAME=${PYPI_USERNAME} --build-arg PYPI_PASSWORD=${PYPI_PASSWORD} -f dockerfiles/Dockerfile.prod -t piotrekzie100/dev:ssdir .

shell: ## Run docker dev shell
	$(DOCKER_RUN) -it piotrekzie100/dev:ssdir-dev /bin/bash

args ?=  -n auto -vvv --cov ssdir
test: ## Run tests
	$(DOCKER_RUN) piotrekzie100/dev:ssdir-dev pytest $(args)

gpu ?= 3
ssdir_args ?= ssdir --config-file config.yml train
run: ## Run model
	$(DOCKER_RUN) --gpus '"device=$(gpu)"' --shm-size 24G piotrekzie100/dev:ssdir $(ssdir_args)
