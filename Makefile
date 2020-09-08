DOCKER_RUN := docker run -u `id -u $(USER)`:`id -g $(USER)` --rm -v $(shell pwd):/app

help: ## Show this help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

format: ## Run pre-commit hooks to format code
	 pre-commit run --all-files

args ?=  -n auto -vvv --cov ssdir
test: ## Run tests
	pytest $(args)

shell: ## Run poetry shell
	poetry shell

build: ## Build docker image
	bash -c 'read -sp "PyPI trasee_rd password: " && docker build --build-arg PYPI_PASSWORD=$(REPLY) --build-arg UID=`id -u $(USER)` --build-arg GID=`id -g $(USER)` -f Dockerfile -t ssdir:latest .'

gpu ?= 3
ssdir_args ?= --config-file config.yml train
run: ## Run model
	$(DOCKER_RUN) --gpus '"device=$(gpu)"' --shm-size 24G ssdir:latest $(ssdir_args)
