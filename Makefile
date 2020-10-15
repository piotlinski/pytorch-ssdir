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
	poetry build -f wheel && docker build -f Dockerfile -t piotrekzie100/dev:ssdir .

gpu ?= 3
ssdir_args ?= ssdir --config-file config.yml train
run: ## Run model
	$(DOCKER_RUN) --gpus '"device=$(gpu)"' --shm-size 24G piotrekzie100/dev:ssdir $(ssdir_args)
