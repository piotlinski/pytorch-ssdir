DOCKER_RUN := docker run --rm -v $(shell pwd):/app
LOCAL_USER := -e LOCAL_USER_ID=`id -u $(USER)` -e LOCAL_GROUP_ID=`id -g $(USER)`
tag = piotrekzie100/dev:ssdir

help: ## Show this help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

format: ## Run pre-commit hooks to format code
	 pre-commit run --all-files

PYPI_USERNAME ?= trasee_rd
PYPI_PASSWORD ?=
WANDB_API_KEY ?=
build.dev: ## Build docker development image
	docker build --build-arg PYPI_USERNAME=${PYPI_USERNAME} --build-arg PYPI_PASSWORD=${PYPI_PASSWORD} -f dockerfiles/Dockerfile.dev -t $(tag)-dev .

build.prod: ## Build docker production image
	docker build --build-arg PYPI_USERNAME=${PYPI_USERNAME} --build-arg PYPI_PASSWORD=${PYPI_PASSWORD} --build-arg WANDB_API_KEY=$(WANDB_API_KEY)  -f dockerfiles/Dockerfile.prod -t $(tag) .

shell: ## Run docker dev shell
	$(DOCKER_RUN) -it $(tag)-dev /bin/bash

args ?=  -n auto -vvv --cov ssdir
test: ## Run tests
	poetry run pytest $(args)

gpu ?= 3
ssdir_args ?= ssdir --default_root_dir runs
run: ## Run model
	$(DOCKER_RUN) $(LOCAL_USER) --gpus '"device=$(gpu)"' --shm-size 24G $(tag) $(ssdir_args)
