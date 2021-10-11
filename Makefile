DOCKER_RUN := docker run --rm -v $(shell pwd):/app
LOCAL_USER := -e LOCAL_USER_ID=`id -u $(USER)` -e LOCAL_GROUP_ID=`id -g $(USER)`
tag = piotrekzie100/dev:ssdir
DOCKER_ARGS ?=

help:  ## Show this help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

format:  ## Run pre-commit hooks to format code
	 pre-commit run --all-files

WANDB_API_KEY ?=
build.dev:  ## Build docker development image
	docker build -f dockerfiles/Dockerfile.dev -t $(tag)-dev .

build.prod:  ## Build docker production image
	docker build --build-arg WANDB_API_KEY=$(WANDB_API_KEY) -f dockerfiles/Dockerfile.prod -t $(tag) .

build.basic: ## Build basic docker image with all dependencies
	docker build --build-arg WANDB_API_KEY=$(WANDB_API_KEY) -f dockerfiles/Dockerfile.basic -t piotrekzie100/dev:basic .

shell:  ## Run basic docker dev shell
	$(DOCKER_RUN) -it piotrekzie100/dev:basic /bin/bash

args ?=  -n auto -vvv --cov pytorch_ssdir
test:  ## Run tests
	poetry run pytest $(args)

gpu ?= 3
ssdir_args ?= ssdir --default_root_dir runs
run:  ## Run model
	$(DOCKER_RUN) $(LOCAL_USER) $(DOCKER_ARGS) --gpus '"device=$(gpu)"' --shm-size 24G $(tag) $(ssdir_args)

cmd ?= python3 train.py $(ssdir_args)
run.basic:  ## Run model using basic docker
	$(DOCKER_RUN) $(LOCAL_USER) $(DOCKER_ARGS) --gpus '"device=$(gpu)"' --shm-size 24G --cpus 16 piotrekzie100/dev:basic $(cmd)
