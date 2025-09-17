help:
	cat Makefile

# start (or restart) the services
server: .FORCE
	docker compose down --remove-orphans || true;
	docker compose up

# start (or restart) the services in detached mode
server-detached: .FORCE
	docker compose down || true;
	docker compose up -d

# build or rebuild the services WITHOUT cache (updated with latest versions)
build: .FORCE
	chmod 777 Gemfile.lock
	docker compose stop || true; docker compose rm || true;
	docker buildx build --platform=linux/arm64,linux/amd64 --no-cache -t fastai/fastpages-nbdev:latest -f _action_files/fastpages-nbdev.Dockerfile . --load
	docker buildx build --platform=linux/arm64,linux/amd64 --no-cache -t fastai/fastpages-jekyll:latest -f _action_files/fastpages-jekyll.Dockerfile . --load
	docker compose build --force-rm --no-cache

# rebuild the services WITH cache (optimized for development)
quick-build: .FORCE
	docker compose stop || true;
	docker buildx build --platform=linux/arm64,linux/amd64 -t fastai/fastpages-nbdev:latest -f _action_files/fastpages-nbdev.Dockerfile . --load
	docker buildx build --platform=linux/arm64,linux/amd64 -t fastai/fastpages-jekyll:latest -f _action_files/fastpages-jekyll.Dockerfile . --load
	docker compose build

# build for current platform only (faster for development)
build-local: .FORCE
	chmod 777 Gemfile.lock
	docker compose stop || true; docker compose rm || true;
	docker build --no-cache -t fastai/fastpages-nbdev:latest -f _action_files/fastpages-nbdev.Dockerfile .
	docker build --no-cache -t fastai/fastpages-jekyll:latest -f _action_files/fastpages-jekyll.Dockerfile .
	docker compose build --force-rm --no-cache

# clean up old images and containers for efficiency
clean: .FORCE
	docker compose down --remove-orphans || true;
	docker system prune -f
	docker image prune -f

# update base images to latest versions
update-images: .FORCE
	docker pull python:3.12-slim-bookworm
	docker pull jekyll/jekyll:4.2.2 

# convert word & nb without Jekyll services
convert: .FORCE
	docker compose up converter

# stop all containers
stop: .FORCE
	docker compose stop
	docker ps | grep fastpages | awk '{print $1}' | xargs docker stop

# remove all containers
remove: .FORCE
	docker compose stop  || true; docker compose rm || true;

# get shell inside the notebook converter service (Must already be running)
bash-nb: .FORCE
	docker compose exec watcher /bin/bash

# get shell inside jekyll service (Must already be running)
bash-jekyll: .FORCE
	docker compose exec jekyll /bin/bash

# restart just the Jekyll server
restart-jekyll: .FORCE
	docker compose restart jekyll

.FORCE:
