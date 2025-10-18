#!/usr/bin/env sh

set -e

cd "$(dirname "$0")/"

RUN_NAME="$(. ./.env && echo "$RUN_NAME")"

GEN_RUN_NAME=0
if [ -z "$RUN_NAME" ]; then
	# shellcheck disable=SC2155
	export RUN_NAME="$(date '+%Y-%m-%d %H-%M-%S-%N %Z')"
	GEN_RUN_NAME=1
fi
echo "RUN_NAME=$RUN_NAME"

RUN_DIR="./runs/$RUN_NAME"
mkdir -p "$RUN_DIR"

cp ./.env "$RUN_DIR"

# TODO: the copied .env file might had changed in the mean time, therefore not containing the originally non-generated RUN_NAME variable.
if [ "$GEN_RUN_NAME" -eq 1 ]; then
	tail -c1 "$RUN_DIR/.env" | read -r _ || echo >> "$RUN_DIR/.env"
	echo "RUN_NAME=\"$RUN_NAME\"" >> "$RUN_DIR/.env"
fi

set -o allexport
. "$RUN_DIR/.env"
set +o allexport

if [ -z "${SKIP_BUILD+false}" ] && [ -z "${SKIP_GEN+false}" ]; then
	echo Building the Compose generator image...

	docker build -t localhost/p2psims/dfl/compose-gen ./compose-gen/

	echo Built the Compose generator image.
else
	echo Skipped building the Compose generator image.
fi

if [ -z "${SKIP_GEN+false}" ]; then
	echo Generating the Compose...

	# TODO: read the target Compose file name from the `-f` or `--file` Docker Compose option, and adhere to it.
	docker run --rm \
		--gpus all \
		-v ./compose-gen/:/opt/p2psims/dfl/compose-gen/ \
		--env-file ./.env \
		-e BUILD_DIR="$(realpath --relative-to "$RUN_DIR" ./)" \
		-e LOGS_DIR='./' \
		-e DATA_DIR="$(realpath --relative-to "$RUN_DIR" ./data/)" \
		-e REPO_DIR="$(realpath --relative-to "$RUN_DIR" ./)" \
		-e RUN_DIR='./' \
		-e PREPARE_DATA_DISTRIBUTION \
		localhost/p2psims/dfl/compose-gen:latest
	mv ./compose-gen/compose.yaml "$RUN_DIR"

	echo Generated the Compose.
fi

if [ "$#" -gt 0 ]; then
	cd "$RUN_DIR"
	docker compose "$@"
fi