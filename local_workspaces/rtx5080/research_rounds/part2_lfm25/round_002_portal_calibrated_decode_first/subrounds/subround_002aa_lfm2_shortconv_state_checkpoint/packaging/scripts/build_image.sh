#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$(git -C "$PACKAGE_DIR" rev-parse --show-toplevel)"
IMAGE="${R002AA_IMAGE:-localhost/part2-lfm25-r002aa:shortconv-sparse-nopad}"

docker build \
  --file "$PACKAGE_DIR/Dockerfile" \
  --tag "$IMAGE" \
  "$ROOT_DIR"

docker image inspect "$IMAGE" --format 'image={{.RepoTags}} id={{.Id}} size={{.Size}}'
