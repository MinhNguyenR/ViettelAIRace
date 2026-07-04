#!/usr/bin/env bash
set -euo pipefail

SUBMISSION_DIR="${1:-submissions/phase1}"

required=(
  "docker-compose.yml"
  ".env.example"
  "README.md"
)

for file in "${required[@]}"; do
  if [[ ! -f "${SUBMISSION_DIR}/${file}" ]]; then
    echo "Missing ${SUBMISSION_DIR}/${file}" >&2
    exit 1
  fi
done

docker compose -f "${SUBMISSION_DIR}/docker-compose.yml" --env-file "${SUBMISSION_DIR}/.env.example" config >/tmp/llm-race-submission-compose.yaml

echo "Submission skeleton is structurally valid: ${SUBMISSION_DIR}"
