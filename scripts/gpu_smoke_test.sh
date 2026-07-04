#!/usr/bin/env bash
set -euo pipefail

echo "Host GPU:"
nvidia-smi

echo
echo "Docker GPU:"
docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
