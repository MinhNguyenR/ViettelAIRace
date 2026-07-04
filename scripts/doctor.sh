#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3.11 -m llm_race_lab.cli doctor
