#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python -m pytest -q tests/unit
python -m py_compile \
  core/parallel/window_assignment.py \
  models/model_utils/get_model.py \
  models/attention.py
