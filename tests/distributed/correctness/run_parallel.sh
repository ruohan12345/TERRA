#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SEQ_CORRECTNESS_SUITE="parallel"
export CASES="${PARALLEL_CASES:-topo_wp topo_wp_sp topo_wp_tp topo_wp_sp_tp}"

echo "===== Running parallel correctness cases only ====="
echo "CASES=${CASES}"

exec bash "${SCRIPT_DIR}/run.sh" "$@"
