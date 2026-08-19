#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RESULT_DIR="${SEQ_CORRECTNESS_RESULT_DIR:-${SEQ_CORRECTNESS_OUT_DIR:-tests/distributed/correctness/results/latest}}"
CONFIG_DIR="${SEQ_CORRECTNESS_CONFIG_DIR:-tests/distributed/correctness/configs/generated}"
OUTPUT_DIR="${SEQ_CORRECTNESS_OUTPUT_DIR:-${RESULT_DIR}}"
BASELINE="${BASELINE:-seq_fsdp}"
CASES="${CASES:-seq_fsdp topo_wp topo_wp_sp topo_wp_tp topo_wp_sp_tp}"

cd "${ROOT_DIR}"

echo "===== seq_ddp_vs_parallel_correctness eval ====="
echo "RESULT_DIR=${RESULT_DIR}"
echo "CONFIG_DIR=${CONFIG_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "BASELINE=${BASELINE}"
echo "CASES=${CASES}"

python "${SCRIPT_DIR}/analyze_correctness.py" \
    --result_dir "${RESULT_DIR}" \
    --config_dir "${CONFIG_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --baseline "${BASELINE}" \
    --cases ${CASES}

echo "===== eval outputs ====="
echo "${OUTPUT_DIR}/summary.csv"
echo "${OUTPUT_DIR}/correctness_report.md"
echo "${OUTPUT_DIR}/loss_curves.png"
echo "${OUTPUT_DIR}/loss_delta_vs_baseline.png"
