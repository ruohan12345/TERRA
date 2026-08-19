#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CALLER_NODE_RANK="${NODE_RANK:-0}"
if [[ "${CALLER_NODE_RANK}" != "0" ]]; then
    echo "[run_seq.sh] NODE_RANK=${CALLER_NODE_RANK}; seq baseline runs on node0 only, skip."
    exit 0
fi

export SEQ_CORRECTNESS_SUITE="seq"
export SEQ_CORRECTNESS_NUM_NODES=1
export NODE_RANK=0
export CASES="${SEQ_CASES:-seq_fsdp}"
export SEQ_NPROC_PER_NODE="${SEQ_NPROC_PER_NODE:-2}"

echo "===== Running sequential baseline only ====="
echo "CASES=${CASES}"
echo "SEQ_NUM_NODES=${SEQ_CORRECTNESS_NUM_NODES}"
echo "SEQ_NPROC_PER_NODE=${SEQ_NPROC_PER_NODE}"

exec bash "${SCRIPT_DIR}/run.sh" "$@"
