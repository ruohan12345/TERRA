#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${1:-${NODE_RANK:-0}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29731}"
RUN_SEQ="${RUN_SEQ:-0}"
RUN_PARALLEL="${RUN_PARALLEL:-1}"
SCRIPT="${SCRIPT:-./train_scripts/test_pretrain.py}"
SEQ_MODEL_CFG="${SEQ_MODEL_CFG:-./configs/model/correctness/credit_hierarchical_swin_correctness_seq.yaml}"
PARALLEL_MODEL_CFG="${PARALLEL_MODEL_CFG:-./configs/model/correctness/credit_hierarchical_swin_correctness_para_m1.yaml}"
STEPS="${STEPS:-4}"
OUT_DIR="${OUT_DIR:-./log/correctness}"
WRITE_CSV="${WRITE_CSV:-1}"
WRITE_METADATA="${WRITE_METADATA:-1}"
QUIET_METRICS="${QUIET_METRICS:-0}"
DISABLE_RANK_LOG_REDIRECT="${DISABLE_RANK_LOG_REDIRECT:-1}"
ENABLE_TORCH_PROF="${ENABLE_TORCH_PROF:-0}"

if (( NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
  echo "Invalid NODE_RANK=${NODE_RANK}; expected 0 <= NODE_RANK < ${NUM_NODES}" >&2
  exit 1
fi
if (( NUM_NODES * NPROC_PER_NODE % DATA_PARALLEL_SIZE != 0 )); then
  echo "world size must be divisible by DATA_PARALLEL_SIZE" >&2
  exit 1
fi
for path in "${SCRIPT}" "${SEQ_MODEL_CFG}" "${PARALLEL_MODEL_CFG}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
done

run_case() {
  local name="$1"
  local port="$2"
  local nproc="$3"
  local config="$4"
  local output_dir="${OUT_DIR}/${name,,}"
  local args=(
    --data_parallel_group_size "${DATA_PARALLEL_SIZE}"
    --model_cfg "${config}"
    --steps "${STEPS}"
  )

  mkdir -p "${output_dir}"
  [[ "${WRITE_CSV}" == 1 ]] && args+=(--metrics_csv "${output_dir}/loss.csv")
  if [[ "${WRITE_METADATA}" == 1 ]]; then
    args+=(--metrics_json "${output_dir}/metadata.json")
  else
    args+=(--metrics_no_json)
  fi
  [[ "${QUIET_METRICS}" == 1 ]] && args+=(--quiet_metrics)
  [[ "${DISABLE_RANK_LOG_REDIRECT}" == 1 ]] && args+=(--disable_rank_log_redirect)
  [[ "${ENABLE_TORCH_PROF}" == 0 ]] && args+=(--disable_torch_prof)

  echo "===== ${name}: node ${NODE_RANK}/${NUM_NODES}, nproc=${nproc} ====="
  torchrun \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${port}" \
    --nnodes "${NUM_NODES}" \
    --node_rank "${NODE_RANK}" \
    --nproc_per_node "${nproc}" \
    "${SCRIPT}" "${args[@]}"
}

if [[ "${RUN_SEQ}" == 1 ]]; then
  if (( DATA_PARALLEL_SIZE % NUM_NODES != 0 )); then
    echo "Sequential mode requires DATA_PARALLEL_SIZE divisible by NUM_NODES" >&2
    exit 1
  fi
  run_case "SEQ" "${MASTER_PORT_BASE}" "$((DATA_PARALLEL_SIZE / NUM_NODES))" "${SEQ_MODEL_CFG}"
fi

if [[ "${RUN_PARALLEL}" == 1 ]]; then
  run_case "PARALLEL" "$((MASTER_PORT_BASE + 1))" "${NPROC_PER_NODE}" "${PARALLEL_MODEL_CFG}"
fi
