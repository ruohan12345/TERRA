#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

NUM_NODES="${SEQ_CORRECTNESS_NUM_NODES:-${NUM_NODES:-2}}"
NPROC_PER_NODE="${SEQ_CORRECTNESS_NPROC_PER_NODE:-${NPROC_PER_NODE:-8}}"
SEQ_NPROC_PER_NODE="${SEQ_NPROC_PER_NODE:-2}"
DATA_PARALLEL_SIZE="${SEQ_CORRECTNESS_DATA_PARALLEL_SIZE:-${DATA_PARALLEL_SIZE:-2}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29671}"

RUN_STEPS="${RUN_STEPS:-${STEPS:-1000}}"
CASES="${CASES:-seq_fsdp topo_wp topo_wp_sp topo_wp_tp topo_wp_sp_tp}"
SEQ_CORRECTNESS_SUITE="${SEQ_CORRECTNESS_SUITE:-custom}"
CONFIG_DIR="${SEQ_CORRECTNESS_CONFIG_DIR:-tests/distributed/correctness/configs/generated}"
OUT_DIR="${SEQ_CORRECTNESS_OUT_DIR:-tests/distributed/correctness/results/latest}"
QUIET_METRICS="${QUIET_METRICS:-1}"
CSV_ONLY="${CSV_ONLY:-1}"
DISABLE_TORCH_PROF="${DISABLE_TORCH_PROF:-1}"
DISABLE_RANK_LOG_REDIRECT="${DISABLE_RANK_LOG_REDIRECT:-1}"
SKIP_FINISHED="${SKIP_FINISHED:-1}"
RESET_RESULTS="${RESET_RESULTS:-0}"
RESET_WAIT_SECONDS="${RESET_WAIT_SECONDS:-5}"

if (( NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
    echo "Invalid NODE_RANK=${NODE_RANK}; expected 0 <= NODE_RANK < ${NUM_NODES}" >&2
    exit 2
fi

if (( NUM_NODES > 1 )) && [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
    echo "Please set MASTER_ADDR=<node0_ip> for this 2-node experiment." >&2
    exit 2
fi

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export DISABLE_RANK_LOG_REDIRECT

CONFIG_DIR_ABS="$(realpath -m "${CONFIG_DIR}")"
OUT_DIR_ABS="$(realpath -m "${OUT_DIR}")"
RESULTS_ROOT_ABS="$(realpath -m "${SCRIPT_DIR}/results")"
mkdir -p "${CONFIG_DIR_ABS}" "${RESULTS_ROOT_ABS}" "${OUT_DIR_ABS}"

if [[ "${NODE_RANK}" == "0" ]]; then
    python "${SCRIPT_DIR}/generate_configs.py"
fi

manifest="${CONFIG_DIR_ABS}/manifest.txt"
for _ in $(seq 1 120); do
    [[ -s "${manifest}" ]] && break
    sleep 1
done
if [[ ! -s "${manifest}" ]]; then
    echo "Missing generated configs manifest: ${manifest}" >&2
    exit 2
fi

read -r -a CASE_ARRAY <<< "${CASES}"
TOTAL_CASES="${#CASE_ARRAY[@]}"

case_nproc_per_node() {
    case "$1" in
        seq_fsdp) echo "${SEQ_NPROC_PER_NODE}" ;;
        *) echo "${NPROC_PER_NODE}" ;;
    esac
}

case_port_offset() {
    case "$1" in
        seq_fsdp) echo 0 ;;
        topo_wp) echo 1 ;;
        topo_wp_sp) echo 2 ;;
        topo_wp_tp) echo 3 ;;
        topo_wp_sp_tp) echo 4 ;;
        *) echo "$2" ;;
    esac
}

status_ok() {
    [[ -f "$1/status.ok" ]]
}

safe_reset_case_dir() {
    local case_dir="$1"
    local case_dir_abs
    case_dir_abs="$(realpath -m "${case_dir}")"
    case "${case_dir_abs}" in
        "${RESULTS_ROOT_ABS}"/*) rm -rf "${case_dir_abs}" ;;
        *) echo "Refusing to reset case_dir outside ${RESULTS_ROOT_ABS}: ${case_dir_abs}" >&2; exit 2 ;;
    esac
}

run_case() {
    local case_name="$1"
    local idx="$2"
    local cfg="${CONFIG_DIR_ABS}/${case_name}.yaml"
    local case_dir="${OUT_DIR_ABS}/${case_name}"
    local nproc_per_node
    local port
    local log_path
    nproc_per_node="$(case_nproc_per_node "${case_name}")"
    port="$((MASTER_PORT_BASE + $(case_port_offset "${case_name}" "${idx}")))"
    log_path="${case_dir}/node${NODE_RANK}.log"

    if [[ ! -f "${cfg}" ]]; then
        echo "Missing config: ${cfg}" >&2
        exit 2
    fi

    if [[ "${RESET_RESULTS}" == "1" && "${NODE_RANK}" == "0" ]]; then
        safe_reset_case_dir "${case_dir}"
    fi
    if [[ "${NUM_NODES}" != "1" ]]; then
        sleep "${RESET_WAIT_SECONDS}"
    fi
    mkdir -p "${case_dir}"

    if [[ "${SKIP_FINISHED}" == "1" ]] && status_ok "${case_dir}"; then
        [[ "${NODE_RANK}" == "0" ]] && echo "[skip] ${case_name}"
        return 0
    fi

    if [[ "${NODE_RANK}" == "0" ]]; then
        cp -f "${cfg}" "${case_dir}/config.yaml"
        echo
        echo "[$((idx + 1))/${TOTAL_CASES}] running ${case_name}"
        echo "cfg=${cfg}"
        echo "out=${case_dir}"
        echo "steps=${RUN_STEPS} num_nodes=${NUM_NODES} node_rank=${NODE_RANK} nproc_per_node=${nproc_per_node} dp_size=${DATA_PARALLEL_SIZE}"
    fi

    local metrics_args=(--metrics_csv "${case_dir}/loss.csv")
    if [[ "${CSV_ONLY}" == "1" ]]; then
        metrics_args+=(--metrics_no_json)
    else
        metrics_args+=(--metrics_json "${case_dir}/metadata.json")
    fi

    local profiler_args=()
    if [[ "${DISABLE_TORCH_PROF}" == "1" ]]; then
        profiler_args+=(--disable_torch_prof)
    fi

    local quiet_args=()
    if [[ "${QUIET_METRICS}" == "1" ]]; then
        quiet_args+=(--quiet_metrics)
    fi

    local rank_log_args=()
    if [[ "${DISABLE_RANK_LOG_REDIRECT}" == "1" ]]; then
        rank_log_args+=(--disable_rank_log_redirect)
    fi

    local cmd=(
        torchrun
        --nproc_per_node="${nproc_per_node}"
        --nnodes="${NUM_NODES}"
        --node_rank="${NODE_RANK}"
        --master_addr="${MASTER_ADDR}"
        --master_port="${port}"
        "train_scripts/test_pretrain.py"
        --data_parallel_group_size "${DATA_PARALLEL_SIZE}"
        --model_cfg "${cfg}"
        --steps "${RUN_STEPS}"
        "${metrics_args[@]}"
        "${profiler_args[@]}"
        "${quiet_args[@]}"
        "${rank_log_args[@]}"
    )

    set +e
    if [[ "${QUIET_METRICS}" == "1" ]]; then
        "${cmd[@]}" > "${log_path}" 2>&1
    else
        "${cmd[@]}" 2>&1 | tee "${log_path}"
    fi
    local status=$?
    set -e

    if [[ "${status}" == "0" ]]; then
        if [[ "${NODE_RANK}" == "0" ]]; then
            echo "ok" > "${case_dir}/status.ok"
            rm -f "${case_dir}/failed.txt"
            echo "finished ${case_name}"
        fi
    else
        echo "case ${case_name} failed on node ${NODE_RANK} with status ${status}" > "${case_dir}/failed_node${NODE_RANK}.txt"
        if [[ "${NODE_RANK}" == "0" ]]; then
            echo "case ${case_name} failed with status ${status}" > "${case_dir}/failed.txt"
            echo "[failed] ${case_name} status=${status}"
        fi
    fi
}

echo "===== seq_ddp_vs_parallel_correctness run ====="
echo "SUITE=${SEQ_CORRECTNESS_SUITE}"
echo "NODE_RANK=${NODE_RANK}/${NUM_NODES}, MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT_BASE=${MASTER_PORT_BASE}"
echo "RUN_STEPS=${RUN_STEPS}, DATA_PARALLEL_SIZE=${DATA_PARALLEL_SIZE}, CASES=${CASES}"
echo "CONFIG_DIR=${CONFIG_DIR_ABS}"
echo "OUT_DIR=${OUT_DIR_ABS}"

idx=0
for case_name in "${CASE_ARRAY[@]}"; do
    run_case "${case_name}" "${idx}"
    idx=$((idx + 1))
done

if [[ "${NODE_RANK}" == "0" ]]; then
    echo
    echo "Done. Evaluate with:"
    echo "  bash ${SCRIPT_DIR}/eval.sh"
fi
