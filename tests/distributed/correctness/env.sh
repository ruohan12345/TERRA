#!/usr/bin/env bash

# 2-node / 16-GPU convergence correctness experiment.
#
# Usage:
#   export MASTER_ADDR=<node0_ip>
#   export NODE_RANK=0        # node0/node1 set 0/1
#   source tests/distributed/correctness/env.sh
#   bash tests/distributed/correctness/run_seq.sh       # node0 only; node1 skips if invoked
#   bash tests/distributed/correctness/run_parallel.sh  # run on both nodes
#
# Evaluation after both nodes finish:
#   bash tests/distributed/correctness/eval.sh

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export NUM_NODES=2
export NPROC_PER_NODE=8
export SEQ_NPROC_PER_NODE=2
export DATA_PARALLEL_SIZE=2

export MASTER_PORT_BASE=29671

export SEQ_CASES="seq_fsdp"
export PARALLEL_CASES="topo_wp topo_wp_sp topo_wp_tp topo_wp_sp_tp"
export CASES="${SEQ_CASES} ${PARALLEL_CASES}"

export RUN_STEPS=1000
export STEPS="${RUN_STEPS}"

export MODEL_ARCHITECTURE="credit_hierarchical_swin"
export PATCH_SIZE=4
export WINDOW_SIZE=8
export PADDED_SHAPE_H=2304
export PADDED_SHAPE_W=4352

export SMALL_EMBEDDING_DIM=1024
export SMALL_NUM_LAYERS=10
export SMALL_NUM_HEADS=8

export ACTIVATION_MODE="torch_recompute"
export ACTIVATION_POLICY="uniform"
export SAMPLING_CHECKPOINT_DOWN="D1"
export SAMPLING_CHECKPOINT_UP="U1"

export SEQ_CORRECTNESS_OUT_DIR="tests/distributed/correctness/results/latest"
export SEQ_CORRECTNESS_CONFIG_DIR="tests/distributed/correctness/configs/generated"
export SEQ_CORRECTNESS_OUTPUT_DIR="${SEQ_CORRECTNESS_OUT_DIR}"

# Kept for compatibility with older commands after sourcing this env file.
export OUT_DIR="${SEQ_CORRECTNESS_OUT_DIR}"
export CONFIG_DIR="${SEQ_CORRECTNESS_CONFIG_DIR}"

export QUIET_METRICS=1
export CSV_ONLY=1
export DISABLE_RANK_LOG_REDIRECT=1
export DISABLE_TORCH_PROF=1

export SKIP_FINISHED=1
export RESET_RESULTS=0
export RESET_WAIT_SECONDS=5

export OMP_NUM_THREADS=1

echo "[seq_ddp_vs_parallel_correctness/env.sh] NUM_NODES=${NUM_NODES}, NPROC_PER_NODE=${NPROC_PER_NODE}, SEQ_NPROC_PER_NODE=${SEQ_NPROC_PER_NODE}"
echo "[seq_ddp_vs_parallel_correctness/env.sh] DATA_PARALLEL_SIZE=${DATA_PARALLEL_SIZE}, RUN_STEPS=${RUN_STEPS}"
echo "[seq_ddp_vs_parallel_correctness/env.sh] CASES=${CASES}"
echo "[seq_ddp_vs_parallel_correctness/env.sh] OUT_DIR=${SEQ_CORRECTNESS_OUT_DIR}"
