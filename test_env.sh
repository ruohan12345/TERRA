#!/usr/bin/env bash

# Public fake-input correctness environment. Override any value after sourcing.
export RUN_SEQ=0
export RUN_PARALLEL=1

export NUM_NODES=1
export NODE_RANK=0
export NPROC_PER_NODE=8
export DATA_PARALLEL_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29731

export SCRIPT="./train_scripts/test_pretrain.py"
export SEQ_MODEL_CFG="./configs/model/correctness/credit_hierarchical_swin_correctness_seq.yaml"
export PARALLEL_MODEL_CFG="./configs/model/correctness/credit_hierarchical_swin_correctness_para_m1.yaml"

export TERRA_USE_FAKE_INPUT=1
export STEPS=4
export OUT_DIR="./log/correctness"
export WRITE_CSV=1
export WRITE_METADATA=1
export QUIET_METRICS=0
export DISABLE_RANK_LOG_REDIRECT=1
export ENABLE_TORCH_PROF=0
export TORCH_PROF_WRITE_TRACE=0
export OMP_NUM_THREADS=1
