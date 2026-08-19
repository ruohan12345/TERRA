# TERRA

[![arXiv](https://img.shields.io/badge/arXiv-2608.15211-b31b1b.svg)](https://arxiv.org/abs/2608.15211)

**Paper:** [TERRA: A Hierarchical Parallel Training and Memory Orchestration Framework for High-Resolution AI-based Earth Modeling](https://arxiv.org/pdf/2608.15211)

TERRA is a distributed training system for high-resolution Earth forecasting Transformers. The artifact contains the SAWSTP parallel runtime, row-major ragged window assignment, multi-level WP/SP/TP execution, and profiling-guided memory orchestration used by the accompanying TPDS paper.

This directory is a publication-oriented source snapshot derived from the internal TERRA research tree. It intentionally excludes datasets, dataset-construction code, benchmark drivers, paper assets, checkpoints, machine addresses, and generated outputs.

## Public model boundary

The production Wenhai model implementation and model-specific training artifacts
used in the paper cannot be redistributed. They are not included in this public
snapshot. Instead, the repository provides `credit_hierarchical_swin`, a public
reference workload that combines the checked-in Swin Transformer and patch
embedding/recovery operators with CREDIT FuXi-style convolutional down/up
sampling blocks. The serial and distributed smoke tests use this reference model
to validate TERRA's layout routes, WP/SP/TP execution, halo exchange,
checkpointing, and activation offloading without releasing Wenhai source code or
weights.

The reference workload exercises the same classes of hierarchical operators, but
it is not a drop-in release of the production Wenhai implementation or its
checkpoints. CREDIT-derived components and their Apache-2.0 license are recorded
in `credit/NOTICE.md` and `credit/LICENSE`. Swin Transformer attribution and its
MIT license are recorded in `THIRD_PARTY_NOTICES.md` and
`licenses/SWIN_TRANSFORMER_LICENSE`.

## Repository layout

```text
core/          Distributed runtime, checkpointing, and parallel layouts
models/        Public reference hierarchical Swin model and Transformer implementation
credit/        CREDIT-derived domain-parallel utilities
dataloader/    Runtime GLORYS loaders and environment-based path resolution
optimizer/     FSDP/DDP/TP optimizer construction
profiler/      Memory and performance instrumentation
train_scripts/ Training and finetuning Python entry points
configs/       Curated public YAML configurations
tests/         Unit and distributed correctness tests
scripts/       User-facing launch helpers
test_env.sh    Fake-input correctness environment
test_g.sh      Serial/parallel torchrun launcher
```

## Installation

The reference environment uses Python 3.10, PyTorch 2.4, and CUDA 12.4.

```bash
conda env create -f environment.yml
conda activate terra
pip install -e .
pytest -q tests/unit
```

Before publishing or transferring the artifact, run the full static and unit
validation pass:

```bash
bash scripts/validate_release.sh
```

FlashAttention is optional for ordinary tests but required when a YAML file sets `USE_FLASH_ATTENTION: true`.

## Correctness smoke test

The checked-in launcher defaults to a one-node, eight-GPU parallel run with synthetic input, so no dataset is needed:

```bash
source test_env.sh
bash test_g.sh
```

Set `RUN_SEQ=1` and `RUN_PARALLEL=1` to run both correctness paths. Multi-node runs may override `NUM_NODES`, `NODE_RANK`, `MASTER_ADDR`, and `NPROC_PER_NODE` after sourcing `test_env.sh`.

## External data

No private path is embedded in this snapshot. Configure GLORYS through environment variables:

```bash
export TERRA_GLORYS_MM_ROOT=/path/to/sequential_fp16
export TERRA_GLORYS_ZS_ROOT=/path/to/zscore_sequential_fp16
export TERRA_GLORYS_PARALLEL_ROOT=/path/to/prepared_window_parallel
export TERRA_GLORYS_MASK_PATH=/path/to/mask.pt
```

Prepared `.pt` files are external to this repository. Runtime paths are supplied through the variables above; preprocessing and redistribution remain the dataset provider's responsibility.

## Release status

This public snapshot provides a reference workload for validating TERRA without
redistributing the production model. Environment-specific multi-GPU validation
items are tracked in `RELEASE_CHECKLIST.md`.

## License

TERRA's original code is released under the Apache License 2.0. Third-party
components retain their respective licenses and notices as documented in
`THIRD_PARTY_NOTICES.md`, `credit/NOTICE.md`, and `licenses/`.
