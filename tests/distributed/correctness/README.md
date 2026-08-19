# seq_ddp_vs_parallel_correctness

This experiment checks convergence/loss consistency between a serial FSDP
baseline and TERRA window-parallel variants on 2 nodes / 16 GPUs.

## Cases

- `seq_fsdp`: Serial/FSDP baseline.
- `topo_wp`: TERRA WP, `Topo(8,1,1,1)`.
- `topo_wp_sp`: TERRA WP+SP, `Topo(4,1,2,1)`.
- `topo_wp_tp`: TERRA WP+TP, `Topo(4,1,1,2)`.
- `topo_wp_sp_tp`: TERRA WP+SP+TP, `Topo(2,1,2,2)`.

All parallel cases use:

- `wp_topo: '(8, 1)'`
- `window_topology: 'm1'`
- `window_assignment_mode: 'terra_m1_ragged_auto'`
- `WP * SP * TP = 8`

All cases use the same model, padded shape, mask/bias setting, and activation
checkpoint configuration.

## Run

Generate configs once:

```bash
export MASTER_ADDR=<node0_ip>
export NODE_RANK=0  # node0 uses 0, node1 uses 1
source tests/distributed/correctness/env.sh
python tests/distributed/correctness/generate_configs.py
```

Run the serial baseline separately on both nodes:

```bash
bash tests/distributed/correctness/run_seq.sh
```

Then run the parallel cases separately on both nodes:

```bash
bash tests/distributed/correctness/run_parallel.sh
```

After both nodes finish, run on one node:

```bash
bash tests/distributed/correctness/eval.sh
```

Outputs are written to `tests/distributed/correctness/results/latest`.
