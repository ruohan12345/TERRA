#!/usr/bin/env python3
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def env_str(name, default):
    return os.environ.get(name, default)


def list_literal(values):
    return "[" + ", ".join(str(v) for v in values) + "]"


def checkpoint_layers(num_layers):
    return list(range(num_layers))


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


CASES = {
    "seq_fsdp": {
        "label": "Serial/FSDP",
        "topology_label": "Serial/FSDP",
        "model_type": "sequential",
        "optimizer_config": 7,
        "wp": None,
        "sp": None,
        "tp": None,
    },
    "topo_wp": {
        "label": "TERRA WP",
        "topology_label": "Topo(8,1,1,1)",
        "model_type": "parallel",
        "optimizer_config": 7,
        "wp_topo": "(8, 1)",
        "xfmr_wp_topo": "(8, 1)",
        "xfmr_sp_size": 1,
        "tensor_parallel_size": 1,
    },
    "topo_wp_sp": {
        "label": "TERRA WP+SP",
        "topology_label": "Topo(4,1,2,1)",
        "model_type": "parallel",
        "optimizer_config": 7,
        "wp_topo": "(8, 1)",
        "xfmr_wp_topo": "(4, 1)",
        "xfmr_sp_size": 2,
        "tensor_parallel_size": 1,
    },
    "topo_wp_tp": {
        "label": "TERRA WP+TP",
        "topology_label": "Topo(4,1,1,2)",
        "model_type": "parallel",
        "optimizer_config": 9,
        "wp_topo": "(8, 1)",
        "xfmr_wp_topo": "(4, 1)",
        "xfmr_sp_size": 1,
        "tensor_parallel_size": 2,
    },
    "topo_wp_sp_tp": {
        "label": "TERRA WP+SP+TP",
        "topology_label": "Topo(2,1,2,2)",
        "model_type": "parallel",
        "optimizer_config": 9,
        "wp_topo": "(8, 1)",
        "xfmr_wp_topo": "(2, 1)",
        "xfmr_sp_size": 2,
        "tensor_parallel_size": 2,
    },
}


def validate_case(case_name, cfg):
    if cfg["model_type"] != "parallel":
        return
    wp_m = int(cfg["xfmr_wp_topo"].strip("()").split(",")[0].strip())
    sp = int(cfg["xfmr_sp_size"])
    tp = int(cfg["tensor_parallel_size"])
    if wp_m * sp * tp != 8:
        raise ValueError(f"{case_name}: expected WP*SP*TP=8, got {wp_m}*{sp}*{tp}")
    if cfg["wp_topo"] != "(8, 1)":
        raise ValueError(f"{case_name}: wp_topo must be (8, 1), got {cfg['wp_topo']}")


def build_yaml(case_name, case_cfg):
    model_architecture = env_str("MODEL_ARCHITECTURE", "credit_hierarchical_swin")
    patch_size = env_int("PATCH_SIZE", 4)
    window_size = env_int("WINDOW_SIZE", 8)
    padded_h = env_int("PADDED_SHAPE_H", 2304)
    padded_w = env_int("PADDED_SHAPE_W", 4352)
    num_layers = env_int("SMALL_NUM_LAYERS", 10)
    embedding_dim = env_int("SMALL_EMBEDDING_DIM", 1024)
    num_heads = env_int("SMALL_NUM_HEADS", 8)
    activation_mode = env_str("ACTIVATION_MODE", "torch_recompute")
    activation_policy = env_str("ACTIVATION_POLICY", "uniform")
    sampling_down = env_str("SAMPLING_CHECKPOINT_DOWN", "D1")
    sampling_up = env_str("SAMPLING_CHECKPOINT_UP", "U1")
    checkpoint = checkpoint_layers(num_layers)

    lines = [
        f"description: {quote('correctness_' + case_name)}",
        "experiment_name: 'seq_ddp_vs_parallel_correctness'",
        f"case_name: {quote(case_name)}",
        f"case_label: {quote(case_cfg['label'])}",
        f"topology_label: {quote(case_cfg['topology_label'])}",
        "global_gpu_count: 16",
        "data_parallel_size: 2",
        "train_path: './tests/distributed/correctness/train_runs/" + case_name + "'",
        "saved_data_path: './tests/distributed/correctness/saved_data/" + case_name + "'",
        "ckpt_start: False",
        "load_saved_data_dict: False",
        "ckpt_epoch: 0",
        "clip_grad: False",
        "use_fake_input: False",
        "",
        "task_type: 'glorys'",
        f"model_architecture: {quote(model_architecture)}",
        "num_channels: 93",
        f"optimizer_config: {case_cfg['optimizer_config']}",
        "fsdp_checkpoint_type: 'full'",
        "FSDP_CONFIG7_PREFETCH_POLICY: 'none'",
        "FSDP_SAMPLING_WRAPPER_CFG: 0",
        "precision: 'fp16'",
        "half_model: False",
        "use_splited_data: False",
        "norm_type: 'mm'",
        f"model_type: {quote(case_cfg['model_type'])}",
    ]

    if case_cfg["model_type"] == "parallel":
        lines.extend([
            f"wp_topo: {quote(case_cfg['wp_topo'])}",
            "embedding_parallel_type: 'window_linear'",
            f"xfmr_wp_topo: {quote(case_cfg['xfmr_wp_topo'])}",
            f"xfmr_sp_size: {case_cfg['xfmr_sp_size']}",
            f"tensor_parallel_size: {case_cfg['tensor_parallel_size']}",
            "sp_tp_placement: 'tp_first'",
            "window_topology: 'm1'",
            "window_assignment_mode: 'terra_m1_ragged_auto'",
        ])

    lines.extend([
        "",
        "micro_batch_size: 1",
        "padding_policy: explicit",
        f"padded_shape: [{padded_h}, {padded_w}]",
        "padding_scale: 8",
        f"num_layers: {num_layers}",
        f"embedding_dim: {embedding_dim}",
        f"num_heads: {num_heads}",
        f"patch_size: {patch_size}",
        f"window_size: {window_size}",
        "kaiming_init: True",
        "num_workers: 2",
        "learning_rate: 0.0001",
        "",
        "use_attn_mask: True",
        "use_relative_position_bias: True",
        "USE_FLASH_ATTENTION: False",
        "",
        "activation:",
        f"  mode: {activation_mode}",
        f"  policy: {activation_policy}",
        f"  checkpoint_layers: {list_literal(checkpoint)}",
        "  sampling_checkpoint:",
        f"    down: {sampling_down}",
        f"    up: {sampling_up}",
        "",
        "loss_func: 'L1'",
        "start_epoch: 0",
        "num_pre_train_epochs: 300",
        "data_resolution: -1",
        "",
    ])
    return "\n".join(lines)


def main():
    config_dir = Path(env_str("SEQ_CORRECTNESS_CONFIG_DIR", str(SCRIPT_DIR / "configs" / "generated")))
    if not config_dir.is_absolute():
        config_dir = ROOT_DIR / config_dir
    config_dir.mkdir(parents=True, exist_ok=True)

    selected = env_str("CASES", " ".join(CASES)).split()
    unknown = [name for name in selected if name not in CASES]
    if unknown:
        raise ValueError(f"unknown cases: {unknown}")

    manifest = []
    for case_name in selected:
        case_cfg = CASES[case_name]
        validate_case(case_name, case_cfg)
        path = config_dir / f"{case_name}.yaml"
        path.write_text(build_yaml(case_name, case_cfg), encoding="utf-8")
        manifest.append(case_name)
        print(f"[generate_configs] wrote {path}")

    (config_dir / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"[generate_configs] manifest: {config_dir / 'manifest.txt'}")


if __name__ == "__main__":
    main()
