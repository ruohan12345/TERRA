from copy import deepcopy

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from core.checkpoint.boundary_offload import checkpoint as boundary_offload_checkpoint


_MODE_ALIASES = {
    "none": "none",
    "no": "none",
    "off": "none",
    "disabled": "none",
    "torch": "torch_recompute",
    "torch_checkpoint": "torch_recompute",
    "torch_recompute": "torch_recompute",
    "boundary_offload": "cpu_boundary_offload_recompute",
    "cpu_boundary_offload": "cpu_boundary_offload_recompute",
    "cpu_boundary_offload_recompute": "cpu_boundary_offload_recompute",
    "cpu_offload_boundary": "cpu_boundary_offload_recompute",
}


def normalize_activation_mode(mode):
    if mode is None:
        return "none"
    key = str(mode).strip().lower()
    if key not in _MODE_ALIASES:
        raise ValueError(f"Unsupported activation mode: {mode}")
    return _MODE_ALIASES[key]


def _mode_from_legacy_selective(config):
    if not bool(config.get("enabled", False)):
        return "none"
    backend = config.get("backend", config.get("impl", "torch"))
    return normalize_activation_mode(backend)


def _normalize_sampling_mode(value, axis):
    if value is None:
        return "none"

    key = str(value).strip().lower()
    if key in ("", "none", "no", "off", "false", "0", "disabled"):
        return "none"

    aliases = {
        "down": {
            "d0": "D0",
            "fused": "D0",
            "patch_embed_down": "D0",
            "patch_sampling": "D0",
            "d1": "D1",
            "split": "D1",
            "granular": "D1",
        },
        "up": {
            "u0": "U0",
            "fused": "U0",
            "up_patch_recovery": "U0",
            "patch_sampling": "U0",
            "u1": "U1",
            "split": "U1",
            "granular": "U1",
        },
    }
    if key not in aliases[axis]:
        valid = ", ".join(["none"] + sorted(set(aliases[axis].values())))
        raise ValueError(f"Unsupported {axis} sampling checkpoint mode: {value}. Valid modes: {valid}")
    return aliases[axis][key]


def _sampling_modes_from_config(config):
    deprecated_keys = (
        "checkpoint_sampling_blocks",
        "checkpoint_patch_sampling_blocks",
        "checkpoint_patch_blocks",
        "checkpoint_patch_embed_down",
        "checkpoint_up_patch_recovery",
    )
    used_deprecated = [key for key in deprecated_keys if key in config]
    if used_deprecated:
        keys = ", ".join(used_deprecated)
        raise ValueError(
            f"Deprecated activation sampling config key(s): {keys}. "
            "Use activation.sampling_checkpoint.down/up instead."
        )

    sampling = config.get("sampling_checkpoint", None)
    if sampling is None:
        return "none", "none"

    if not isinstance(sampling, dict):
        raise ValueError("activation.sampling_checkpoint must be a dict with optional down/up fields")
    down_mode = _normalize_sampling_mode(sampling.get("down", None), "down")
    up_mode = _normalize_sampling_mode(sampling.get("up", None), "up")
    return down_mode, up_mode


def _parse_index_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]
    return [int(value)]


def _as_checkpoint_item_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _parse_transformer_segment(value):
    if isinstance(value, int):
        return int(value), int(value) + 1

    text = str(value).strip()
    if text == "":
        raise ValueError("Empty activation checkpoint layer entry")

    try:
        idx = int(text)
        return idx, idx + 1
    except ValueError:
        pass

    if "-" not in text:
        raise ValueError(f"Unsupported activation checkpoint layer entry: {value}")

    lhs, rhs = [part.strip() for part in text.split("-", 1)]
    if not lhs or not rhs:
        raise ValueError(f"Malformed activation checkpoint layer range: {value}")

    start = int(lhs)
    end_inclusive = int(rhs)
    if end_inclusive < start:
        raise ValueError(f"Invalid activation checkpoint layer range: {value}")
    return start, end_inclusive + 1


def _parse_larger_wrapper_plan(checkpoint_layers, down_mode, up_mode):
    segments = []
    for item in _as_checkpoint_item_list(checkpoint_layers):
        item_text = str(item).strip()
        item_key = item_text.lower()
        if item_key in ("d0", "d1"):
            down_mode = _normalize_sampling_mode(item_key, "down")
            continue
        if item_key in ("u0", "u1"):
            up_mode = _normalize_sampling_mode(item_key, "up")
            continue
        segments.append(_parse_transformer_segment(item))

    return down_mode, up_mode, segments


def _normalize_layer_modes(raw):
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("activation.layer_modes must be a dict")

    layer_modes = {}
    for key, value in raw.items():
        key_str = str(key).strip()
        try:
            layer_idx = int(key_str)
            layer_modes[layer_idx] = normalize_activation_mode(value)
            continue
        except ValueError:
            pass

        mode = normalize_activation_mode(key_str)
        for layer_idx in _parse_index_list(value):
            layer_modes[int(layer_idx)] = mode
    return layer_modes


def _normalize_sampling_strategy_modes(raw):
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("activation.sampling_modes must be a dict with optional down/up fields")

    out = {}
    for key, value in raw.items():
        axis = str(key).strip().lower()
        if axis not in ("down", "up"):
            raise ValueError(f"Unsupported activation.sampling_modes key: {key}. Expected down/up")
        out[axis] = normalize_activation_mode(value)
    return out


def normalize_activation_config(model_config=None, config=None):
    """Normalize activation checkpoint/offload configuration.

    Public API:
        activation:
          mode: none | torch_recompute | cpu_boundary_offload_recompute
          checkpoint_layers: [...]
          sampling_checkpoint:
            down: none | D0 | D1
            up: none | U0 | U1

    Legacy config is accepted only as an input migration path:
        selective_recompute:
          enabled: ...
          impl: torch | ours
    """
    if config is not None:
        raw = deepcopy(config)
    elif model_config is not None and model_config.get("activation", None) is not None:
        raw = deepcopy(model_config["activation"])
    elif model_config is not None and model_config.get("selective_recompute", None) is not None:
        raw = deepcopy(model_config["selective_recompute"])
        raw["mode"] = _mode_from_legacy_selective(raw)
    else:
        raw = {"mode": "none"}

    if isinstance(raw, str):
        raw = {"mode": raw}

    mode = raw.get("mode", None)
    if mode is None:
        enabled = raw.get("enabled", None)
        if enabled is False:
            mode = "none"
        else:
            mode = raw.get("backend", raw.get("impl", "none"))
    mode = normalize_activation_mode(mode)

    config = dict(raw)
    config["mode"] = mode
    config["enabled"] = mode != "none"

    if not config["enabled"]:
        config["policy"] = "none"
        config["checkpoint_layers"] = []
        config["block_indices"] = []
        config["checkpoint_segments"] = []
        config["checkpoint_patch_embed"] = False
        config["checkpoint_patch_recovery"] = False
        config["checkpoint_down_mode"] = "none"
        config["checkpoint_up_mode"] = "none"
        config["sampling_checkpoint"] = {"down": "none", "up": "none"}
        return config

    if config.get("policy", None) is None:
        config["policy"] = "full_checkpoint"
    else:
        config["policy"] = config.get("policy", "full_checkpoint")

    if config.get("checkpoint_layers", None) is None and config.get("block_indices", None) is not None:
        config["checkpoint_layers"] = config["block_indices"]

    down_mode, up_mode = _sampling_modes_from_config(config)
    config["checkpoint_segments"] = []
    if str(config.get("policy", "")).strip().lower() == "larger_wrapper":
        down_mode, up_mode, segments = _parse_larger_wrapper_plan(
            config.get("checkpoint_layers", None),
            down_mode,
            up_mode,
        )
        config["checkpoint_segments"] = segments

    config["checkpoint_down_mode"] = down_mode
    config["checkpoint_up_mode"] = up_mode
    config["sampling_checkpoint"] = {"down": down_mode, "up": up_mode}
    config["layer_modes"] = _normalize_layer_modes(config.get("layer_modes", None))
    config["sampling_modes"] = _normalize_sampling_strategy_modes(config.get("sampling_modes", None))

    config["checkpoint_patch_embed"] = bool(config.get("checkpoint_patch_embed", False))
    config["checkpoint_patch_recovery"] = bool(config.get("checkpoint_patch_recovery", False))
    return config


def activation_config_with_mode(config, mode):
    out = dict(normalize_activation_config(config=config))
    out["mode"] = normalize_activation_mode(mode)
    out["enabled"] = out["mode"] != "none"
    return out


def activation_config_for_layer(config, layer_idx):
    base = normalize_activation_config(config=config)
    mode = base.get("layer_modes", {}).get(int(layer_idx), None)
    if mode is None:
        return base
    return activation_config_with_mode(base, mode)


def activation_config_for_sampling(config, axis):
    base = normalize_activation_config(config=config)
    mode = base.get("sampling_modes", {}).get(str(axis).strip().lower(), None)
    if mode is None:
        return base
    return activation_config_with_mode(base, mode)


def activation_checkpoint(function, *args, activation_config=None, mode=None):
    if activation_config is None:
        activation_config = {"mode": mode}
    config = normalize_activation_config(config=activation_config)
    mode = config["mode"]

    if mode == "none":
        return function(*args)

    if mode == "torch_recompute":
        return torch_checkpoint(function, *args, use_reentrant=False, preserve_rng_state=True)

    if mode == "cpu_boundary_offload_recompute":
        return boundary_offload_checkpoint(function, *args, offload_config=config.get("offload", None))

    raise ValueError(f"Unsupported activation mode: {mode}")
