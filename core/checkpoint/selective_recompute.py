from core.checkpoint.activation import (
    activation_config_for_layer,
    normalize_activation_config,
    normalize_activation_mode,
)


class SelectiveRecomputeScheduler:
    def __init__(
            self,
            config=None,
            depth=0,
            input_resolution=None,
            hidden_dim=None,
            module_name="transformer",
            ):
        self.config = normalize_activation_config(config=config)
        self.depth = int(depth)
        self.input_resolution = input_resolution
        self.hidden_dim = hidden_dim
        self.module_name = module_name

        self.enabled = bool(self.config.get("enabled", False))
        self.mode = self.config.get("mode", "none")
        self.policy = str(self.config.get("policy", "none")).strip().lower()
        self.impl = self.mode
        self.checkpoint_every_n_layers = int(self.config.get("checkpoint_every_n_layers", 4))

        self.checkpoint_segments = self._build_checkpoint_segments()
        self.block_indices = {
            idx
            for start, end in self.checkpoint_segments
            for idx in range(start, end)
        }

    def set_checkpoint_layers(self, checkpoint_layers):
        self.config["checkpoint_layers"] = checkpoint_layers
        self.config.pop("checkpoint_segments", None)
        self.checkpoint_segments = self._build_checkpoint_segments()
        self.block_indices = {
            idx
            for start, end in self.checkpoint_segments
            for idx in range(start, end)
        }

    def set_checkpoint_plan(
            self,
            checkpoint_layers,
            activation_mode=None,
            offload_config=None,
            segment_activation_modes=None,
            ):
        runtime_config = dict(self.config)
        if activation_mode is not None:
            runtime_config["mode"] = str(activation_mode)
        if offload_config is not None:
            runtime_config["offload"] = dict(offload_config)
        runtime_config["checkpoint_layers"] = list(checkpoint_layers)
        runtime_config.pop("checkpoint_segments", None)
        if segment_activation_modes is not None:
            segment_activation_modes = list(segment_activation_modes)
            segments = self._segments_from_checkpoint_layers(
                runtime_config["checkpoint_layers"]
            )
            if len(segment_activation_modes) != len(segments):
                raise ValueError(
                    "segment_activation_modes must match checkpoint_layers: "
                    f"{len(segment_activation_modes)} != {len(segments)}"
                )
            runtime_config["layer_modes"] = {
                int(start): normalize_activation_mode(mode)
                for (start, _), mode in zip(
                    segments, segment_activation_modes
                )
            }
        self.config = normalize_activation_config(config=runtime_config)
        self.enabled = bool(self.config.get("enabled", False))
        self.mode = self.config.get("mode", "none")
        self.impl = self.mode
        self.set_checkpoint_layers(checkpoint_layers)

    def _clip_segment(self, start, end):
        start = int(start)
        end = int(end)
        start = max(0, min(start, self.depth))
        end = max(0, min(end, self.depth))
        if end <= start:
            return None
        return start, end

    def _validate_segments(self, segments):
        out = []
        last_end = -1
        for start, end in segments:
            clipped = self._clip_segment(start, end)
            if clipped is None:
                continue
            start, end = clipped
            if start < last_end:
                raise ValueError(
                    "activation checkpoint segments must be non-overlapping and sorted, "
                    f"but got segment ({start}, {end}) after previous end {last_end}"
                )
            out.append((start, end))
            last_end = end
        return out

    def _singleton_segments(self, indices):
        return self._validate_segments((idx, idx + 1) for idx in sorted(set(indices)))

    def _segment_from_checkpoint_item(self, item):
        if isinstance(item, int):
            return int(item), int(item) + 1

        text = str(item).strip()
        key = text.lower()
        if key in ("d0", "d1", "d2", "u0", "u1", "u2"):
            return None

        try:
            idx = int(text)
            return idx, idx + 1
        except ValueError:
            pass

        if "-" not in text:
            raise ValueError(f"Unsupported activation checkpoint layer entry: {item}")

        lhs, rhs = [part.strip() for part in text.split("-", 1)]
        if not lhs or not rhs:
            raise ValueError(f"Malformed activation checkpoint layer range: {item}")

        start = int(lhs)
        end_inclusive = int(rhs)
        if end_inclusive < start:
            raise ValueError(f"Invalid activation checkpoint layer range: {item}")
        return start, end_inclusive + 1

    def _segments_from_checkpoint_layers(self, explicit):
        segments = []
        for item in explicit:
            segment = self._segment_from_checkpoint_item(item)
            if segment is not None:
                segments.append(segment)
        return self._validate_segments(segments)

    def _build_checkpoint_segments(self):
        if not self.enabled or self.policy == "none":
            return []

        if self.policy == "larger_wrapper":
            return self._validate_segments(self.config.get("checkpoint_segments", []))

        explicit = self.config.get("checkpoint_layers", None)
        if explicit is None:
            explicit = self.config.get("block_indices", None)
        if explicit is not None:
            return self._segments_from_checkpoint_layers(explicit)

        if self.policy == "full_checkpoint" or bool(self.config.get("full_checkpoint", False)):
            return self._singleton_segments(range(self.depth))

        if self.policy == "uniform":
            n = max(1, self.checkpoint_every_n_layers)
            return self._singleton_segments(idx for idx in range(self.depth) if idx % n == 0)

        raise ValueError(f"Unsupported recompute policy: {self.policy}")

    def should_checkpoint(self, block_idx):
        return self.enabled and int(block_idx) in self.block_indices

    def config_for_block(self, block_idx):
        return activation_config_for_layer(self.config, block_idx)

    def execution_segments(self):
        if not self.enabled:
            for idx in range(self.depth):
                yield idx, idx + 1, False
            return

        checkpoint_by_start = {start: end for start, end in self.checkpoint_segments}
        idx = 0
        while idx < self.depth:
            end = checkpoint_by_start.get(idx, None)
            if end is not None:
                yield idx, end, True
                idx = end
            else:
                yield idx, idx + 1, False
                idx += 1

    def summary(self):
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "policy": self.policy,
            "depth": self.depth,
            "input_resolution": self.input_resolution,
            "hidden_dim": self.hidden_dim,
            "block_indices": sorted(self.block_indices),
            "checkpoint_segments": [list(segment) for segment in self.checkpoint_segments],
            "layer_modes": dict(sorted(self.config.get("layer_modes", {}).items())),
            "segment_activation_modes": [
                self.config_for_block(start)["mode"]
                for start, _ in self.checkpoint_segments
            ],
            "sampling_checkpoint": dict(self.config.get("sampling_checkpoint", {"down": "none", "up": "none"})),
            "sampling_modes": dict(self.config.get("sampling_modes", {})),
        }
