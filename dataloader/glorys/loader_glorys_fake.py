import ast

import torch
import torch.utils.data as data

from utils import get_padded_shape


_MAX_TORCH_SEED = (1 << 63) - 1
_DEFAULT_RANDOM_ROW_BLOCK = 64


def _block_seed(base_seed, sample_index, state_index, block_index):
    """Derive a stable CPU RNG seed from global sample coordinates."""
    value = int(base_seed) & _MAX_TORCH_SEED
    for coordinate in (sample_index, state_index, block_index):
        value = (
            value * 6364136223846793005
            + (int(coordinate) + 1) * 1442695040888963407
        ) & _MAX_TORCH_SEED
    return value


def _fill_deterministic_random_rows(
    destination,
    *,
    base_seed,
    sample_index,
    state_index,
    raw_row_start,
    raw_row_end,
    global_height,
    global_width,
    channels,
    row_block_size=_DEFAULT_RANDOM_ROW_BLOCK,
):
    """Fill a raw-row interval with the same values as the global fake field.

    RNG streams are anchored to fixed global row blocks. A DMP-local rank can
    therefore generate only its overlapping raw rows while remaining exactly
    equal to the corresponding slice of the full-sequence fake dataset.
    """
    raw_row_start = int(raw_row_start)
    raw_row_end = int(raw_row_end)
    expected_shape = (raw_row_end - raw_row_start, global_width, channels)
    if tuple(destination.shape) != expected_shape:
        raise ValueError(
            f"destination shape={tuple(destination.shape)} does not match "
            f"raw interval shape={expected_shape}"
        )
    if raw_row_start < 0 or raw_row_end > global_height or raw_row_start > raw_row_end:
        raise ValueError(
            f"invalid raw row interval [{raw_row_start}, {raw_row_end}) "
            f"for height={global_height}"
        )

    first_block = raw_row_start // row_block_size
    last_block = (raw_row_end + row_block_size - 1) // row_block_size
    for block_index in range(first_block, last_block):
        block_start = block_index * row_block_size
        block_end = min(block_start + row_block_size, global_height)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _block_seed(base_seed, sample_index, state_index, block_index)
        )
        block = torch.randn(
            (block_end - block_start, global_width, channels),
            generator=generator,
            dtype=destination.dtype,
        )
        overlap_start = max(raw_row_start, block_start)
        overlap_end = min(raw_row_end, block_end)
        if overlap_end <= overlap_start:
            continue
        source_start = overlap_start - block_start
        source_end = overlap_end - block_start
        destination_start = overlap_start - raw_row_start
        destination_end = overlap_end - raw_row_start
        destination[destination_start:destination_end].copy_(
            block[source_start:source_end]
        )


class FakeGLORYSSequentialDataset(data.Dataset):
    """Deterministic fake GLORYS reference workload sequential dataset.

    Shape contract:
      pretrain:  (input, label), each [H, W, C]
      finetune:  sequence tensor [lead_time + 1, H, W, C]

    When ``random`` is enabled, every sample is reproducible from ``seed`` and
    its global sample/state/row-block coordinates. The block-addressable stream
    is shared with ``FakeGLORYSWindowLinearDataset``.
    """

    def __init__(
        self,
        length=1000,
        lead_time=1,
        height=2041,
        width=4320,
        channels=93,
        dtype=torch.float16,
        return_sequence=False,
        random=False,
        seed=1234,
        random_row_block=_DEFAULT_RANDOM_ROW_BLOCK,
    ):
        if lead_time < 1:
            raise ValueError(f"lead_time must be >= 1, got {lead_time}")

        self.length = int(length)
        self.lead_time = int(lead_time)
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        self.dtype = dtype
        self.return_sequence = bool(return_sequence)
        self.random = bool(random)
        self.seed = int(seed)
        self.random_row_block = int(random_row_block)
        if self.random_row_block < 1:
            raise ValueError("random_row_block must be >= 1")

    def __len__(self):
        return self.length

    def _fake_state(self, sample_index, state_index):
        shape = (self.height, self.width, self.channels)
        if not self.random:
            return torch.zeros(shape, dtype=self.dtype)
        state = torch.empty(shape, dtype=self.dtype)
        _fill_deterministic_random_rows(
            state,
            base_seed=self.seed,
            sample_index=sample_index,
            state_index=state_index,
            raw_row_start=0,
            raw_row_end=self.height,
            global_height=self.height,
            global_width=self.width,
            channels=self.channels,
            row_block_size=self.random_row_block,
        )
        return state

    def __getitem__(self, idx):
        sample_index = int(idx)
        if self.return_sequence:
            sequence = torch.empty(
                (self.lead_time + 1, self.height, self.width, self.channels),
                dtype=self.dtype,
            )
            for state_index in range(self.lead_time + 1):
                sequence[state_index].copy_(
                    self._fake_state(sample_index, state_index)
                )
            return sequence

        return (
            self._fake_state(sample_index, 0),
            self._fake_state(sample_index, 1),
        )


class FakeGLORYSWindowLinearDataset(data.Dataset):
    """Deterministic rank-local fake finetune data for window-linear GLORYS.

    The returned sequence is already padded, patchified, and WP/DMP sharded:

      [lead_time + 1, local_patch_h, patch_w, channels * patch_size**2]

    Only raw rows overlapping this rank are generated. Padding is always zero,
    and valid values are exactly equal to the corresponding shard produced from
    ``FakeGLORYSSequentialDataset`` with the same seed and sample index.
    """

    def __init__(
        self,
        model_archi_params,
        other_params,
        wp_rank,
        length=1000,
        lead_time=1,
        channels=93,
        dtype=torch.float16,
        random=False,
        seed=1234,
        random_row_block=_DEFAULT_RANDOM_ROW_BLOCK,
    ):
        if lead_time < 1:
            raise ValueError(f"lead_time must be >= 1, got {lead_time}")

        self.length = int(length)
        self.lead_time = int(lead_time)
        self.channels = int(channels)
        self.dtype = dtype
        self.random = bool(random)
        self.seed = int(seed)
        self.random_row_block = int(random_row_block)
        if self.random_row_block < 1:
            raise ValueError("random_row_block must be >= 1")

        embedding_parallel_type = other_params["embedding_parallel_type"]
        if embedding_parallel_type != "window_linear":
            raise ValueError(
                "Rank-local GLORYS fake data requires window_linear, got "
                f"{embedding_parallel_type}"
            )

        wp_topo_value = other_params["wp_topo"]
        if isinstance(wp_topo_value, str):
            wp_topo_value = ast.literal_eval(wp_topo_value)
        wp_topo = tuple(int(value) for value in wp_topo_value)
        if len(wp_topo) != 2 or wp_topo[1] != 1:
            raise ValueError(
                "Rank-local GLORYS fake data currently requires wp_topo=(m, 1), "
                f"got {wp_topo}"
            )
        self.wp_size = wp_topo[0] * wp_topo[1]
        self.wp_rank = int(wp_rank)
        if self.wp_rank < 0 or self.wp_rank >= self.wp_size:
            raise ValueError(
                f"wp_rank={self.wp_rank} is outside [0, {self.wp_size})"
            )

        self.height = int(model_archi_params["height"])
        self.width = int(model_archi_params["width"])
        self.patch_size = int(model_archi_params["patch_size"])
        window_size = int(model_archi_params["window_size"])
        padding_scale = int(model_archi_params.get("padding_scale", 1))

        # Match dataloader_utils._get_resolved_padded_shape exactly: model
        # padding_spec takes precedence, then other_params, then padded_shape.
        padding_spec = model_archi_params.get("padding_spec", None)
        if padding_spec is None:
            padding_spec = other_params.get("padding_spec", None)
        if padding_spec is not None:
            requested_padded_shape = padding_spec.get("padded_shape", None)
        else:
            requested_padded_shape = model_archi_params.get("padded_shape", None)

        _need_padding, initial_padding, padded_shape = get_padded_shape(
            self.height,
            self.width,
            self.patch_size,
            window_size,
            padding_scale=padding_scale,
            padded_shape=requested_padded_shape,
        )
        self.initial_padding = tuple(int(value) for value in initial_padding)
        self.padded_shape = tuple(int(value) for value in padded_shape)
        padded_h, padded_w = self.padded_shape
        if padded_h % self.patch_size or padded_w % self.patch_size:
            raise ValueError(
                f"padded_shape={self.padded_shape} must be divisible by "
                f"patch_size={self.patch_size}"
            )
        patch_h = padded_h // self.patch_size
        if patch_h % self.wp_size:
            raise ValueError(
                f"patch_h={patch_h} must be divisible by wp_size={self.wp_size}"
            )

        self.local_patch_h = patch_h // self.wp_size
        self.patch_w = padded_w // self.patch_size
        self.local_pixel_h = self.local_patch_h * self.patch_size
        self.feature_channels = self.channels * self.patch_size * self.patch_size
        self.sample_shape = (
            self.lead_time + 1,
            self.local_patch_h,
            self.patch_w,
            self.feature_channels,
        )

        padding_left, _padding_right, padding_top, _padding_bottom = (
            self.initial_padding
        )
        self.padding_left = padding_left
        local_pixel_start = self.wp_rank * self.local_pixel_h
        local_pixel_end = local_pixel_start + self.local_pixel_h
        valid_start = padding_top
        valid_end = padding_top + self.height
        overlap_start = max(local_pixel_start, valid_start)
        overlap_end = min(local_pixel_end, valid_end)
        self.raw_row_start = max(0, overlap_start - padding_top)
        self.raw_row_end = max(self.raw_row_start, overlap_end - padding_top)
        self.local_row_start = max(0, overlap_start - local_pixel_start)
        self.local_row_end = self.local_row_start + (
            self.raw_row_end - self.raw_row_start
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample_index = int(idx)
        if not self.random:
            return torch.zeros(self.sample_shape, dtype=self.dtype)

        sequence = torch.empty(self.sample_shape, dtype=self.dtype)
        padded_w = self.padded_shape[1]
        for state_index in range(self.lead_time + 1):
            local_pixels = torch.zeros(
                (self.local_pixel_h, padded_w, self.channels),
                dtype=self.dtype,
            )
            if self.raw_row_end > self.raw_row_start:
                valid_destination = local_pixels[
                    self.local_row_start : self.local_row_end,
                    self.padding_left : self.padding_left + self.width,
                ]
                _fill_deterministic_random_rows(
                    valid_destination,
                    base_seed=self.seed,
                    sample_index=sample_index,
                    state_index=state_index,
                    raw_row_start=self.raw_row_start,
                    raw_row_end=self.raw_row_end,
                    global_height=self.height,
                    global_width=self.width,
                    channels=self.channels,
                    row_block_size=self.random_row_block,
                )

            source = local_pixels.view(
                self.local_patch_h,
                self.patch_size,
                self.patch_w,
                self.patch_size,
                self.channels,
            ).permute(0, 2, 1, 3, 4)
            destination = sequence[state_index].view(
                self.local_patch_h,
                self.patch_w,
                self.patch_size,
                self.patch_size,
                self.channels,
            )
            destination.copy_(source)
        return sequence
