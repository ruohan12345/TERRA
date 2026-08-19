import os

import torch
from torch.utils.data import DataLoader

from dataloader.glorys.loader_glorys_seq import GLORYSSequentialDataset
from dataloader.glorys.loader_glorys_parallel import GLORYSWindowLinearDataset
from dataloader.glorys.loader_glorys_fake import (
    FakeGLORYSSequentialDataset,
    FakeGLORYSWindowLinearDataset,
)

from dataloader.glorys_paths import (
    get_glorys_parallel_file_path_prefix,
    resolve_glorys_sequential_root,
)
from dataloader.glorys_utils import get_wp_slice_from_bhwc


from utils import get_padded_shape


def _get_resolved_padded_shape(model_archi_params, other_params=None):
    padding_spec = model_archi_params.get('padding_spec', None)
    if padding_spec is None and other_params is not None:
        padding_spec = other_params.get('padding_spec', None)
    if padding_spec is not None:
        return padding_spec.get('padded_shape', None)
    return model_archi_params.get('padded_shape', None)

def get_dataset_for_task(manager, task_type, model_type, model_archi_params, other_params, status, for_pretrain = True, lead_time = 1):
    if task_type != 'glorys':
        raise ValueError(f"Unsupported task_type={task_type!r}; only 'glorys' is supported")

    if task_type == 'glorys':
        norm_type = other_params.get('norm_type', 'zs')
        data_precision = other_params.get('data_precision', 'fp16')
        explicit_root = other_params.get('glorys_data_root', None)
        GLORYS_SEQUENTIAL_ROOT = resolve_glorys_sequential_root(
            norm_type=norm_type,
            data_precision=data_precision,
            explicit_root=explicit_root,
        )


        if model_type == 'sequential':
            cur_dataset = GLORYSSequentialDataset(
                root_full_path = GLORYS_SEQUENTIAL_ROOT,
                status=status,
                norm_type=norm_type,
                data_precision=data_precision,
                lead_time=lead_time,
                return_sequence=not for_pretrain,
            )
        elif model_type == 'parallel':
            GLORYS_PARALLEL_ROOT = get_glorys_parallel_file_path_prefix(
                other_params['embedding_parallel_type'],
                model_archi_params['height'],
                model_archi_params['width'],
                other_params['wp_topo'],
                model_archi_params['patch_size'],
                model_archi_params['window_size'],
                model_archi_params['padding_scale'],
                norm_type=norm_type,
                data_precision=data_precision,
                padding_spec=other_params.get('padding_spec', model_archi_params.get('padding_spec', None)),
            )
            cur_dataset = GLORYSWindowLinearDataset(
                root_full_path=GLORYS_PARALLEL_ROOT,
                wp_rank=manager.get_wp_rank(),
                status=status,
                norm_type=norm_type,
                data_precision=data_precision,
                lead_time=lead_time,
                return_sequence=not for_pretrain,
            )
        else:
            print('unsupported model_type', model_type, 'for glorys dataset in get_dataset_for_task')
            exit(0)


    return cur_dataset


def get_dataloader_for_task(
                            task_type,
                            micro_batch_size,
                            use_splited_data = True,
                            status = 0,
                            num_workers = 1,
                            simplified = True,
                            use_uv = True,

                            manager = None,
                            model_type = 'sequential',
                            model_archi_params = None,
                            other_params = None,
                            use_fake_input = False,
                            ):
    if task_type != 'glorys':
        raise ValueError(f"Unsupported task_type={task_type!r}; only 'glorys' is supported")

    data_parallel_group_size = manager.get_dp_group_size()

    if use_fake_input:
        if task_type == 'glorys':
            cur_dataset = FakeGLORYSSequentialDataset(
                height=model_archi_params['height'],
                width=model_archi_params['width'],
                lead_time=1,
                return_sequence=False,
                random=True,
            )
    else:
        if model_type == 'sequential':
            cur_dataset = get_dataset_for_task(manager, task_type, 'sequential', model_archi_params, other_params, status, for_pretrain = True)
        elif model_type == 'hybrid' or model_type =='parallel':
            if use_splited_data:
                cur_dataset = get_dataset_for_task(manager, task_type, model_type, model_archi_params, other_params, status, for_pretrain = True)
            else:
                cur_dataset = get_dataset_for_task(manager, task_type, 'sequential', model_archi_params, other_params, status, for_pretrain = True)
        else:
            print('we do not support model_type', model_type, 'in get_dataloader_for_task')
            exit(0)

    cur_sampler = torch.utils.data.distributed.DistributedSampler(
        cur_dataset,
        num_replicas = data_parallel_group_size,
        rank = manager.get_dp_rank(),
        shuffle=True,
        drop_last=False,
        )
    cur_dataloader = DataLoader(cur_dataset,
                            num_workers=num_workers,
                            shuffle=False,
                            pin_memory=True,
                            batch_size=micro_batch_size,
                            drop_last=True,
                            sampler=cur_sampler)

    return cur_dataloader


def get_finetune_dataloader_for_task(
                            task_type,
                            micro_batch_size,
                            use_splited_data = True,
                            status = 0,
                            num_workers = 1,

                            manager = None,
                            model_type = 'sequential',
                            model_archi_params = None,
                            other_params = None,
                            use_fake_input = False,
                            lead_time = 1,
                            fake_input_random = True,
                            fake_input_dmp_local = False,
                            fake_input_seed = 1234,
                            pin_memory = True,
                            ):
    if task_type != 'glorys':
        raise ValueError(f"Unsupported task_type={task_type!r}; only 'glorys' is supported")

    data_parallel_group_size = manager.get_dp_group_size()

    if use_fake_input:
        if task_type == 'glorys':
            if bool(fake_input_dmp_local):
                if model_type not in ('parallel', 'hybrid'):
                    raise ValueError(
                        "fake_input_dmp_local requires a parallel or hybrid GLORYS model"
                    )
                cur_dataset = FakeGLORYSWindowLinearDataset(
                    model_archi_params=model_archi_params,
                    other_params=other_params,
                    wp_rank=manager.get_wp_rank(),
                    lead_time=lead_time,
                    random=bool(fake_input_random),
                    seed=int(fake_input_seed),
                )
            else:
                cur_dataset = FakeGLORYSSequentialDataset(
                    height=model_archi_params['height'],
                    width=model_archi_params['width'],
                    lead_time=lead_time,
                    return_sequence=True,
                    random=bool(fake_input_random),
                    seed=int(fake_input_seed),
                )


    else:
        if model_type == 'sequential':
            cur_dataset = get_dataset_for_task(manager, task_type, 'sequential', model_archi_params, other_params, status, for_pretrain = False, lead_time = lead_time)
        elif model_type == 'hybrid' or model_type =='parallel':
            if use_splited_data:
                cur_dataset = get_dataset_for_task(manager, task_type, model_type, model_archi_params, other_params, status, for_pretrain = False, lead_time = lead_time)
            else:
                cur_dataset = get_dataset_for_task(manager, task_type, 'sequential', model_archi_params, other_params, status, for_pretrain = False, lead_time = lead_time)

    cur_sampler = torch.utils.data.distributed.DistributedSampler(
        cur_dataset,
        num_replicas = data_parallel_group_size,
        rank = manager.get_dp_rank(),
        shuffle=True,
        drop_last=False,
        )
    cur_dataloader = DataLoader(cur_dataset,
                                num_workers=num_workers,
                                shuffle=False,
                                pin_memory=bool(pin_memory),
                                batch_size=micro_batch_size,
                                drop_last=True,
                                sampler=cur_sampler)
    return cur_dataloader


def _format_glorys_no_sample_seq_tensor(tensor, target_height, target_width, channel_count=93):
    if tensor.ndim != 4:
        raise RuntimeError(f"Expected batched GLORYS tensor with 4 dims, got shape={tuple(tensor.shape)}")
    if (
        tensor.shape[1] >= channel_count
        and tensor.shape[2] >= target_height
        and tensor.shape[3] >= target_width
    ):
        return tensor[:, :channel_count, :target_height, :target_width].permute(0, 2, 3, 1).contiguous()
    if (
        tensor.shape[-1] >= channel_count
        and tensor.shape[1] >= target_height
        and tensor.shape[2] >= target_width
    ):
        return tensor[:, :target_height, :target_width, :channel_count].contiguous()
    raise RuntimeError(
        f"Cannot format GLORYS no-sample sequential tensor shape={tuple(tensor.shape)} "
        f"to BHWC target=({target_height}, {target_width}, {channel_count})"
    )


def _format_glorys_no_sample_seq_sequence(tensor, target_height, target_width, channel_count=93):
    if tensor.ndim != 5:
        raise RuntimeError(f"Expected batched GLORYS sequence tensor with 5 dims, got shape={tuple(tensor.shape)}")
    if (
        tensor.shape[2] >= channel_count
        and tensor.shape[3] >= target_height
        and tensor.shape[4] >= target_width
    ):
        return tensor[:, :, :channel_count, :target_height, :target_width].permute(0, 1, 3, 4, 2).contiguous()
    if (
        tensor.shape[-1] >= channel_count
        and tensor.shape[2] >= target_height
        and tensor.shape[3] >= target_width
    ):
        return tensor[:, :, :target_height, :target_width, :channel_count].contiguous()
    raise RuntimeError(
        f"Cannot format GLORYS no-sample sequential sequence shape={tuple(tensor.shape)} "
        f"to BTHWC target=({target_height}, {target_width}, {channel_count})"
    )


def _need_glorys_no_sample_seq_format(task_specific_data_dict):
    return (
        task_specific_data_dict.get('model_type', None) == 'sequential'
        and task_specific_data_dict.get('model_architecture', None) == 'swin_reference'
    )


class GlorysLeadDoubleBuffer:
    """Sequential two-slot H2D prefetcher for long-lead GLORYS finetuning.

    The provider keeps Python references only to the current and next lead.
    CUDA/autograd retains any tensor that is still required for backward, so a
    buffer is never overwritten while the loss graph still owns it.
    """

    def __init__(self, sequence_tensor, device, dtype, lead_time):
        self.sequence_tensor = sequence_tensor
        self.device = torch.device(device)
        self.dtype = dtype
        self.lead_time = int(lead_time)
        if self.device.type != "cuda":
            raise RuntimeError("GLORYS lead double buffering requires CUDA")
        if sequence_tensor.is_cuda:
            raise RuntimeError("GLORYS lead double buffering expects a CPU sequence")
        if not sequence_tensor.is_pinned():
            raise RuntimeError(
                "GLORYS lead double buffering requires DataLoader pin_memory=true"
            )
        if sequence_tensor.ndim < 2 or sequence_tensor.shape[1] < self.lead_time + 1:
            raise RuntimeError(
                f"sequence shape={tuple(sequence_tensor.shape)} cannot provide "
                f"lead_time={self.lead_time}"
            )

        self.copy_stream = torch.cuda.Stream(device=self.device)
        self._slots = {}
        self._ready_events = {}
        self._timing_events = []
        self._last_index = -1
        self._schedule(0)

    def __len__(self):
        return self.lead_time

    def _copy_frame(self, frame):
        return frame.to(
            device=self.device,
            dtype=self.dtype,
            non_blocking=True,
        )

    def _schedule(self, lead_index):
        if lead_index >= self.lead_time or lead_index in self._slots:
            return
        start_event = torch.cuda.Event(enable_timing=True)
        ready_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.copy_stream):
            start_event.record(self.copy_stream)
            slot = {
                "label_tensor": self._copy_frame(
                    self.sequence_tensor[:, lead_index + 1]
                )
            }
            if lead_index == 0:
                slot["input_tensor"] = self._copy_frame(
                    self.sequence_tensor[:, 0]
                )
            ready_event.record(self.copy_stream)
        self._slots[lead_index] = slot
        self._ready_events[lead_index] = ready_event
        self._timing_events.append((start_event, ready_event))

    def __getitem__(self, lead_index):
        lead_index = int(lead_index)
        if lead_index != self._last_index + 1:
            raise RuntimeError(
                "GLORYS lead double buffer requires sequential access; "
                f"last={self._last_index}, requested={lead_index}"
            )
        if lead_index not in self._slots:
            self._schedule(lead_index)

        compute_stream = torch.cuda.current_stream(self.device)
        compute_stream.wait_event(self._ready_events[lead_index])
        slot = self._slots[lead_index]
        for tensor in slot.values():
            tensor.record_stream(compute_stream)

        previous = lead_index - 1
        if previous in self._slots:
            del self._slots[previous]
            del self._ready_events[previous]
        self._schedule(lead_index + 1)
        self._last_index = lead_index
        return slot

    def transfer_time_s(self):
        return sum(
            start.elapsed_time(end) for start, end in self._timing_events
        ) / 1000.0

    def release(self):
        self._slots.clear()
        self._ready_events.clear()
        self.sequence_tensor = None


def resolve_glorys_finetune_double_buffer(
    x,
    device,
    task_specific_data_dict,
    my_dtype,
    lead_time,
):
    sequence_tensor = x
    if _need_glorys_no_sample_seq_format(task_specific_data_dict):
        target_height = task_specific_data_dict['height']
        target_width = task_specific_data_dict['width']
        sequence_tensor = _format_glorys_no_sample_seq_sequence(
            sequence_tensor, target_height, target_width
        )
    return GlorysLeadDoubleBuffer(
        sequence_tensor,
        device=device,
        dtype=my_dtype,
        lead_time=lead_time,
    )

def resolve_required_tensors_from_dataloader(
    task_type,
    x,
    device,
    task_specific_data_dict,
    my_dtype,
    is_pretrain=True,
    lead_time=1,
    input_non_blocking=False,
):

    if task_type != 'glorys':
        raise ValueError(f"Unsupported task_type={task_type!r}; only 'glorys' is supported")

    if task_type == 'glorys':
        if is_pretrain:
            input_tensor, label_tensor = x
            if _need_glorys_no_sample_seq_format(task_specific_data_dict):
                target_height = task_specific_data_dict['height']
                target_width = task_specific_data_dict['width']
                input_tensor = _format_glorys_no_sample_seq_tensor(input_tensor, target_height, target_width)
                label_tensor = _format_glorys_no_sample_seq_tensor(label_tensor, target_height, target_width)
            if input_non_blocking:
                input_tensor = input_tensor.to(
                    device=device,
                    dtype=my_dtype,
                    non_blocking=True,
                )
                label_tensor = label_tensor.to(
                    device=device,
                    dtype=my_dtype,
                    non_blocking=True,
                )
            else:
                input_tensor = input_tensor.to(device).type(my_dtype)
                label_tensor = label_tensor.to(device).type(my_dtype)
            required_tensors = {
                'input_tensor': input_tensor,
                'label_tensor': label_tensor,
            }
        else:
            sequence_tensor = x
            if _need_glorys_no_sample_seq_format(task_specific_data_dict):
                target_height = task_specific_data_dict['height']
                target_width = task_specific_data_dict['width']
                sequence_tensor = _format_glorys_no_sample_seq_sequence(sequence_tensor, target_height, target_width)
            if input_non_blocking:
                # The Memory Orchestra input experiment receives pinned CPU
                # batches from DataLoader. Fuse the device and dtype copy so
                # CUDA can enqueue one asynchronous H2D transfer.
                sequence_tensor = sequence_tensor.to(
                    device=device,
                    dtype=my_dtype,
                    non_blocking=True,
                )
            else:
                sequence_tensor = sequence_tensor.to(device).type(my_dtype)
            required_tensors = []
            for cur_lead_time in range(0, lead_time):
                cur_required_tensors = {
                    'label_tensor': sequence_tensor[:, cur_lead_time + 1],
                }
                if cur_lead_time == 0:
                    cur_required_tensors['input_tensor'] = sequence_tensor[:, 0]
                required_tensors.append(cur_required_tensors)


    return required_tensors


def pad_tensor(tensor, initial_padding, data_format):
     with torch.no_grad():
        initial_pad = torch.nn.ZeroPad2d(initial_padding)
        if data_format=='NCHW':
            tensor = initial_pad(tensor).contiguous()
            return tensor

        elif data_format=='NHWC':
            print('we do not support NHWC in pad_tensor now')
            exit(0)
            '''
            input_tensor = input_tensor.permute(0, 3, 1, 2).contiguous()
            input_tensor = input_tensor.permute(0, 3, 1, 2).contiguous()
            label_tensor = label_tensor.permute(0, 3, 1, 2).contiguous()
            input_tensor = initial_pad(input_tensor).contiguous()
            label_tensor = initial_pad(label_tensor).contiguous()
            input_tensor = input_tensor.permute(0, 2, 3, 1).contiguous() # [1, 2112, 4320, 93]
            label_tensor = label_tensor.permute(0, 2, 3, 1).contiguous()
            '''
        else:
            print('unsupported data_format', data_format)
            exit(0)


def patch_fy(x, patch_size):
    B, h, w, hidden_dim = x.shape
    x = x.view(B, h//patch_size, patch_size, w//patch_size, patch_size, hidden_dim)# [1, 576, 2, 960, 2, 70]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, (h//patch_size), (w//patch_size), patch_size*patch_size*hidden_dim)
    return x

def get_patch_fy_slice_from_wp_rank(x, patch_size, wp_group_size, wp_rank):


    x = patch_fy(x, patch_size) # [1, 576, 960, 280]


    chunks = torch.chunk(x, wp_group_size, dim=1)

    #exit(0)

    return chunks[wp_rank].clone().contiguous()


def _cpu_tensor_to_nchw_for_glorys(tensor):
    if tensor.is_cuda:
        raise RuntimeError("CPU online split expects GLORYS tensors to stay on CPU before slicing")
    if tensor.ndim != 4:
        raise RuntimeError(f"Expected GLORYS tensor with batch dimension, got shape={tuple(tensor.shape)}")
    if tensor.shape[1] == 93:
        return tensor.contiguous()
    if tensor.shape[-1] == 93:
        return tensor.permute(0, 3, 1, 2).contiguous()
    raise RuntimeError(
        "Cannot infer GLORYS channel layout; expected NCHW or NHWC with 93 channels, "
        f"got shape={tuple(tensor.shape)}"
    )


def _move_local_cpu_tensor_to_device(tensor, device, dtype):
    tensor = tensor.to(dtype=dtype).contiguous()
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        tensor = tensor.pin_memory()
    return tensor.to(device=device, non_blocking=True)


def _glorys_window_linear_cpu_online_split_one(tensor, device, my_dtype, model_archi_params, other_params, manager):
    embedding_parallel_type = other_params['embedding_parallel_type']
    if embedding_parallel_type != 'window_linear':
        raise RuntimeError(f"CPU online split for GLORYS only supports window_linear, got {embedding_parallel_type}")

    wp_topo = other_params['wp_topo']
    if wp_topo[1] > 1:
        raise RuntimeError(f"CPU online split for GLORYS window_linear requires wp_topo=(m, 1), got {wp_topo}")

    patch_size = model_archi_params['patch_size']
    window_size = model_archi_params['window_size']
    resolved_padded_shape = _get_resolved_padded_shape(model_archi_params, other_params)
    _need_padding, initial_padding, padded_shape = get_padded_shape(
        model_archi_params['height'],
        model_archi_params['width'],
        patch_size,
        window_size,
        padding_scale=model_archi_params['padding_scale'],
        padded_shape=resolved_padded_shape,
    )

    x = _cpu_tensor_to_nchw_for_glorys(tensor)
    batch, channels, height, width = x.shape
    padding_left, padding_right, padding_top, _padding_bottom = initial_padding
    padded_h, padded_w = padded_shape
    if padded_h % patch_size != 0 or padded_w % patch_size != 0:
        raise RuntimeError(f"padded_shape={padded_shape} must be divisible by patch_size={patch_size}")

    wp_size = wp_topo[0] * wp_topo[1]
    patch_h = padded_h // patch_size
    if patch_h % wp_size != 0:
        raise RuntimeError(f"patched height {patch_h} is not divisible by wp_size={wp_size}")

    local_patch_h = patch_h // wp_size
    local_pixel_start = manager.get_wp_rank() * local_patch_h * patch_size
    local_pixel_end = local_pixel_start + local_patch_h * patch_size

    expected_h = local_patch_h * patch_size
    valid_start = padding_top
    valid_end = padding_top + height
    overlap_start = max(local_pixel_start, valid_start)
    overlap_end = min(local_pixel_end, valid_end)

    local = x.new_zeros((batch, channels, expected_h, width))
    if overlap_end > overlap_start:
        src_start = overlap_start - padding_top
        src_end = overlap_end - padding_top
        dst_start = overlap_start - local_pixel_start
        dst_end = overlap_end - local_pixel_start
        if dst_start < 0 or dst_end > expected_h:
            raise RuntimeError(
                "bad local GLORYS overlap "
                f"wp_rank={manager.get_wp_rank()} local_interval=({local_pixel_start}, {local_pixel_end}) "
                f"valid_interval=({valid_start}, {valid_end}) overlap=({overlap_start}, {overlap_end}) "
                f"dst=({dst_start}, {dst_end}) expected_h={expected_h} padded_shape={padded_shape}"
            )
        local[:, :, dst_start:dst_end, :] = x[:, :, src_start:src_end, :]

    local = torch.nn.functional.pad(
        local,
        (padding_left, padding_right, 0, 0),
    )
    if local.shape[-2] != expected_h or local.shape[-1] != padded_w:
        raise RuntimeError(
            f"bad local GLORYS slice shape={tuple(local.shape)}, expected H/W={(expected_h, padded_w)}, "
            f"wp_rank={manager.get_wp_rank()}, local_interval=({local_pixel_start}, {local_pixel_end}), "
            f"valid_interval=({valid_start}, {valid_end}), padded_shape={padded_shape}"
        )

    local = local.permute(0, 2, 3, 1).contiguous()
    local = patch_fy(local, patch_size).clone().contiguous()
    return _move_local_cpu_tensor_to_device(local, device, my_dtype)


def resolve_required_tensors_online_split_for_parallel(
    task_type,
    x,
    device,
    model_archi_params,
    other_params,
    manager,
    my_dtype,
    is_pretrain=True,
    lead_time=1,
):
    if task_type != 'glorys':
        raise RuntimeError("CPU online split currently supports GLORYS reference workload only")

    if is_pretrain:
        input_tensor, label_tensor = x
        return {
            'input_tensor': _glorys_window_linear_cpu_online_split_one(
                input_tensor, device, my_dtype, model_archi_params, other_params, manager
            ),
            'label_tensor': _glorys_window_linear_cpu_online_split_one(
                label_tensor, device, my_dtype, model_archi_params, other_params, manager
            ),
        }

    sequence_tensor = x
    required_tensors = []
    for cur_lead_time in range(0, lead_time):
        cur_required_tensors = {
            'label_tensor': _glorys_window_linear_cpu_online_split_one(
                sequence_tensor[:, cur_lead_time + 1],
                device,
                my_dtype,
                model_archi_params,
                other_params,
                manager,
            ),
        }
        if cur_lead_time == 0:
            cur_required_tensors['input_tensor'] = _glorys_window_linear_cpu_online_split_one(
                sequence_tensor[:, 0],
                device,
                my_dtype,
                model_archi_params,
                other_params,
                manager,
            )
        required_tensors.append(cur_required_tensors)
    return required_tensors


def manually_split_data_for_parallel_training(required_tensors, task_type, model_archi_params, my_dtype, other_params, manager, data_format, padding_scale=1):
    if task_type != 'glorys':
        raise ValueError(f"Unsupported task_type={task_type!r}; only 'glorys' is supported")

    patch_size = model_archi_params['patch_size']
    window_size = model_archi_params['window_size']
    embedding_parallel_type = other_params['embedding_parallel_type']
    resolved_padded_shape = _get_resolved_padded_shape(model_archi_params, other_params)

    if task_type=='glorys':

        data_format = 'NHWC'
        height = model_archi_params['height']
        width = model_archi_params['width']

        need_padding, initial_padding, padded_shape = get_padded_shape(
            height,
            width,
            patch_size,
            window_size,
            padding_scale=padding_scale,
            padded_shape=resolved_padded_shape,
        )

        for key in required_tensors:
            cur_tensor = required_tensors[key]
            cur_tensor = _cpu_tensor_to_nchw_for_glorys(cur_tensor)

            required_tensors[key] = pad_tensor(cur_tensor, initial_padding, data_format='NCHW').type(my_dtype) # [1, 93, 2112, 4416]
        data_format = 'NCHW'

    if embedding_parallel_type == 'domain_parallel':
        raise ValueError('domain_parallel is not supported in manually_split_data_for_parallel_training')
    elif embedding_parallel_type == 'window_embedding':
        wp_topo = other_params['wp_topo']

        for key in required_tensors:
            cur_tensor = required_tensors[key]

            for k in range(0, wp_topo[0]*wp_topo[1]):
                if k==manager.get_wp_rank():


                    if data_format=='NCHW':
                        num_channel = cur_tensor.shape[1]
                        cur_tensor = cur_tensor.permute(0, 2, 3, 1)

                    required_tensors[key] = get_wp_slice_from_bhwc(cur_tensor, num_channel, patch_size, window_size,
                                                wp_topo = wp_topo, wp_rank = k) # [1, 200, 36, 3348]       [1, 1104, 49, 630])，

    elif embedding_parallel_type == 'window_linear':


        wp_topo = other_params['wp_topo']

        if wp_topo[1]>1:
            print('we do not support 1, m split for window_linear embedding')
            exit(0)


        for key in required_tensors:
            cur_tensor = required_tensors[key]

            for k in range(0, wp_topo[0]*wp_topo[1]):
                if k==manager.get_wp_rank():
                    if data_format=='NCHW':
                        num_channel = cur_tensor.shape[1]

                        cur_tensor = cur_tensor.permute(0, 2, 3, 1) # [1, 1152, 1920, 70]

                    required_tensors[key] = get_patch_fy_slice_from_wp_rank(cur_tensor, patch_size, wp_topo[0]*wp_topo[1], k)


    elif embedding_parallel_type == 'window_domain':
        raise ValueError('window_domain is not supported in manually_split_data_for_parallel_training')
