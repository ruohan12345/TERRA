import torch

from dataloader.glorys_utils import get_land_sea_mask, get_wp_slice_from_bhwc

from utils import get_padded_shape

from core.global_env_config import ON_H200
from dataloader.dataloader_utils import pad_tensor, get_patch_fy_slice_from_wp_rank, _get_resolved_padded_shape


def get_task_specific_data(task_type, device, optimizer_state_tuple, model_archi_params, other_params, manager, micro_batch_size, model_type='sequential', dataset_config = None):
    task_specific_data_dict = {}
    #USE_DDP, USE_FSDP, ZERO_STAGE_NUMBER = optimizer_state_tuple

    if task_type=='glorys': # land_sea_mask padding_loss_scale

        land_sea_mask = get_land_sea_mask(

        ).contiguous() # Keep the full mask on CPU. Only the local slice is moved to GPU below.


        height = model_archi_params['height']
        width = model_archi_params['width']
        patch_size = model_archi_params['patch_size']
        window_size = model_archi_params['window_size']
        padding_scale = model_archi_params['padding_scale']
        model_architecture = model_archi_params.get('model_architecture', None)
        resolved_padded_shape = _get_resolved_padded_shape(model_archi_params, other_params)

        need_padding, initial_padding, padded_shape = get_padded_shape(
            height,
            width,
            patch_size,
            window_size,
            padding_scale=padding_scale,
            padded_shape=resolved_padded_shape,
        ) # (2112, 4320)

        if need_padding and model_type=='parallel':
            with torch.no_grad():
                initial_pad = torch.nn.ZeroPad2d(initial_padding)
                land_sea_mask = initial_pad(land_sea_mask).contiguous() #

        if ON_H200 and model_type=='sequential' and model_architecture != 'swin_reference':
            pass
        else:
            land_sea_mask = land_sea_mask.permute(1, 2, 0).contiguous() #[2112, 4416, 93]


        embedding_parallel_type = other_params['embedding_parallel_type']
        wp_topo = other_params['wp_topo']
        if model_type=='parallel':
            if embedding_parallel_type == 'window_linear':
                land_sea_mask = get_patch_fy_slice_from_wp_rank(land_sea_mask[None], model_archi_params['patch_size'], wp_topo[0]*wp_topo[1], manager.get_wp_rank()) # [1, 132, 1104, 1488]
                land_sea_mask = land_sea_mask[0]
            elif embedding_parallel_type == 'window_embedding':
                num_channel = land_sea_mask.shape[-1]
                for k in range(0, wp_topo[0]*wp_topo[1]):
                    if k==manager.get_wp_rank():
                        land_sea_mask = get_wp_slice_from_bhwc(land_sea_mask[None], num_channel, patch_size, window_size, wp_topo = wp_topo, wp_rank = k)[0] # [1012, 36, 5952]


        if need_padding and model_type == 'parallel':

            padding_loss_scale = (padded_shape[0]*padded_shape[1])/(model_archi_params['height']*model_archi_params['width']) # 0.9663825757575758
        else:
            padding_loss_scale = 1.0

        land_sea_mask = land_sea_mask.contiguous().to(device, non_blocking=True)

        task_specific_data_dict['padding_loss_scale'] = padding_loss_scale
        task_specific_data_dict['land_sea_mask']= land_sea_mask
        task_specific_data_dict['model_type'] = model_type
        task_specific_data_dict['model_architecture'] = model_architecture
        task_specific_data_dict['height'] = height
        task_specific_data_dict['width'] = width


    else:
        raise ValueError(f"Unsupported task_type: {task_type}; only GLORYS reference workload is available")

    return task_specific_data_dict
