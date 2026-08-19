import torch
from einops import rearrange
import numpy as np


from core.dist_matmul import XWT_apply

from core.global_env_config import EMB_1_input_split, EMB_1_output_split, EMB_2_input_split, EMB_2_output_split, EMB_1_SPLIT_PARAM, EMB_2_SPLIT_PARAM

USE_BIAS = True


# [1, 93 ,720, 1440] -> 【b , 28,800, 3,348】*[3348, 768]

# [M, K] @ [K, N]

class SeqPatchEmbedding(torch.nn.Module):
    def __init__(self,
                kaiming_init = True,
                patch_size  = -1,
                num_channel = -1,
                embedding_dim = -1,
                ):
        super().__init__()

        self.patch_size = patch_size

        linear_in_dim = num_channel*patch_size*patch_size
        linear_out_dim = embedding_dim


        self.linear = torch.nn.Linear(linear_in_dim, linear_out_dim, bias = USE_BIAS)

    def forward(self, x):


        x = rearrange(x, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)',
                        p1=self.patch_size, p2=self.patch_size) # [1, 28800, 2592]


        x = self.linear(x)


        if False:
            y = x.detach().clone()


            if False:
                y_list = y.split(y.shape[-2]//2, dim=-2)

                y0 = y_list[0]
                y1 = y_list[1]
                y0_list = y0.split(y0.shape[-1]//2, dim=-1)
                y1_list = y1.split(y1.shape[-1]//2, dim=-1)
                print('seq y00', y0_list[0].sum(), y0_list[0].shape)
                print('seq y01', y0_list[1].sum(), y0_list[1].shape)

                print('seq y10', y1_list[0].sum(), y1_list[0].shape)
                print('seq y11', y1_list[1].sum(), y1_list[1].shape)
            else:
                y_list = y.split(y.shape[-2]//4, dim=-2)

                print('seq y0', y_list[0].sum(), y_list[0].shape)
                print('seq y1', y_list[1].sum(), y_list[1].shape)
                print('seq y2', y_list[2].sum(), y_list[2].shape)
                print('seq y3', y_list[3].sum(), y_list[3].shape)


        return x

class SeqPatchRecovery(torch.nn.Module):
    def __init__(self,
                kaiming_init = True,
                height = 720,
                width = 1440,
                patch_size  = -1,
                num_channel = -1,
                embedding_dim = -1,
                ):
        super().__init__()

        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.num_channel = num_channel

        linear_in_dim = embedding_dim
        linear_out_dim = num_channel*patch_size*patch_size

        self.linear = torch.nn.Linear(linear_in_dim, linear_out_dim, bias = USE_BIAS)

    def forward(self, x): # [1, 28800, 4320]
        x = self.linear(x)

        x = rearrange(x, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c',
              p1=self.patch_size, p2=self.patch_size,
              h=self.height // self.patch_size,
              w=self.width // self.patch_size).contiguous()

        return x


def split_embedding_weight_and_bias(
    kaiming_init,
    mp_rank,
    mp_group_size,
    linear_in_dim,
    linear_out_dim,

    input_split,
    output_split,
    split_param,
    ):

    if kaiming_init:
        linear_temp = torch.nn.Linear(
                linear_in_dim,
                linear_out_dim,
                bias = USE_BIAS,
            )
    else:
        print('no kaiming init is not supported here')
        exit(0)

    if input_split == '(1,n)':
        if output_split == '(m,1)':


            weight_list = linear_temp.weight.data.split(linear_temp.weight.data.shape[-1]//mp_group_size, dim=-1)
            tmp_weight = weight_list[mp_rank].contiguous() # [384, 3348]
            weight = torch.nn.Parameter(torch.empty_like(tmp_weight))
            with torch.no_grad():
                weight.copy_(tmp_weight)
            if USE_BIAS:
                tmp_bias = linear_temp.bias.data
                bias = torch.nn.Parameter(torch.empty_like(tmp_bias))
                with torch.no_grad():
                    bias.copy_(tmp_bias)
            else:
                bias = None
        elif output_split == '(1,n)':


            weight_list = linear_temp.weight.data.split(linear_temp.weight.data.shape[-1]//mp_group_size, dim=-1)
            tmp_weight = weight_list[mp_rank].contiguous() # [768, 1674]
            weight = torch.nn.Parameter(torch.empty_like(tmp_weight))
            with torch.no_grad():
                weight.copy_(tmp_weight)
            if USE_BIAS:
                bias_list = linear_temp.bias.data.split(linear_temp.bias.data.shape[-1]//mp_group_size, dim=-1)
                tmp_bias = bias_list[mp_rank].contiguous() # [384]
                bias = torch.nn.Parameter(torch.empty_like(tmp_bias))
                with torch.no_grad():
                    bias.copy_(tmp_bias)
            else:
                bias = None
        else:
            print('unrecognized output_split', output_split)
            exit(0)
    elif input_split == '(m,1)':
        if output_split == '(m,1)':

            weight_list = linear_temp.weight.data.split(linear_temp.weight.data.shape[-2]//mp_group_size, dim=-2)
            tmp_weight = weight_list[mp_rank].contiguous() # [384, 3348]

            weight = torch.nn.Parameter(torch.empty_like(tmp_weight))
            with torch.no_grad():
                weight.copy_(tmp_weight)

            if USE_BIAS:
                tmp_bias = linear_temp.bias.data
                bias = torch.nn.Parameter(torch.empty_like(tmp_bias))
                with torch.no_grad():
                    bias.copy_(tmp_bias)
            else:
                bias = None
        elif output_split == '(1,n)':


            weight_list = linear_temp.weight.data.split(linear_temp.weight.data.shape[-2]//mp_group_size, dim=-2)
            tmp_weight = weight_list[mp_rank].contiguous() # [1674, 768]

            weight = torch.nn.Parameter(torch.empty_like(tmp_weight))  #[384, 1674] for (2, 2),split_param, [192, 3348] for (4, 1),split_param
            with torch.no_grad():
                weight.copy_(tmp_weight)

            if USE_BIAS:
                bias_list = linear_temp.bias.data.split(linear_temp.bias.data.shape[-1]//mp_group_size, dim=-1)
                tmp_bias = bias_list[mp_rank].contiguous() # [1674]

                bias = torch.nn.Parameter(torch.empty_like(tmp_bias))
                with torch.no_grad():
                    bias.copy_(tmp_bias)
            else:
                bias = None
        else:
            print('unrecognized output_split', output_split)
            exit(0)
    else:
        print('unrecognized input_split', input_split)
        exit(0)


    return weight, bias

class Domain_ParaPatchEmbedding(torch.nn.Module):
    def __init__(self,
                kaiming_init = True,
                manager=None,

                device = None,
                patch_size  = -1,
                num_channel = -1,
                embedding_dim = -1,
                ):
        super().__init__()
        self.manager = manager
        self.device = device

        self.linear_in_dim = num_channel*patch_size*patch_size
        self.linear_out_dim = embedding_dim

        self.input_split  = EMB_1_input_split  #  '(1,n)'
        self.output_split = EMB_1_output_split #  '(m,1)'
        self.split_param = EMB_1_SPLIT_PARAM

        if self.split_param:
            self.weight, self.bias = split_embedding_weight_and_bias(
                kaiming_init = kaiming_init,
                mp_rank = manager.mp_rank,
                mp_group_size = manager.get_mp_group_size(),
                linear_in_dim = self.linear_in_dim,
                linear_out_dim = self.linear_out_dim,

                input_split = self.input_split,
                output_split = self.output_split,
                split_param = self.split_param,
            )
        else:
            if self.input_split == '(m,1)' and self.output_split=='(m,1)':
                self.linear = torch.nn.Linear(self.linear_in_dim, self.linear_out_dim, bias = USE_BIAS)
            else:
                print('we do not support not split_param except for (m,1) and (m,1)')
                exit(0)

    def forward(self, x): # [2, 28800, 1674]
        if self.split_param:
            x = XWT_apply(x,
                self.weight,
                self.bias,
                self.manager,
                self.device,

                input_split = self.input_split,
                output_split = self.output_split,
                split_param = self.split_param,
                use_bias = USE_BIAS,
                )
        else:
            x = self.linear(x)

        return x

class Domain_ParaPatchRecovery(torch.nn.Module):
    def __init__(self,
                kaiming_init = True,
                manager = None,

                device = None,
                height = 720,
                width = 1440,
                patch_size  = -1,
                num_channel = -1,
                embedding_dim = -1,
                ):
        super().__init__()
        self.manager = manager
        self.device = device

        self.domain_topo = manager.domain_topo # (1, 2)

        self.linear_in_dim = embedding_dim
        self.linear_out_dim = num_channel*patch_size*patch_size

        self.input_split  = EMB_2_input_split  #  (m,1)
        self.output_split = EMB_2_output_split #  (1,n)
        self.split_param = EMB_2_SPLIT_PARAM

        if self.split_param:
            self.weight, self.bias = split_embedding_weight_and_bias(
                kaiming_init = kaiming_init,
                mp_rank = manager.mp_rank,
                mp_group_size = manager.get_mp_group_size(),
                linear_in_dim = self.linear_in_dim,
                linear_out_dim = self.linear_out_dim,

                input_split = self.input_split,
                output_split = self.output_split,
                split_param = self.split_param,
            )
        else:
            if self.input_split == '(m,1)' and self.output_split=='(m,1)':
                self.linear = torch.nn.Linear(self.linear_in_dim, self.linear_out_dim, bias = USE_BIAS)
            else:
                print('we do not support not split_param except for (m,1) and (m,1)')
                exit(0)

    def forward(self, x):
        if self.split_param:
            x = XWT_apply(x,
                self.weight,
                self.bias,
                self.manager,

                self.device,

                input_split = self.input_split,
                output_split = self.output_split,
                split_param = self.split_param,
                use_bias = USE_BIAS,
                )
        else:
            x = self.linear(x)

        return x

class Window_ParaPatchEmbedding(torch.nn.Module):
    def __init__(self,
                kaiming_init = True,
                manager = None,
                device = None,
                patch_size  = -1,
                num_channel = -1,
                embedding_dim = -1,
                ):
        super().__init__()
        self.manager = manager

        self.device = device
        self.embedding_dim = embedding_dim

        linear_in_dim = (num_channel*patch_size*patch_size) # 3348
        linear_out_dim = embedding_dim # 768

        self.linear = torch.nn.Linear(linear_in_dim, linear_out_dim)

    def forward(self, x):# [2, 200, 6*6, 3348]


        x = self.linear(x)


        return x

class Window_ParaPatchRecovery(torch.nn.Module):
    def __init__(self,
                kaiming_init = True,
                manager = None,
                device = None,
                patch_size  = -1,
                num_channel = -1,
                embedding_dim = -1,
                ):
        super().__init__()
        self.manager = manager


        linear_in_dim = embedding_dim
        linear_out_dim = num_channel*patch_size*patch_size

        self.linear = torch.nn.Linear(linear_in_dim, linear_out_dim)

    def forward(self, x): # [2, 200, 36, 768]

        x = self.linear(x) # [2, 200, 36, 3348]

        return x


class WrappedLinear(torch.nn.Linear):
    """Linear wrapper whose parameters may be marked for all-reduce."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
