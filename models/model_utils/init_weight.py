import torch


def _get_rng_state():
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def _restore_rng_state(state):
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def make_linear_with_seed(linear_in_dim, linear_out_dim, bias=True, init_seed=None):
    if init_seed is None:
        return torch.nn.Linear(linear_in_dim, linear_out_dim, bias=bias)

    rng_state = _get_rng_state()
    torch.manual_seed(int(init_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(init_seed))
    linear = torch.nn.Linear(linear_in_dim, linear_out_dim, bias=bias)
    _restore_rng_state(rng_state)
    return linear


def run_with_seed(init_seed, fn):
    if init_seed is None:
        return fn()

    rng_state = _get_rng_state()
    torch.manual_seed(int(init_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(init_seed))
    out = fn()
    _restore_rng_state(rng_state)
    return out


def init_like_linear(
            model_type, # sequential or parallel
            linear_in_dim,
            linear_out_dim,
            mp_rank=0,
            mp_group_size=1,
            split_dim=None, # only for parallel
            split_bias = True,
            use_bias = True,
            init_seed = None,
    ):


    linear_temp = make_linear_with_seed(
        linear_in_dim,
        linear_out_dim,
        bias=True,
        init_seed=init_seed,
    )

    if False:
        print('mp_rank', mp_rank, 'linear_temp sum', linear_temp.weight.data.sum(), "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    if model_type=='sequential':

        weights = torch.nn.Parameter(torch.zeros(linear_out_dim, linear_in_dim))
        bias = torch.nn.Parameter(torch.zeros(linear_out_dim))


        with torch.no_grad():
            weights.data = linear_temp.weight.data
            bias.data = linear_temp.bias.data
    elif model_type=='parallel':


        if split_dim==-1:


            weights = torch.nn.Parameter(torch.zeros(linear_out_dim, linear_in_dim//mp_group_size))
            tmp_weight = torch.split(linear_temp.weight.data, linear_in_dim//mp_group_size, dim = split_dim)[mp_rank].contiguous()
            if split_bias:
                bias = torch.nn.Parameter(torch.zeros(linear_out_dim//mp_group_size))
                tmp_bias = torch.split(linear_temp.bias.data, linear_out_dim//mp_group_size, dim = 0)[mp_rank].contiguous() # dim is not important for bias
            else:
                bias = torch.nn.Parameter(torch.zeros(linear_out_dim))
                tmp_bias = linear_temp.bias.data
        elif split_dim==0:
            weights = torch.nn.Parameter(torch.zeros(linear_out_dim//mp_group_size, linear_in_dim))
            tmp_weight = torch.split(linear_temp.weight.data, linear_out_dim//mp_group_size, dim = split_dim)[mp_rank]
            if split_bias:
                bias = torch.nn.Parameter(torch.zeros(linear_out_dim//mp_group_size))
                tmp_bias = torch.split(linear_temp.bias.data, linear_out_dim//mp_group_size, dim = 0)[mp_rank].contiguous()
            else:
                print('we do not support split_dim==0 and not split bias')
                exit(0)
        else:
            print('unrecognized split_dim', split_dim)
            exit(0)
    else:
        print('unrecognized model_type', model_type)
        exit(0)

    with torch.no_grad():
        weights.copy_(tmp_weight)

    with torch.no_grad():
        bias.copy_(tmp_bias)

    return weights, bias


def safe_linear_with_weight(weight, bias=None):
    """Create a linear layer with fixed weights without advancing global RNG state."""

    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None


    linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=(bias is not None))


    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


    with torch.no_grad():
        linear.weight.copy_(weight)
        if bias is not None:
            linear.bias.copy_(bias)

    return linear
