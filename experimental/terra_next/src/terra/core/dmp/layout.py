"""Owned full-channel spatial stripes <-> convolution compute partitions."""
import torch
from terra.runtime.tensor_layout import contiguous_for_device, has_channels_last_layout
from .communication import exchange_pieces


def _check(tensor, mesh):
    mesh.check_open()
    if tensor.ndim != 4 or not tensor.is_floating_point() or min(tensor.shape) < 1:
        raise ValueError('Layout operations require nonempty floating NCHW tensors')


def _restore(tensor, source, mesh):
    channels_last = has_channels_last_layout(source)
    mesh.stats['logical_memory_formats'].add('channels_last' if channels_last else 'contiguous')
    return contiguous_for_device(tensor, channels_last=channels_last)


def _team_ranges(ranges, mesh):
    first = mesh.spatial_rank * mesh.tp_size
    return ranges[first:first+mesh.tp_size]


def spatial_to_compute(tensor, mesh, row_ranges=None):
    """Gather only each Ci slice; Co replicas receive the same input channels."""
    _check(tensor, mesh)
    batch, channels, height, width = tensor.shape
    if channels % mesh.ci_parts:
        raise ValueError('Input channels must divide Ci')
    if row_ranges is None:
        row_ranges = tuple((r*height, (r+1)*height) for r in range(mesh.size))
    if len(row_ranges) != mesh.size or row_ranges[mesh.rank][1]-row_ranges[mesh.rank][0] != height:
        raise ValueError('Input does not match its explicit spatial ownership')
    mesh.peak('max_boundary_input_numel', tensor.numel())
    if mesh.tp_size == 1:
        output = tensor
    else:
        chunk = channels // mesh.ci_parts
        pieces = [tensor[:, (dest % mesh.ci_parts)*chunk:(dest % mesh.ci_parts+1)*chunk]
                  for dest in range(mesh.tp_size)]
        team = _team_ranges(row_ranges, mesh)
        counts = [batch*chunk*(b-a)*width for a,b in team]
        received = exchange_pieces(pieces, counts, mesh)
        parts = [part.reshape(batch, chunk, b-a, width)
                 for part,(a,b) in zip(received.split(counts), team)]
        output = _restore(torch.cat(parts, dim=2), tensor, mesh)
    mesh.peak('max_compute_input_numel', output.numel())
    mesh.record('spatial_to_compute', input_shape=list(tensor.shape), output_shape=list(output.shape),
                input_ownership_fraction=height/row_ranges[-1][1],
                compute_input_ratio=output.numel()/tensor.numel(),
                packed_numel=tensor.numel()*mesh.co_parts,
                logical_dtype=str(tensor.dtype).removeprefix('torch.'))
    return output


def compute_to_spatial(partial, mesh, row_ranges=None):
    """Route each destination's true H interval, SUM Ci and concatenate Co."""
    _check(partial, mesh)
    batch, out_chunk, macro_height, width = partial.shape
    if row_ranges is None:
        if macro_height % mesh.tp_size:
            raise ValueError('Provide explicit output owners for a ragged macro strip')
        height = macro_height // mesh.tp_size
        row_ranges = tuple((r*height, (r+1)*height) for r in range(mesh.size))
    team = _team_ranges(row_ranges, mesh)
    if sum(b-a for a,b in team) != macro_height:
        raise ValueError('Convolution output does not cover its owned macro strip')
    mesh.peak('max_partial_output_numel', partial.numel())
    if mesh.tp_size == 1:
        output = partial
    else:
        offset = 0
        pieces = []
        for a,b in team:
            pieces.append(partial[:, :, offset:offset+b-a, :])
            offset += b-a
        height = row_ranges[mesh.rank][1]-row_ranges[mesh.rank][0]
        counts = [batch*out_chunk*height*width]*mesh.tp_size
        received = exchange_pieces(pieces, counts, mesh)
        grouped = received.reshape(mesh.co_parts, mesh.ci_parts, batch, out_chunk, height, width)
        accumulation = torch.float32 if partial.dtype in (torch.float16, torch.bfloat16) else partial.dtype
        reduced = grouped.to(accumulation).sum(1).to(partial.dtype) if mesh.ci_parts > 1 else grouped[:,0]
        output = _restore(torch.cat(tuple(reduced.unbind(0)), dim=1), partial, mesh)
    mesh.peak('max_boundary_output_numel', output.numel())
    mesh.record('compute_to_spatial', input_shape=list(partial.shape), output_shape=list(output.shape),
                output_ownership_fraction=output.shape[2]/row_ranges[-1][1],
                packed_numel=partial.numel(), logical_dtype=str(partial.dtype).removeprefix('torch.'),
                reduction_dtype='float32' if partial.dtype in (torch.float16,torch.bfloat16) else str(partial.dtype).removeprefix('torch.'))
    return output
