"""Global-coordinate ownership, independent of communication and devices."""
from dataclasses import dataclass


def balanced_ranges(length, parts):
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (length, parts)):
        raise ValueError('length and parts must be positive integers')
    return tuple((r * length // parts, (r + 1) * length // parts) for r in range(parts))


def sampling_patch_ranges(patch_height, parts, sampling=True):
    """Bind patch owners to stride-two outputs; clip only the global tail."""
    if not sampling:
        ranges = balanced_ranges(patch_height, parts)
    else:
        coarse = (patch_height + 1) // 2
        ranges = tuple((min(2*a, patch_height), min(2*b, patch_height))
                       for a, b in balanced_ranges(coarse, parts))
    if any(a == b for a, b in ranges):
        raise ValueError('Spatial degree exceeds nonempty model ownership units')
    return ranges


@dataclass(frozen=True)
class SpatialDomain:
    """Unique full-channel H owners for one logical model feature grid."""
    shape: tuple
    ranges: tuple

    def __post_init__(self):
        h, w = self.shape
        if min(h, w) < 1 or not self.ranges or self.ranges[0][0] != 0 or self.ranges[-1][1] != h:
            raise ValueError('Invalid spatial domain coverage')
        for i, (a, b) in enumerate(self.ranges):
            if a >= b or (i and a != self.ranges[i-1][1]):
                raise ValueError('Spatial owners must be nonempty, disjoint and contiguous')

    def local_shape(self, rank):
        a, b = self.ranges[rank]
        return b-a, self.shape[1]

    def scaled(self, factor, width=None):
        return SpatialDomain((self.shape[0]*factor, self.shape[1]*factor if width is None else width),
                             tuple((a*factor, b*factor) for a, b in self.ranges))
