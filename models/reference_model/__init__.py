"""Public reference models used to exercise the TERRA runtime.

These classes intentionally provide a CREDIT-derived hierarchical Swin
reference workload rather than the production model implementation.
"""

from .parallel_hierarchical_swin import ParallelHierarchicalSwin, ParallelSwinReference
from .sequential_hierarchical_swin import SequentialHierarchicalSwin, SequentialSwinReference

__all__ = [
    "ParallelHierarchicalSwin",
    "ParallelSwinReference",
    "SequentialHierarchicalSwin",
    "SequentialSwinReference",
]
