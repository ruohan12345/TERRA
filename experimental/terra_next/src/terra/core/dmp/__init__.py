"""Native H-sharded DMP with per-operator channel compute layouts."""
from .mesh import DMPMesh
from .layout import spatial_to_compute, compute_to_spatial
from .communication import halo_exchange

__all__ = ["DMPMesh", "spatial_to_compute", "compute_to_spatial", "halo_exchange"]
