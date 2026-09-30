"""Brain-wide association of voxel-to-voxel functional connections."""

from .core import BWASResult, run_bwas
from .visualize import plot_bwas_connectivity

__all__ = ["BWASResult", "run_bwas", "plot_bwas_connectivity"]
