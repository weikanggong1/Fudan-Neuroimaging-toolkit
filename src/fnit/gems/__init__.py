"""FreeSurfer-independent PyTorch GEMS Bayesian atlas segmentation."""

from .atlas import GEMSAtlas, read_compression_lut
from .core import TorchGEMS, TorchGEMSResult
from .deformation import ashburner_prior
from .initialize import estimate_label_centroid_affine
from .rasterize import BlockIndex, build_block_index, rasterize_priors


def __getattr__(name):
    if name in {"SubregionResult", "segment_subregions"}:
        from . import pipeline
        return getattr(pipeline, name)
    if name == "prepare_brainstem_atlas":
        from .setup import prepare_brainstem_atlas
        return prepare_brainstem_atlas
    if name in {"prepare_nuclei_atlas", "segment_nuclei"}:
        from . import nuclei
        return getattr(nuclei, name)
    raise AttributeError(name)


__all__ = [
    "GEMSAtlas", "read_compression_lut", "TorchGEMS", "TorchGEMSResult",
    "ashburner_prior", "estimate_label_centroid_affine", "SubregionResult", "segment_subregions",
    "BlockIndex", "build_block_index", "rasterize_priors",
    "prepare_brainstem_atlas",
    "prepare_nuclei_atlas", "segment_nuclei",
]
