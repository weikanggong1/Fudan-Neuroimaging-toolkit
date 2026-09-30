"""Compatibility-preserving wrapper for the validated BrainstemSS recipe."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from .base import RecipeResult
from ..atlas import GEMSAtlas


class BrainstemRecipe:
    name = "brainstem"

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def run(self, context, device):
        from ..pipeline import _segment_atlas_packs

        result = _segment_atlas_packs(
            context.image, self.directory.parent, structures="brainstem",
            coarse_segmentation=context.coarse_segmentation,
            auto_initialize=True, device=device)
        fit = result.structure_results["brainstem"]
        labels = fit.labels.detach().cpu().numpy().astype(np.int32)
        native = np.asarray(result.labels.dataobj, dtype=np.int32)
        confidence = result.confidence.detach().cpu().numpy().astype(np.float32)
        atlas = GEMSAtlas.from_freesurfer(
            self.directory / "AtlasMesh.gz", self.directory / "compressionLookupTable.txt")
        voxel_volume = abs(np.linalg.det(fit.affine[:3, :3]))
        posterior = fit.posterior.detach().cpu()
        volumes = {int(label): float(posterior[index].sum() * voxel_volume)
                   for index, label in enumerate(atlas.label_ids)
                   if int(label) in (173, 174, 175, 178)}
        highres = np.where(np.isin(labels, (173, 174, 175, 178)), labels, 0)
        return RecipeResult(fit, nib.Nifti1Image(highres, fit.affine), native,
                            confidence, native != 0, volumes,
                            result.initialization["brainstem"])
