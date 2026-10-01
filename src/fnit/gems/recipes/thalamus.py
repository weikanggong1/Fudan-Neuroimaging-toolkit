"""Bilateral thalamic nuclei GEMS recipe, including two intensity components."""

from __future__ import annotations

import numpy as np
import nibabel as nib
from scipy import ndimage

from .base import (GEMSRecipe, RecipeResult, _native, group_labels,
                   spherical_neighborhood, working_image)


_NUCLEI = (
    "L-Sg", "LGN", "MGN", "PuI", "PuM", "H", "PuL", "VPI", "PuA",
    "MV(Re)", "Pf", "CM", "LP", "VLa", "VPL", "VLp", "MDm", "VM",
    "CeM", "MDl", "Pc", "MDv", "Pv", "CL", "VA", "VPM", "AV",
    "VAmc", "Pt", "AD", "LD",
)
_BRIGHT = {"PuA", "PuI", "PuL", "PuM", "MDl", "MDm"}
_BASE = (
    ("Unknown",), ("Left-Cerebral-White-Matter",),
    ("Left-Cerebral-Cortex",), ("Left-Cerebellum-Cortex",),
    ("Left-Cerebellum-White-Matter",), ("Brain-Stem",),
    ("Left-Lateral-Ventricle",), ("Left-choroid-plexus",),
    ("Left-Putamen",), ("Left-Pallidum",),
    ("Left-Accumbens-area",), ("Left-Caudate",),
)


class ThalamusRecipe(GEMSRecipe):
    resolution_mm = 0.5
    alignment_ids = (10, 49, 28, 60)
    support_ids = (10, 49)
    seg_schedule = ((3.0, 300), (2.0, 150))
    image_schedule = ((1.5, 7), (1.125, 5), (0.75, 5), (0.0, 3))
    fast_image_schedule = ((1.5, 3), (1.125, 3), (0.75, 2), (0.0, 2))

    def segmentation_groups(self, atlas):
        groups = _BASE + (
            tuple(f"{side}-{name}" for name in _NUCLEI + ("R",)
                  for side in ("Left",) if f"{side}-{name}" in atlas.label_names)
            + ("Left-VentralDC",),
            tuple(f"Right-{name}" for name in _NUCLEI + ("R",)
                  if f"Right-{name}" in atlas.label_names) + ("Right-VentralDC",),
        )
        return group_labels(atlas, groups)

    def intensity_groups(self, atlas, stage):
        base = list(_BASE)
        base[1] = base[1] + tuple(name for name in ("Left-R", "Right-R")
                                   if name in atlas.label_names)
        base.append(("Left-VentralDC", "Right-VentralDC"))
        if stage == 0:
            base.append(tuple(f"{side}-{name}" for side in ("Left", "Right")
                              for name in _NUCLEI if f"{side}-{name}" in atlas.label_names))
        else:
            base.append(tuple(f"{side}-{name}" for side in ("Left", "Right")
                              for name in _NUCLEI if name not in _BRIGHT
                              and f"{side}-{name}" in atlas.label_names))
            base.append(tuple(f"{side}-{name}" for side in ("Left", "Right")
                              for name in _NUCLEI if name in _BRIGHT
                              and f"{side}-{name}" in atlas.label_names))
        return group_labels(atlas, tuple(base))

    def _recoded_segmentation(self, coarse):
        source = np.asarray(coarse)
        data = source.copy()
        recode = {5: 4, 44: 4, 14: 4, 15: 4, 17: 3, 53: 3, 18: 3, 54: 3,
                  24: 4, 30: 2, 62: 2, 72: 4, 77: 2, 80: 0, 85: 0,
                  41: 2, 42: 3, 43: 4, 46: 7, 47: 8, 50: 11, 51: 12,
                  52: 13, 58: 26, 63: 31}
        for old, new in recode.items():
            data[source == old] = new
        data[source > 250] = 2
        data[data == 0] = 1
        return data

    def synthetic_labels(self, coarse):
        data = self._recoded_segmentation(coarse)
        data[data == 28] = 10
        data[data == 60] = 49
        return data

    def prepare_working_image(self, context):
        image, labels, report = working_image(context, self.alignment_ids, self.resolution_mm)
        mask = ndimage.binary_dilation(self.synthetic_labels(context.coarse_segmentation) > 1,
                                       structure=np.ones((3, 3, 3)), iterations=2)
        sampled = _native(mask.astype(np.uint8), context.image.affine, image, 0).astype(bool)
        data = np.asarray(image.dataobj).copy()
        data[~sampled] = 0
        return nib.Nifti1Image(data, image.affine), labels, report

    def synthetic_means(self, atlas, classes):
        means = np.zeros(int(classes.max()) + 1, np.float32)
        for group in range(len(means)):
            ids = atlas.label_ids[classes == group]
            first = int(ids[0])
            means[group] = (10 if np.any((ids >= 8100) & (ids < 8200)) else
                            49 if np.any((ids >= 8200) & (ids < 8300)) else
                            1 if first == 0 else first)
        return means

    def gaussian_hyperparameters(self, context, atlas, classes):
        means = np.zeros(int(classes.max()) + 1, np.float32)
        counts = np.zeros_like(means)
        voxel_volume = abs(np.linalg.det(context.image.affine[:3, :3]))
        coarse = self._recoded_segmentation(context.coarse_segmentation)
        radius = max(1, int(np.rint(1 / np.linalg.norm(context.image.affine[:3, :3], axis=0).mean())))
        neighborhood = spherical_neighborhood(radius)
        for group in range(len(means)):
            ids = atlas.label_ids[classes == group]
            if np.any(ids > 8225):
                target_ids = (10, 49)
            elif np.any(ids == 28):
                target_ids = (28, 60)
            elif np.any(ids == 0):
                target_ids = (1,)
            else:
                target_ids = tuple(int(v) for v in ids)
            mask = ndimage.binary_erosion(np.isin(coarse, target_ids),
                                          structure=neighborhood, border_value=1)
            samples = context.data[mask & (context.data > 0)]
            means[group] = float(np.median(samples)) if samples.size else 55
            counts[group] = (10 if np.any(ids == 28) else
                             10 + samples.size * voxel_volume / self.resolution_mm**3)
        if len(means) == 15:
            thalamus_mean = means[13]
            means[13:] = (thalamus_mean + 5, thalamus_mean - 5)
            counts[13:] = 25
        return means, counts

    def foreground_ids(self, atlas):
        return {int(value) for value in atlas.label_ids
                if 8100 <= int(value) < 8300 and int(value) not in (8125, 8225)}

    def postprocess(self, fit, context, atlas):
        raw = fit.labels.detach().cpu().numpy().astype(np.int32)
        foreground = np.isin(raw, list(self.foreground_ids(atlas)))
        keep = np.zeros(raw.shape, bool)
        for low, high in ((8100, 8200), (8200, 8300)):
            connected, count = ndimage.label(foreground & (raw >= low) & (raw < high))
            if count:
                sizes = np.bincount(connected.ravel())
                sizes[0] = 0
                keep |= connected == sizes.argmax()
        labels = np.where(keep, raw, 0).astype(np.int32)
        native = _native(labels, fit.affine, context.image, 0).astype(np.int32)
        confidence = fit.posterior.max(0).values.detach().cpu().numpy().astype(np.float32)
        native_conf = _native(confidence, fit.affine, context.image, 1)
        support = ndimage.binary_dilation(np.isin(context.coarse_segmentation,
                                                 self.support_ids), iterations=5)
        native[~support] = 0
        voxel_volume = abs(np.linalg.det(fit.affine[:3, :3]))
        posterior = fit.posterior.detach().cpu()
        volumes = {int(label): float(posterior[index].sum() * voxel_volume)
                   for index, label in enumerate(atlas.label_ids)
                   if int(label) in self.foreground_ids(atlas)}
        return RecipeResult(fit, nib.Nifti1Image(labels, fit.affine),
                            native, native_conf, support, volumes, {})
