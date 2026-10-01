"""Hemisphere-specific hippocampal subfield and amygdala GEMS recipe."""

from __future__ import annotations

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.stats import gaussian_kde
import torch

from .base import (GEMSRecipe, RecipeResult, _native, group_labels,
                   spherical_neighborhood, working_image)
from ..rasterize import rasterize_priors


_GM = tuple(name for name in (
    "Left-Cerebral-Cortex", "Hippocampal_tail", "subiculum-body", "CA1-body",
    "subiculum-head", "presubiculum-head", "CA1-head", "presubiculum-body",
    "parasubiculum", "molecular_layer_HP-head", "molecular_layer_HP-body",
    "GC-ML-DG-head", "CA3-body", "GC-ML-DG-body", "CA4-head", "CA4-body",
    "CA3-head", "HATA", "Lateral-nucleus", "Basal-nucleus",
    "Accessory-Basal-nucleus", "Anterior-amygdaloid-area-AAA",
    "Central-nucleus", "Medial-nucleus", "Cortical-nucleus",
    "Corticoamygdaloid-transitio", "Paralaminar-nucleus",
))


class HippoAmygdalaRecipe(GEMSRecipe):
    # Keep the fine hippocampal/amygdala data integral at all valid voxels.
    # Sparse quadrature changed right-side boundaries on real stage inputs.
    fast_mesh_sampling_stride = 1

    resolution_mm = 0.33333
    seg_schedule = ((3.0, 300), (2.0, 150))
    image_schedule = ((1.5, 7), (0.75, 5), (0.0, 3))

    def __init__(self, side: str, directory):
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        self.side = side
        self.alignment_ids = (17, 18) if side == "left" else (53, 54)
        self.crop_ids = (17,) if side == "left" else (53,)
        self.support_ids = self.alignment_ids
        self.high_res_input = False
        super().__init__(f"hippo-amygdala-{side}", directory)

    def run(self, context, device):
        self._preparation_device = torch.device(device)
        self.high_res_input = np.linalg.norm(context.image.affine[:3, :3], axis=0).mean() < 0.99
        return super().run(context, device)

    def alignment_image(self):
        image = super().alignment_image()
        if self.side == "left":
            return image
        affine = image.affine.copy()
        affine[0] *= -1
        return nib.Nifti1Image(np.asarray(image.dataobj), affine)

    def prepare_working_image(self, context):
        image, labels, report = working_image(context, self.crop_ids, self.resolution_mm)
        roi = np.isin(context.coarse_segmentation, self.alignment_ids)
        merged = ndimage.binary_dilation(self.synthetic_labels(context.coarse_segmentation) > 1,
                                         structure=np.ones((3, 3, 3)), iterations=2)
        sampled = _native(roi.astype(np.float32), context.image.affine, image, 1) >= 0.5
        sampled = ndimage.binary_dilation(sampled, structure=np.ones((3, 3, 3)),
                                          iterations=int(round(3 / self.resolution_mm)))
        sampled &= _native(merged.astype(np.uint8), context.image.affine, image, 0).astype(bool)
        data = np.asarray(image.dataobj).copy()
        data[~sampled] = 0
        return nib.Nifti1Image(data, image.affine), labels, report

    def segmentation_groups(self, atlas):
        groups = (
            _GM + ("alveus", "fimbria"), ("Left-Cerebral-White-Matter",),
            ("Left-Lateral-Ventricle",), ("Left-choroid-plexus",),
            ("Unknown", "hippocampal-fissure"), ("Left-VentralDC",),
            ("Left-Putamen",), ("Left-Pallidum",),
            ("Left-Thalamus-Proper",), ("Left-Accumbens-area",),
            ("Left-Caudate",),
        )
        return group_labels(atlas, groups)

    def intensity_groups(self, atlas, stage):
        molecular = ("molecular_layer_HP-head", "molecular_layer_HP-body")
        groups = (
            tuple(name for name in _GM if not self.high_res_input or name not in molecular),
            *((molecular,) if self.high_res_input else ()),
            ("Left-Cerebral-White-Matter", "fimbria"), ("alveus",),
            ("Left-Lateral-Ventricle",), ("hippocampal-fissure",),
            ("Left-Pallidum",), ("Left-Putamen",), ("Left-Caudate",),
            ("Left-Thalamus-Proper",), ("Left-choroid-plexus",),
            ("Left-VentralDC",), ("Left-Accumbens-area",), ("Unknown",),
        )
        return group_labels(atlas, groups)

    def synthetic_labels(self, coarse):
        source = np.asarray(coarse)
        data = np.ones(source.shape, dtype=np.int32)
        mappings = ({2: 2, 3: 3, 4: 4, 10: 10, 11: 11, 12: 12, 13: 13,
                     17: 3, 18: 3, 26: 26, 28: 28, 31: 31}
                    if self.side == "left" else
                    {41: 2, 42: 3, 43: 4, 49: 10, 50: 11, 51: 12,
                     52: 13, 53: 3, 54: 3, 58: 26, 60: 28, 63: 31})
        for old, new in mappings.items():
            data[source == old] = new
        for old, new in ((14, 4), (24, 4), (77, 2), (72, 4)):
            data[source == old] = new
        data[source == (30 if self.side == "left" else 62)] = 2
        data[source == (5 if self.side == "left" else 44)] = 4
        data[source > 250] = 2
        return data

    def synthetic_means(self, atlas, classes):
        values = (3, 2, 4, 31, 1, 28, 12, 13, 10, 26, 11)
        return np.asarray(values, dtype=np.float32)

    def gaussian_hyperparameters(self, context, atlas, classes):
        means = np.full(int(classes.max()) + 1, 55.0, np.float32)
        counts = np.full_like(means, 10.0)
        coarse = context.wmparc_proxy
        side = self.side
        side_map = {4: 4, 13: 13, 12: 12, 11: 11, 10: 10, 31: 31, 28: 28, 26: 26}
        if side == "right":
            side_map = {4: 43, 13: 52, 12: 51, 11: 50,
                        10: 49, 31: 63, 28: 60, 26: 58}
        voxel_volume = abs(np.linalg.det(context.image.affine[:3, :3]))
        radius = max(1, int(np.rint(1 / np.linalg.norm(context.image.affine[:3, :3], axis=0).mean())))
        neighborhood = spherical_neighborhood(radius)
        local_background = ndimage.binary_dilation(
            np.isin(context.coarse_segmentation, self.alignment_ids),
            structure=np.ones((3, 3, 3)), iterations=5)
        for group in range(len(means)):
            ids = set(int(v) for v in atlas.label_ids[classes == group])
            if 3 in ids or any(7000 <= value < 8000 for value in ids):
                target_ids = (17 if side == "left" else 53,)
            elif 2 in ids:
                target_ids = ((3006, 3007, 3016) if side == "left" else
                              (4006, 4007, 4016))
            elif 0 in ids:
                target_ids = (0,)
            else:
                target_ids = tuple(side_map[value] for value in ids if value in side_map)
            if not target_ids:
                continue
            sampling_mask = np.isin(coarse, target_ids)
            if 0 in ids:
                sampling_mask &= local_background
            mask = ndimage.binary_erosion(sampling_mask,
                                          structure=neighborhood, border_value=1)
            values = context.data[mask & (context.data > 0)]
            if values.size:
                means[group] = float(np.median(values))
                counts[group] = 10 + values.size * voxel_volume / self.resolution_mm**3
        return self._partial_volume_hyperparameters(atlas, classes, means, counts, context)

    def _partial_volume_hyperparameters(self, atlas, classes, means, counts, context):
        """Convolve atlas-derived tissue intensities before sampling thin labels."""
        grouped = np.zeros((len(atlas.vertices), len(means)), np.float32)
        for channel, group in enumerate(classes):
            grouped[:, group] += atlas.alphas[:, channel]
        device = torch.device(getattr(self, "_preparation_device", "cpu"))
        vertices = torch.as_tensor(atlas.vertices, dtype=torch.float32, device=device)
        tetra = torch.as_tensor(atlas.tetrahedra, dtype=torch.long, device=device)
        shape = tuple(np.ceil(atlas.vertices.max(0)).astype(int) + 2)
        if min(shape) <= 0:
            return means, counts
        priors, covered = rasterize_priors(vertices, tetra, torch.as_tensor(grouped, device=device), shape,
                                            background_channel=None)
        winning = priors.argmax(0)
        del priors
        def group(label):
            matches = np.flatnonzero(atlas.label_ids == label)
            return int(classes[matches[0]]) if len(matches) else None
        gm, wm, alveus = group(3), group(2), group(201)
        csf, fissure, molecular = group(4), group(215), group(245)
        synthetic = torch.as_tensor(means, device=device)[winning]
        if alveus is not None:
            synthetic[winning == alveus] = float(means[wm])
        if fissure is not None:
            synthetic[winning == fissure] = float(means[csf])
        if molecular is not None and molecular != gm:
            synthetic[winning == molecular] = float(means[wm])
        synthetic[~covered] = 0
        voxel_size = np.linalg.norm(context.image.affine[:3, :3], axis=0).mean()
        sigma = voxel_size / (2.355 * self.resolution_mm)
        if device.type == "cuda":
            from ..smoothing import _gaussian_filter3d
            smoothed = _gaussian_filter3d(synthetic, sigma)
        else:
            smoothed = torch.as_tensor(ndimage.gaussian_filter(synthetic.numpy(), sigma))
        for thin, first, second in ((alveus, gm, wm),
                                    (fissure, csf, gm),
                                    (molecular if molecular != gm else None, wm, gm)):
            if thin is None:
                continue
            # Only thin-label intensities leave the device for the existing
            # scalar median/KDE estimator, rather than full dense priors.
            samples = smoothed[winning == thin].cpu().numpy()
            if samples.size:
                if thin == alveus and samples.size > 1 and np.ptp(samples) > 0:
                    grid = np.linspace(samples.min(), samples.max(), 1000)
                    means[thin] = float(grid[np.argmax(gaussian_kde(samples)(grid))])
                else:
                    means[thin] = float(np.median(samples))
                counts[thin] = (counts[first] + counts[second]) / 2
        return means, counts

    def foreground_ids(self, atlas):
        return {int(value) for value in atlas.label_ids
                if (200 <= int(value) <= 246 and int(value) != 201)
                or 7000 <= int(value) < 8000}

    def postprocess(self, fit, context, atlas):
        raw = fit.labels.detach().cpu().numpy().astype(np.int32)
        mask = np.isin(raw, list(self.foreground_ids(atlas)))
        components, count = ndimage.label(mask)
        if count:
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            mask &= components == sizes.argmax()
        offset = 0 if self.side == "left" else 10000
        labels = np.where(mask, raw + offset, 0).astype(np.int32)
        native = _native(labels, fit.affine, context.image, 0).astype(np.int32)
        confidence = fit.posterior.max(0).values.detach().cpu().numpy().astype(np.float32)
        native_conf = _native(confidence, fit.affine, context.image, 1)
        support = ndimage.binary_dilation(np.isin(context.coarse_segmentation,
                                                 self.support_ids), iterations=2)
        native[~support] = 0
        voxel_volume = abs(np.linalg.det(fit.affine[:3, :3]))
        posterior = fit.posterior.detach().cpu()
        volumes = {int(label) + offset: float(posterior[index].sum() * voxel_volume)
                   for index, label in enumerate(atlas.label_ids)
                   if int(label) in self.foreground_ids(atlas)}
        return RecipeResult(fit, nib.Nifti1Image(labels, fit.affine), native,
                            native_conf, support, volumes, {})
