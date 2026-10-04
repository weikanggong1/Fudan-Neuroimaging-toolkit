"""WMH-SynthSeg inference and output volumes.

Adapted from FreeSurfer 8.2.0-1 ``mri_WMHsynthseg/WMHSynthSeg/inference.py``.
See ``THIRD_PARTY_NOTICES.md`` and ``licenses/FreeSurfer.txt``.
"""

from dataclasses import dataclass
import csv
import os
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

from .._dmri import configure_device
from .._nib import FNITNifti1Image, load_image
from ..weights import resolve_weights
from .model import UNet3D
from .spatial import align_volume_to_ref, myzoom_torch


LABEL_IDS = (0, 14, 15, 16, 24, 77, 85, 2, 3, 4, 7, 8, 10, 11, 12, 13,
             17, 18, 26, 28, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53,
             54, 58, 60)
LABEL_NAMES = ('background', '3rd-ventricle', '4th-ventricle', 'brainstem',
               'extracerebral_CSF', 'WMH', 'optic-chiasm', 'left-white-matter',
               'left-cortex', 'left-lateral-ventricle', 'left-cerebellum-white-matter',
               'left-cerebellum-cortex', 'left-thalamus', 'left-caudate',
               'left-putamen', 'left-pallidum', 'left-hippocampus', 'left-amygdala',
               'left-accumbens', 'left-ventral-DC', 'right-white-matter',
               'right-cortex', 'right-lateral-ventricle',
               'right-cerebellum-white-matter', 'right-cerebellum-cortex',
               'right-thalamus', 'right-caudate', 'right-putamen',
               'right-pallidum', 'right-hippocampus', 'right-amygdala',
               'right-accumbens', 'right-ventral-DC')


def _output_image(data, affine):
    """Match original MRIwrite's fresh float32 NIfTI header on the new grid.

    The output is RAS-aligned/upscaled and possibly cropped. Input scanner
    extensions and qform/sform codes do not describe that grid and must not
    be copied into the segmentation or lesion-probability image.
    """
    return FNITNifti1Image(np.asarray(data, dtype=np.float32), affine,
                          nib.Nifti1Header())


@dataclass
class WMHResult:
    segmentation: FNITNifti1Image
    lesion_probability: FNITNifti1Image | None
    volumes_mm3: dict[int, float]


def _write_volumes_csv(volumes_mm3, segmentation_path, csv_path):
    ordered = np.asarray([volumes_mm3[label] for label in LABEL_IDS], dtype=np.float32)
    with Path(csv_path).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Input-file', 'Intracranial-volume',
                         *(f'{name}({label})' for label, name in zip(LABEL_IDS, LABEL_NAMES)
                           if label != 0)])
        writer.writerow([str(segmentation_path), str(np.sum(ordered[1:])),
                         *(str(value) for value in ordered[1:])])


class WMHSynthSeg:
    def __init__(self, weights=None, device='cpu', threads=None):
        self.device = configure_device(device)
        if threads is not None:
            torch.set_num_threads(os.cpu_count() if threads < 0 else threads)
        checkpoint_path = resolve_weights('WMH-SynthSeg_v10_231110.pth', explicit=weights)
        self.model = UNet3D().to(self.device)
        # The official checkpoint contains NumPy scalars outside its state_dict.
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.eval()
        self.labels = torch.tensor(LABEL_IDS, device=self.device)

    @torch.no_grad()
    def __call__(self, image, crop=False, save_lesion_probabilities=False):
        volume = load_image(image)
        data, affine = volume.get_fdata(), volume.affine
        data = np.squeeze(data)
        if data.ndim != 3:
            raise ValueError('WMH-SynthSeg accepts a single 3D volume')
        image_torch = torch.tensor(data.astype(float), device=self.device)
        image_torch, aff2 = align_volume_to_ref(image_torch, affine,
                                                aff_ref=np.eye(4), return_aff=True,
                                                n_dims=3)
        image_torch = image_torch / torch.max(image_torch)
        voxsize = np.sqrt(np.sum(aff2 ** 2, axis=0))[:-1]
        upscaled = myzoom_torch(image_torch, voxsize, device=self.device)
        aff_upscaled = aff2.copy()
        for axis in range(3):
            aff_upscaled[:-1, axis] = aff_upscaled[:-1, axis] / voxsize[axis]
        aff_upscaled[:-1, -1] -= aff_upscaled[:-1, :-1] @ (0.5 * (voxsize - 1))
        if crop:
            upscaled, aff_upscaled = self._crop(upscaled, aff_upscaled)
        shape = upscaled.shape
        padded_shape = tuple((np.ceil(np.array(shape) / 32) * 32).astype(int))
        padded = torch.zeros(padded_shape, device=self.device)
        padded[:shape[0], :shape[1], :shape[2]] = upscaled
        pred1 = self.model(padded[None, None])[0, :33, :shape[0], :shape[1], :shape[2]]
        pred2 = torch.flip(self.model(torch.flip(padded, [0])[None, None]), [2])
        pred2 = pred2[0, :33, :shape[0], :shape[1], :shape[2]]
        flip_channels = list(range(7)) + list(range(20, 33)) + list(range(7, 20))
        probabilities = 0.5 * torch.softmax(pred1, dim=0)
        probabilities += 0.5 * torch.softmax(pred2[flip_channels], dim=0)
        segmentation = self.labels[torch.argmax(probabilities, dim=0)].cpu().numpy()
        volumes = probabilities.sum(dim=(1, 2, 3)).cpu().numpy()
        # FreeSurfer MRIwrite uses a default NIfTI header, yielding float32 labels.
        seg_volume = _output_image(segmentation, aff_upscaled)
        lesion_volume = None
        if save_lesion_probabilities:
            lesion = probabilities[LABEL_IDS.index(77)].cpu().numpy()
            lesion_volume = _output_image(lesion, aff_upscaled)
        return WMHResult(seg_volume, lesion_volume,
                         {label: float(value) for label, value in zip(LABEL_IDS, volumes)})


    def _crop(self, upscaled, affine):
        target = (192, 224, 192)
        cuboid = torch.zeros(target, device=self.device)
        input_starts = [max(0, int(np.floor((s - t) / 2))) for s, t in zip(upscaled.shape, target)]
        output_starts = [max(0, int(np.floor((t - s) / 2))) for s, t in zip(upscaled.shape, target)]
        length = [min(s, t) for s, t in zip(upscaled.shape, target)]
        ins = tuple(slice(start, start + n) for start, n in zip(input_starts, length))
        outs = tuple(slice(start, start + n) for start, n in zip(output_starts, length))
        cuboid[outs] = upscaled[ins]
        prelim = self.model(cuboid[None, None])[0, :33]
        p = torch.softmax(prelim, dim=0)
        ventricle = sum(p[LABEL_IDS.index(label)] for label in (10, 49, 4, 43))
        grid = torch.meshgrid(*(torch.arange(size, device=self.device) for size in target),
                              indexing='ij')
        den = ventricle.sum()
        starts = [max(0, int((ventricle * axis).sum() / den + initial - size / 2))
                  for axis, initial, size in zip(grid, input_starts, target)]
        upscaled = upscaled[tuple(slice(start, None) for start in starts)]
        affine = affine.copy()
        affine[:3, 3] += affine[:3, :3] @ np.asarray(starts)
        upscaled = upscaled[tuple(slice(0, size) for size in target)]
        return upscaled, affine
