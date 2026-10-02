"""Exact per-call NIfTI geometry guards for reusable sampling plans."""

from __future__ import annotations

from dataclasses import dataclass
import io

import numpy as np


def _header_bytes(image):
    extensions = io.BytesIO()
    image.header.extensions.write_to(extensions, byteswap=False)
    return image.header.binaryblock, extensions.getvalue()


@dataclass(frozen=True)
class SamplingGeometry:
    """Capture spatial coordinates; optionally also the output header."""

    shape: tuple
    affine: bytes
    pixdim: tuple
    fsl_scaled_mm: bytes
    image_class: type | None = None
    header: tuple | None = None

    @classmethod
    def capture(cls, image, *, output_header=False):
        affine = np.asarray(image.affine, dtype=np.float64)
        pixdim = tuple(float(value) for value in image.header.get_zooms()[:3])
        # The same FSL scaled-mm axes are used by ApplyWarp and MMORF's FLIRT
        # conversion: header pixdim, plus the radiological storage x flip.
        scaled_mm = np.diag((*pixdim, 1.0))
        if np.linalg.det(affine[:3, :3]) > 0:
            scaled_mm[0, 0] *= -1
            scaled_mm[0, 3] = pixdim[0] * (image.shape[0] - 1)
        return cls(
            tuple(image.shape[:3]), affine.tobytes(), pixdim,
            scaled_mm.tobytes(),
            type(image) if output_header else None,
            _header_bytes(image) if output_header else None,
        )

    def require(self, image, name):
        current = self.capture(image, output_header=self.header is not None)
        if current != self:
            raise ValueError(
                f"{name} geometry or reference header differs from the prepared "
                "sampling plan; prepare a separate plan for this image grid"
            )
