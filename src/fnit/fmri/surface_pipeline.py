"""Connect the MNI volume pipeline to UKB-style fsLR32k projection."""

from dataclasses import dataclass
from pathlib import Path

from .surface import SurfaceHemisphere, SurfaceProjectionResult, run_surface_projection
from .surface_qc import SurfaceQCResult, make_ribbon_goodvoxels


@dataclass(frozen=True)
class SurfacePipelineInputs:
    """Subject MNI surfaces and ROI images prepared by a structural pipeline."""

    left: SurfaceHemisphere
    right: SurfaceHemisphere
    subject_rois: str | Path
    atlas_rois: str | Path
    wb_command: str | Path = "wb_command"
    goodvoxels: str | Path | None = None


@dataclass(frozen=True)
class SurfacePipelineResult:
    projection: SurfaceProjectionResult
    qc: SurfaceQCResult | None


def run_surface_from_mni(
    clean_mni: str | Path,
    mni_reference: str | Path,
    inputs: SurfacePipelineInputs,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> SurfacePipelineResult:
    """Project an existing cleaned MNI BOLD with registered subject surfaces.

    When ``goodvoxels`` is absent, estimate it from the white/pial ribbon and
    cleaned fMRI. A supplied mask permits a fixed-input comparison. Sphere
    registration and subject/atlas ROI creation remain upstream inputs.
    """
    output = Path(output_dir).expanduser().resolve()
    qc = None
    goodvoxels = inputs.goodvoxels
    if goodvoxels is None:
        qc = make_ribbon_goodvoxels(
            clean_bold=clean_mni,
            reference=mni_reference,
            left_white=inputs.left.white,
            left_pial=inputs.left.pial,
            right_white=inputs.right.white,
            right_pial=inputs.right.pial,
            output_dir=output / "qc",
            wb_command=inputs.wb_command,
        )
        goodvoxels = qc.goodvoxels
    projection = run_surface_projection(
        clean_mni=clean_mni,
        mni_reference=mni_reference,
        goodvoxels=goodvoxels,
        subject_rois=inputs.subject_rois,
        atlas_rois=inputs.atlas_rois,
        left=inputs.left,
        right=inputs.right,
        output_dir=output / "projection",
        wb_command=inputs.wb_command,
        overwrite=overwrite,
    )
    return SurfacePipelineResult(projection=projection, qc=qc)
