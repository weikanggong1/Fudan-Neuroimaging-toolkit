"""Raw UKB-format dMRI to native and MNI-space DTI/NODDI maps."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from .._dmri import configure_device, image_like, load_bvals
from ..amico_noddi import TorchAMICONODDI
from ..dtifit import TorchDTIFIT, select_shell
from ..eddy import TorchEDDY
from ..eddy.ukb import _brain_mask, prepare_ukb_eddy
from ..flirt import TorchFLIRT
from ..fnirt import resolve_fnirt_config
from ..mmorf import apply_mmorf_warp, run_mmorf
from ..synthstrip import SynthStrip
from ..topup import run_ukb_topup
from ..topup.ukb import _metadata
from .bids import locate_bids_dwi, stage_bids_dwi
from .tbss import TBSSConfig, TorchTBSS


STANDARD_MAP_NAMES = ("FA", "MD", "L1", "L2", "L3", "MO", "ICVF", "OD", "ISOVF")


@dataclass(frozen=True)
class DMRIPipelineResult:
    native_maps: dict[str, nib.Nifti1Image]
    standard_maps: dict[str, nib.Nifti1Image]
    registration_backend: str
    output_dir: Path
    qc: dict


def _require(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _same_grid(paths):
    images = [nib.load(os.fspath(path)) for path in paths]
    first = images[0]
    if any(
        image.shape[:3] != first.shape[:3]
        or not np.allclose(image.affine, first.affine, atol=1e-5, rtol=0)
        for image in images[1:]
    ):
        raise ValueError("all standard templates must use the same MNI grid")


def _prepare_ap_only(raw_dir, output_dir, *, overwrite):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    image_path = _require(raw_dir / "AP.nii.gz")
    bval_path = _require(raw_dir / "AP.bval")
    bvec_path = _require(raw_dir / "AP.bvec")
    _require(raw_dir / "AP.json")
    image = nib.load(str(image_path))
    values = np.asarray(image.dataobj, dtype=np.float32)
    bvals = load_bvals(bval_path)
    if values.ndim != 4 or values.shape[3] != bvals.size:
        raise ValueError("AP image and bvals have inconsistent volume counts")
    b0 = values[..., bvals < 100]
    if b0.shape[3] == 0:
        raise ValueError("AP acquisition contains no b<100 volume")
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / "nodif_brain_mask.nii.gz"
    acqp_path = output_dir / "acqparams.txt"
    index_path = output_dir / "eddy_index.txt"
    existing = [path for path in (mask_path, acqp_path, index_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"output exists: {existing[0]}; pass overwrite=True")
    mask = _brain_mask(image_like(b0.mean(3), image))
    nib.save(image_like(mask.astype(np.float32), image), str(mask_path))
    pe, readout = _metadata(raw_dir, "AP", values.shape[:3])
    np.savetxt(acqp_path, np.asarray([(*pe, readout)]), fmt=("%g", "%g", "%g", "%.7g"))
    np.savetxt(index_path, np.ones((1, bvals.size), dtype=int), fmt="%d")
    return {
        "imain": image_path,
        "mask": mask_path,
        "acqp": acqp_path,
        "index": index_path,
        "bvecs": bvec_path,
        "bvals": bval_path,
        "topup": None,
        "ref_scan_no": int(np.flatnonzero(bvals < 100)[0]),
    }


class DMRIPipeline:
    """Run one UKB-format or raw BIDS DWI through DTI/NODDI and registration."""

    def __init__(
        self,
        device=None,
        *,
        registration_backend="tbss",
        fnirt_config=None,
        synthstrip_weights=None,
        dti_shell=1000,
        dti_tolerance=100,
        bvec_source="rotated",
        noddi_fit_method="amico",
    ):
        if registration_backend not in ("tbss", "mmorf"):
            raise ValueError("registration_backend must be 'tbss' or 'mmorf'")
        if registration_backend != "tbss" and fnirt_config is not None:
            raise ValueError("fnirt_config requires registration_backend='tbss'")
        if bvec_source not in ("rotated", "raw"):
            raise ValueError("bvec_source must be 'rotated' or 'raw'")
        if noddi_fit_method not in ("amico", "classic"):
            raise ValueError("noddi_fit_method must be 'amico' or 'classic'")
        self.device = configure_device(device)
        self.registration_backend = registration_backend
        self.fnirt_config = (
            resolve_fnirt_config(fnirt_config, default="tbss")
            if registration_backend == "tbss" else None
        )
        self.synthstrip_weights = synthstrip_weights
        self.dti_shell = float(dti_shell)
        self.dti_tolerance = float(dti_tolerance)
        self.bvec_source = bvec_source
        self.noddi_fit_method = noddi_fit_method

    def run(
        self,
        raw_dir,
        output_dir,
        *,
        fa_template,
        fa_skeleton=None,
        t1=None,
        t1_template=None,
        tensor_template=None,
        overwrite=False,
    ):
        raw_dir = Path(raw_dir).expanduser()
        output_dir = Path(output_dir).expanduser()
        _require(raw_dir / "AP.nii.gz")
        _require(raw_dir / "AP.bval")
        _require(raw_dir / "AP.bvec")
        _require(raw_dir / "AP.json")
        fa_template = _require(fa_template)
        if self.registration_backend == "tbss":
            fa_skeleton = _require(fa_skeleton)
            _same_grid((fa_template, fa_skeleton))
        else:
            t1 = _require(t1)
            t1_template = _require(t1_template)
            tensor_template = _require(tensor_template)
            _same_grid((fa_template, t1_template, tensor_template))
            if self.synthstrip_weights is None:
                raise ValueError("synthstrip_weights is required for the MMORF branch")
        report_path = output_dir / "dmri_pipeline_report.json"
        if report_path.exists() and not overwrite:
            raise FileExistsError(f"output exists: {report_path}; pass overwrite=True")
        output_dir.mkdir(parents=True, exist_ok=True)
        total_started = time.perf_counter()
        timings = {}

        topup_dir = output_dir / "topup"
        pa_paths = tuple(
            raw_dir / f"PA.{suffix}" for suffix in ("nii.gz", "bval", "json")
        )
        pa_present = tuple(path.is_file() for path in pa_paths)
        if any(pa_present) and not all(pa_present):
            missing = next(
                path.name for path, present in zip(pa_paths, pa_present)
                if not present
            )
            raise FileNotFoundError(
                f"incomplete PA acquisition; missing {missing}"
            )
        has_pa = all(pa_present)
        started = time.perf_counter()
        if has_pa:
            run_ukb_topup(
                raw_dir, topup_dir, device=self.device, overwrite=overwrite
            )
            eddy_inputs = prepare_ukb_eddy(
                raw_dir,
                topup_dir,
                output_dir / "eddy",
                device=self.device,
                overwrite=overwrite,
            )
        else:
            eddy_inputs = _prepare_ap_only(
                raw_dir, output_dir / "eddy", overwrite=overwrite
            )
        timings["topup_and_eddy_preparation"] = time.perf_counter() - started

        started = time.perf_counter()
        eddy_root = output_dir / "eddy" / "data"
        eddy = TorchEDDY(device=self.device).run(
            **eddy_inputs, out=eddy_root, overwrite=overwrite
        )
        timings["eddy"] = time.perf_counter() - started
        corrected = eddy_root.with_name(eddy_root.name + ".nii.gz")
        rotated_bvecs = eddy_root.with_name(eddy_root.name + ".eddy_rotated_bvecs")
        fitting_bvecs = (
            rotated_bvecs if self.bvec_source == "rotated" else raw_dir / "AP.bvec"
        )
        mask_path = Path(eddy_inputs["mask"])

        native_dir = output_dir / "native"
        native_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        shell_image, shell_bval, shell_bvec = select_shell(
            corrected,
            raw_dir / "AP.bval",
            fitting_bvecs,
            native_dir / "data_1_shell",
            shell=self.dti_shell,
            tolerance=self.dti_tolerance,
            overwrite=overwrite,
        )
        dti = TorchDTIFIT(device=self.device).run(
            shell_image,
            mask_path,
            shell_bvec,
            shell_bval,
            output_prefix=native_dir / "dti",
            save_tensor=True,
            overwrite=overwrite,
        )
        timings["dtifit"] = time.perf_counter() - started

        started = time.perf_counter()
        noddi = TorchAMICONODDI(
            device=self.device, fit_method=self.noddi_fit_method
        ).run(
            corrected,
            mask_path,
            fitting_bvecs,
            raw_dir / "AP.bval",
            output_dir=native_dir,
            naming="ukb",
            overwrite=overwrite,
        )
        timings["noddi"] = time.perf_counter() - started
        native_maps = {
            name: dti.maps[name] for name in ("FA", "MD", "L1", "L2", "L3", "MO")
        }
        native_maps.update({"ICVF": noddi.ndi, "OD": noddi.odi, "ISOVF": noddi.fwf})
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        started = time.perf_counter()
        registration_dir = output_dir / "registration"
        if self.registration_backend == "tbss":
            registered = TorchTBSS(
                device=self.device, config=TBSSConfig(fnirt=self.fnirt_config)
            ).run(
                native_maps,
                fa_template,
                fa_skeleton,
                output_dir=registration_dir,
                overwrite=overwrite,
            )
            standard_maps = registered.standard_maps
            registration_qc = registered.qc
        else:
            registration_dir.mkdir(parents=True, exist_ok=True)
            stripped = SynthStrip(
                weights=self.synthstrip_weights, device=self.device
            )(os.fspath(t1))
            t1_brain_path = registration_dir / "t1_brain.nii.gz"
            t1_mask_path = registration_dir / "t1_brain_mask.nii.gz"
            stripped.image.save(t1_brain_path)
            stripped.mask.save(t1_mask_path)
            scalar_affine = TorchFLIRT(device=self.device)(t1_brain_path, t1_template)
            tensor_affine = TorchFLIRT(device=self.device)(
                native_dir / "dti_FA.nii.gz", fa_template
            )
            scalar_matrix_path = registration_dir / "t1_to_MNI_affine.mat"
            tensor_matrix_path = registration_dir / "dti_FA_to_MNI_affine.mat"
            np.savetxt(scalar_matrix_path, scalar_affine.matrix, fmt="%.12g")
            np.savetxt(tensor_matrix_path, tensor_affine.matrix, fmt="%.12g")
            mmorf = run_mmorf(
                t1_brain_path,
                t1_template,
                dti.maps["tensor"],
                tensor_template,
                moving_scalar_affine=scalar_affine.matrix,
                moving_tensor_affine=tensor_affine.matrix,
                device=self.device,
                output_dir=registration_dir,
                overwrite=overwrite,
            )
            standard_dir = registration_dir / "standard"
            standard_dir.mkdir(parents=True, exist_ok=True)
            standard_maps = {}
            for name, image in native_maps.items():
                warped = apply_mmorf_warp(
                    image,
                    t1_template,
                    mmorf.warp,
                    affine=tensor_affine.matrix,
                    device=self.device,
                )
                nib.save(warped, str(standard_dir / f"{name}.nii.gz"))
                standard_maps[name] = warped
            registration_qc = {
                "t1_flirt": scalar_affine.qc,
                "fa_flirt": tensor_affine.qc,
                "mmorf": mmorf.qc,
            }
        timings["registration_and_map_propagation"] = time.perf_counter() - started
        qc = {
            "device": str(self.device or "cpu"),
            "dtype": "float32 images; solver-specific float64 where documented",
            "tf32": bool(str(self.device).startswith("cuda")),
            "registration_backend": self.registration_backend,
            "topup_applied": has_pa,
            "native_maps": list(STANDARD_MAP_NAMES),
            "standard_maps": list(STANDARD_MAP_NAMES),
            "common_standard_output_contract": True,
            "dti_shell": self.dti_shell,
            "dti_tolerance": self.dti_tolerance,
            "bvec_source": self.bvec_source,
            "noddi_fit_method": self.noddi_fit_method,
            "timings_seconds": timings,
            "elapsed_seconds": time.perf_counter() - total_started,
            "eddy": eddy.qc,
            "dtifit": dti.qc,
            "noddi": noddi.qc,
            "registration": registration_qc,
        }
        report_path.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
        return DMRIPipelineResult(
            native_maps, standard_maps, self.registration_backend, output_dir, qc
        )

    def run_bids(
        self,
        bids_root,
        output_dir,
        *,
        subject,
        session=None,
        run=None,
        acquisition=None,
        direction=None,
        fa_template,
        fa_skeleton=None,
        t1=None,
        t1_template=None,
        tensor_template=None,
        overwrite=False,
    ):
        """Select one raw BIDS DWI run and reuse the existing AP/PA pipeline."""
        inputs = locate_bids_dwi(
            bids_root, subject=subject, session=session, run=run,
            acquisition=acquisition, direction=direction, t1=t1,
            select_t1=self.registration_backend == "mmorf",
        )
        if self.registration_backend == "mmorf" and inputs.t1w is None:
            raise ValueError("MMORF requires a T1w image in BIDS anat/ or explicit t1")
        output_dir = Path(output_dir).expanduser()
        if (output_dir / "dmri_pipeline_report.json").exists() and not overwrite:
            raise FileExistsError(output_dir / "dmri_pipeline_report.json")
        raw_dir = stage_bids_dwi(inputs, output_dir / "bids_input", overwrite=overwrite)
        return self.run(
            raw_dir, output_dir, fa_template=fa_template,
            fa_skeleton=fa_skeleton,
            t1=inputs.t1w if self.registration_backend == "mmorf" else None,
            t1_template=t1_template, tensor_template=tensor_template,
            overwrite=overwrite,
        )


__all__ = ["DMRIPipeline", "DMRIPipelineResult", "STANDARD_MAP_NAMES"]
