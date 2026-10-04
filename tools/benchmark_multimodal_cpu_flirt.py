"""Isolated official benchmark adapter for every public FLIRT mode.

The FNIT production API never invokes FSL.  ``reference_command`` only builds
the separate benchmark command; the common runner launches and times it.
Case schema: ``kind`` is ``registration`` or ``applyxfm``; ``input`` and
``reference`` are actual images, and ``kwargs`` contains public FNIT options.
"""

import json
from pathlib import Path


def _paths(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {"moved": str(output_dir / "moved.nii.gz"),
            "matrix": str(output_dir / "transform.mat")}


def run_case(case: dict, output_dir: Path, device: str) -> dict[str, str]:
    """Read input, run the complete supported path, and save all image/matrix outputs."""
    from fnit.flirt import TorchFLIRT, run_flirt

    paths = _paths(output_dir)
    kwargs = dict(case.get("kwargs", {}))
    kind = case.get("kind", "registration")
    if kind == "applyxfm":
        run_flirt(input=case["input"], reference=case["reference"],
                  output=paths["moved"], omat=paths["matrix"], device=device,
                  applyxfm=True, overwrite=True, **kwargs)
    elif kind == "registration":
        model_options = {name: kwargs.pop(name) for name in (
            "angular_search", "dof", "cost", "execution",
            "candidate_batch_size", "memory_budget_gb") if name in kwargs}
        model = TorchFLIRT(device=device, **model_options)
        result = model.run(input=case["input"], reference=case["reference"],
                           output=paths["moved"], omat=paths["matrix"],
                           overwrite=True, **kwargs)
        # Runtime QC is recorded as diagnostics, not a reference output.
        (Path(output_dir) / "fnit_qc.json").write_text(
            json.dumps(result.qc, ensure_ascii=False, indent=2) + "\n")
    else:
        raise ValueError("FLIRT case kind must be registration or applyxfm")
    return paths


def reference_command(case: dict, output_dir: Path, resources: dict) -> list[str]:
    """Build the original FLIRT command with identical numerical options."""
    paths = _paths(output_dir)
    executable = resources.get("flirt_command", resources.get("fsl_flirt"))
    if executable is None:
        executable = str(Path(resources["fsl_dir"]) / "bin" / "flirt")
    command = [str(executable), "-in", case["input"], "-ref", case["reference"],
               "-out", paths["moved"], "-omat", paths["matrix"]]
    kwargs = dict(case.get("kwargs", {}))
    if case.get("kind", "registration") == "applyxfm":
        command.append("-applyxfm")
        if kwargs.get("usesqform", False):
            command.append("-usesqform")
        elif kwargs.get("init") is None:
            raise ValueError("applyxfm requires init or usesqform=True")
    else:
        command += ["-dof", str(kwargs.get("dof", 12)),
                    "-cost", kwargs.get("cost", "corratio")]
        if not kwargs.get("angular_search", True):
            command.append("-nosearch")
        for name in ("inweight", "refweight"):
            if kwargs.get(name) is not None:
                command += ["-" + name, str(kwargs[name])]
    if kwargs.get("init") is not None:
        command += ["-init", str(kwargs["init"])]
    return command


def reference_outputs(case: dict, output_dir: Path, resources: dict) -> dict[str, str]:
    return _paths(output_dir)


def compare_case(case: dict, output_sets: dict, resources: dict) -> dict:
    """Post-timing physical displacement, foreground accuracy and file contracts.

    Reuse the existing FLIRT report's 13x13x13 moving world grid and fixed
    official-output foreground. This does not run a fit or an official command.
    """
    import nibabel as nib
    import numpy as np
    from types import SimpleNamespace
    from fnit.flirt import core
    from tools.benchmark_flirt_gpu import paired_metrics

    moving, fixed = nib.load(case["input"]), nib.load(case["reference"])
    official = output_sets["official"]
    comparisons = {"method": "benchmark_flirt_gpu.paired_metrics: 13^3 moving world grid; official nonzero output foreground"}
    for backend in ("candidate", "baseline"):
        if backend not in output_sets:
            continue
        paths = output_sets[backend]
        image = nib.load(paths["moved"])
        qc_path = Path(paths["moved"]).parent / "fnit_qc.json"
        has_fit_qc = case.get("kind", "registration") == "registration" and qc_path.is_file()
        # The shared metric helper also carries registration diagnostics.
        # applyxfm has no optimization cost; omit those fields from its report.
        qc = (json.loads(qc_path.read_text()) if has_fit_qc
              else {"cost_value": float("nan"), "cost_evaluations": 0})
        result = SimpleNamespace(matrix=np.loadtxt(paths["matrix"]), moved=image, qc=qc)
        metrics = paired_metrics(result, moving, fixed, official["matrix"],
                                 official["moved"], None, core)
        if not has_fit_qc:
            metrics.pop("cost")
            metrics.pop("cost_evaluations")
        reference_image = nib.load(official["moved"])
        for form in ("qform", "sform"):
            left, left_code = getattr(image, "get_" + form)(coded=True)
            right, right_code = getattr(reference_image, "get_" + form)(coded=True)
            metrics["grid_header"][form + "_matrix_max_abs"] = (
                float(np.max(np.abs(left - right))) if left_code and right_code else None)
        metrics["grid_header"]["candidate_dtype"] = str(image.header.get_data_dtype())
        metrics["grid_header"]["official_dtype"] = str(reference_image.header.get_data_dtype())
        comparisons[backend + "_vs_official"] = metrics
    return comparisons


def sample_cases(input_dir: Path, asset_dir: Path) -> list[dict]:
    """Prepare public-image weights/init and the supported feature matrix.

    Weights are continuous functions of the real image grid and nonzero
    intensities.  They are generated once and shared with both implementations.
    The known transform is declared independently, not fitted to an oracle.
    """
    import nibabel as nib
    import numpy as np

    input_dir, asset_dir = Path(input_dir), Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    images = [input_dir / "sub-02_T1w.nii.gz", input_dir / "sub-01_T1w.nii.gz"]
    weights = []
    for index, path in enumerate(images):
        image = nib.load(path)
        data = image.get_fdata(dtype=np.float32)
        ramp = np.linspace(0.2, 1.0, data.shape[0], dtype=np.float32)[:, None, None]
        weight = (data > 0).astype(np.float32) * ramp
        output = asset_dir / f"public_weight_{index}.nii.gz"
        header = image.header.copy()
        header.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(weight, image.affine, header), output)
        weights.append(str(output))
    matrix = np.eye(4)
    matrix[:3, 3] = (1.25, -0.75, 0.5)
    init = asset_dir / "declared_init.mat"
    np.savetxt(init, matrix, fmt="%.12g")
    base = {"input": str(images[0]), "reference": str(images[1]),
            "adapter": "tools/benchmark_multimodal_cpu_flirt.py"}
    cases = []
    for dof, cost in ((12, "corratio"), (6, "normmi")):
        for name, extra in (("default", {}), ("init", {"init": str(init)}),
                            ("nosearch", {"angular_search": False})):
            cases.append(dict(base, id=f"flirt_{dof}_{cost}_{name}",
                              kind="registration", kwargs={"dof": dof, "cost": cost, **extra}))
    for name, extra in (("input_weight", {"inweight": weights[0]}),
                        ("reference_weight", {"refweight": weights[1]}),
                        ("both_weights", {"inweight": weights[0], "refweight": weights[1]})):
        cases.append(dict(base, id="flirt_12_corratio_" + name,
                          kind="registration", kwargs={"dof": 12, "cost": "corratio", **extra}))
    for name, kwargs in (("known_matrix", {"init": str(init)}),
                         ("usesqform", {"usesqform": True})):
        cases.append(dict(base, id="flirt_applyxfm_" + name,
                          kind="applyxfm", kwargs=kwargs))
    return cases
