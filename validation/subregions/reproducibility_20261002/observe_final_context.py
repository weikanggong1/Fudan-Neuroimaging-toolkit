"""Hash actual returned preprocessing arrays while preserving production math.

This validation observer calls the original SubregionContext.prepare and raw
preparation functions exactly once, then records dtype/shape/affine/array SHA
and model choices. It saves no array content or image. Timing includes observer
overhead in the existing driver reports and separately records observer_seconds.
No reference is loaded, no solver option is changed and every hook is restored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
from time import monotonic, time

import numpy as np


def array_identity(value) -> dict:
    if value is None:
        return {"status": "not_available", "sha256": None}
    array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    return {"status": "observed", "shape": [int(size) for size in array.shape],
            "dtype": array.dtype.str, "original_strides_bytes": list(map(int, array.strides)),
            "bytes": int(array.nbytes), "hash_order": "C contiguous; no numeric conversion",
            "sha256": hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()}


def context_identity(context, phase) -> dict:
    coarse = context.coarse_segmentation
    return {"phase": phase,
            "image_geometry": {"shape": list(map(int, context.image.shape)),
                               "affine": np.asarray(context.image.affine).tolist(),
                               "affine_identity": array_identity(context.image.affine)},
            "data": array_identity(context.data),
            "coarse_segmentation": array_identity(coarse),
            "cortical_parcellation": array_identity(context.cortical_parcellation),
            "wmparc_proxy": array_identity(context.wmparc_proxy),
            "brain_mask": array_identity(None if coarse is None else np.asarray(coarse) > 0),
            "brain_mask_definition": "coarse > 0; exact raw FAST mask at prepared_context phase",
            "metadata_and_model_choices": context.metadata}


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-driver", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    driver = args.run_driver.resolve()
    if driver.name != "run_unified.py" or not driver.is_file():
        raise ValueError("--run-driver must name the frozen production run_unified.py")
    if "--structures" not in remaining or remaining[remaining.index("--structures") + 1] != "all":
        raise ValueError("This final observer requires --structures all")
    if "--quick" in remaining or "--output-dir" not in remaining:
        raise ValueError("Final observer requires complete real-data schedules and output-dir")
    output = Path(remaining[remaining.index("--output-dir") + 1])
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "context_identity.json"
    if report_path.exists():
        raise ValueError("Preserve an existing context identity report")
    import fnit
    from fnit.gems.context import SubregionContext
    from fnit.gems import preprocessing
    source_root = Path(fnit.__file__).resolve().parent
    expected_source_root = (driver.parents[2] / "src/fnit").resolve()
    if source_root != expected_source_root:
        raise ValueError("Imported fnit does not match the declared frozen run-driver source")

    def source_hashes():
        return {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(source_root.rglob("*.py"))}

    before = source_hashes()
    metadata = {"validation_only": True, "solver_options_changed": False,
                "reference_used_by_observer": False, "arrays_saved": False,
                "started_unix": time(), "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "driver": str(driver), "driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(),
                "runtime_source_root": str(source_root), "runtime_source_sha256_before": before,
                "input_arguments": remaining, "contexts": [], "observer_seconds": 0.0,
                "timing_scope": "Production compute/API/process wall retain actual observer overhead; observer_seconds separately measured."}
    original_prepare_descriptor = SubregionContext.__dict__["prepare"]
    original_prepare = SubregionContext.prepare
    original_raw = preprocessing.prepare_automatic_raw_input
    original_argv, original_path = sys.argv, list(sys.path)

    def save():
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(metadata, indent=2, default=lambda value:
                             value.item() if isinstance(value, np.generic) else str(value)) + "\n")
        temporary.replace(report_path)

    def observe(context, phase):
        start = monotonic()
        record = context_identity(context, phase)
        metadata["contexts"].append(record)
        # Include the report serialization/write overhead; update its measured
        # total in the final save after all production calls have finished.
        save()
        elapsed = monotonic() - start
        record["observer_seconds"] = elapsed
        metadata["observer_seconds"] += elapsed
        return context

    def observed_prepare(cls, *positional, **options):
        return observe(original_prepare(*positional, **options), "prepared_context")

    def observed_raw(*positional, **options):
        return observe(original_raw(*positional, **options), "raw_processing_context")

    SubregionContext.prepare = classmethod(observed_prepare)
    preprocessing.prepare_automatic_raw_input = observed_raw
    sys.path.insert(0, str(driver.parent))
    sys.argv = [str(driver), *remaining]
    try:
        runpy.run_path(str(driver), run_name="__main__")
    finally:
        SubregionContext.prepare = original_prepare_descriptor
        preprocessing.prepare_automatic_raw_input = original_raw
        sys.argv = original_argv
        sys.path[:] = original_path
        after = source_hashes()
        metadata.update(finished_unix=time(), runtime_source_sha256_after=after,
                        runtime_source_unchanged=after == before)
        save()


if __name__ == "__main__":
    main()
