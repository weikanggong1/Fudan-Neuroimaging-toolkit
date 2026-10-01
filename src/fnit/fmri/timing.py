"""Timing metadata after realignment, following fixed fMRIPrep output rules.

Reference: fMRIPrep 25.2.4,
``fmriprep/workflows/bold/outputs.py::prepare_timing_parameters``.
Modified in FNIT (2026): explicit processing flags, validation and independent
NumPy handling. Copyright The NiPreps Developers, Apache License 2.0.
No workflow runtime is required. See THIRD_PARTY_NOTICES.md.
"""

from copy import deepcopy
from collections.abc import Mapping
import math
from numbers import Real
from typing import Any

import numpy as np


def _finite_number(value, name, *, strictly_positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (strictly_positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if strictly_positive else 'nonnegative'}")


def prepare_timing_parameters(
    metadata: Mapping[str, Any], *, slice_timing_corrected: bool,
    reference_fraction: float = 0.5,
) -> dict[str, Any]:
    """Return truthful post-realignment timing fields without raw SliceTiming.

    ``metadata`` contains inherited raw BIDS timing fields in seconds.
    ``slice_timing_corrected`` describes this output's actual processing;
    ``reference_fraction`` chooses a point in the slice acquisition range.
    Empty or single-value SliceTiming is treated as missing, as upstream.
    Sparse acquisition timing is retained even when STC is disabled.
    """
    if not isinstance(slice_timing_corrected, bool):
        raise ValueError("slice_timing_corrected must be a bool")
    _finite_number(reference_fraction, "reference_fraction")
    if reference_fraction > 1:
        raise ValueError("reference_fraction must be in [0, 1]")
    result = {
        key: deepcopy(metadata[key]) for key in (
            "RepetitionTime", "VolumeTiming", "DelayTime", "AcquisitionDuration",
        ) if key in metadata
    }
    for key in ("RepetitionTime", "DelayTime", "AcquisitionDuration"):
        if key in result:
            _finite_number(result[key], key, strictly_positive=key != "DelayTime")
    if "VolumeTiming" in result:
        volumes = np.asarray(result["VolumeTiming"], dtype=np.float64)
        if (volumes.ndim != 1 or not len(volumes) or not np.isfinite(volumes).all()
                or np.any(volumes < 0) or np.any(np.diff(volumes) <= 0)):
            raise ValueError("VolumeTiming must contain finite increasing nonnegative times")
    slices = np.asarray(metadata.get("SliceTiming", []), dtype=np.float64)
    if slices.ndim != 1 or not np.isfinite(slices).all() or np.any(slices < 0):
        raise ValueError("SliceTiming must contain finite nonnegative times")
    if "RepetitionTime" in result and np.any(slices >= result["RepetitionTime"]):
        raise ValueError("SliceTiming values must be smaller than RepetitionTime")

    run_stc = len(slices) > 1 and slice_timing_corrected
    result["SliceTimingCorrected"] = run_stc
    if len(slices) > 1:
        ordered = np.sort(slices)
        acquisition_time = float(ordered[-1] + (ordered[1] - ordered[0]))
        if "RepetitionTime" in result:
            repetition_time = result["RepetitionTime"]
            if not np.isclose(repetition_time, acquisition_time) and acquisition_time < repetition_time:
                result["DelayTime"] = repetition_time - acquisition_time
        elif "VolumeTiming" in result:
            result["AcquisitionDuration"] = acquisition_time
        if run_stc:
            result["StartTime"] = float(np.round(
                ordered[0] + reference_fraction * (ordered[-1] - ordered[0]), 3,
            ))
    return result
