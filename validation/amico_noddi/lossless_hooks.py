"""Internal A/B hooks for same-input real NODDI validation.

No data are generated here. Use separate runs for timing and for detailed
comparisons; default production calls use both optimizations. This context
changes Python callables in this process and must not overlap concurrent fits.
"""

from contextlib import contextmanager
from functools import partial

from fnit.amico_noddi import core, solver


EQUALITY_QC_FIELDS = (
    "device", "kernel_dtype", "solver_dtype", "tf32", "fit_method",
    "dictionary_atoms", "lut_directions", "voxels", "volumes",
    "lut_directions_used", "lut_batch_size", "support_size_min",
    "support_size_median", "support_size_max", "accepted_updates",
    "rician_sigma_median",
)


@contextmanager
def noddi_work_reuse(*, reuse_gram=True, defer_classic_qc=True):
    """Temporarily select internal work reuse for a same-input A/B run.

    ``reuse_gram=True`` reuses only the full dictionary Gram; linear is
    recomputed in each NNLS stage. ``False`` recomputes both products each time.
    ``defer_classic_qc=False`` reads the exact acceptance count every iteration.
    These flags never change float64 fitting, LUT batches or solver decisions.
    """
    original_lut_batch = solver._fit_lut_batch
    original_classic = core.fit_classic_noddi
    solver._fit_lut_batch = partial(original_lut_batch, _reuse_gram=reuse_gram)
    core.fit_classic_noddi = partial(original_classic, _defer_qc_count=defer_classic_qc)
    try:
        yield
    finally:
        solver._fit_lut_batch = original_lut_batch
        core.fit_classic_noddi = original_classic


def qc_differences(baseline, candidate):
    """Return changed model/result QC; elapsed and allocator peaks are excluded."""
    return {
        name: {"baseline": baseline.get(name), "candidate": candidate.get(name)}
        for name in EQUALITY_QC_FIELDS
        if baseline.get(name) != candidate.get(name)
    }
