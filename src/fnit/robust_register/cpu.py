"""CPU robust registration with source-ordered Float/Double arithmetic.

Normal imports retain process-local module and JIT caches. Each context
serializes its own calls; callers should also serialize first-use JIT across
contexts. See docs/robust_register/README.md for supported inputs and evidence.
"""
from threading import RLock

class CPURegistration:
    """Reusable CPU registration context; closing prevents further calls.

    ``close()`` does not remove modules or process-local Numba caches. This
    context changes neither caller thread settings nor CUDA precision flags.
    """
    def __init__(self):
        self._lock = RLock()
        self._closed = False

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError('CPU registration context is closed')

    def close(self):
        """Close this context; imported modules remain available to others."""
        with self._lock:
            self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def cpu_robust_register(self, source, target, *, device='cpu', **parameters):
        """Register one 3D source to target; return transform, header and report."""
        with self._lock:
            self._ensure_open()
            import torch
            if torch.device(device).type != 'cpu':
                raise ValueError('CPURegistration accepts only device=cpu')
            from ._cpu_engine.registration import robust_register
            return robust_register(source, target, device='cpu', **parameters)

    def cpu_robust_rigid_affine(self, source, target, *, stage_directory, device='cpu', **parameters):
        """Run rigid then affine, saving and reloading the intermediate MGH."""
        with self._lock:
            self._ensure_open()
            import torch
            if torch.device(device).type != 'cpu':
                raise ValueError('CPURegistration accepts only device=cpu')
            from ._cpu_engine.registration import robust_rigid_affine
            return robust_rigid_affine(source, target, stage_directory=stage_directory, device='cpu', **parameters)

def load_cpu_registration():
    """Create a reusable context without importing the registration engine."""
    return CPURegistration()


def FNIT_mri_robust_register(
    source, target, *, mode="rigid", stage_directory=None, device="cpu",
    saturation=50.0, iterations_per_level=5, stop_distance=0.01,
    initialize_translation=True, pyramid_min_size=16, pyramid_max_size=-1,
    highres_iterations=-1, spatial_chunk_size=131072, memory_budget_gb=20.0,
    tf32=True,
):
    """Register 3D images with the mature CPU rigid/affine implementation.

    ``mode="rigid-affine"`` requires a new ``stage_directory`` and retains
    the MGH save/reload boundary between the two stages. Single-stage calls
    return ``RobustRegistrationResult``; the pair returns the existing dict.
    Device, parameter validation, arithmetic and caller state are unchanged.
    """
    if mode not in {"rigid", "affine", "rigid-affine"}:
        raise ValueError("mode must be rigid, affine or rigid-affine")
    if mode == "rigid-affine" and stage_directory is None:
        raise ValueError("stage_directory is required for rigid-affine")
    if mode != "rigid-affine" and stage_directory is not None:
        raise ValueError("stage_directory is only accepted for rigid-affine")
    parameters = dict(
        device=device, saturation=saturation,
        iterations_per_level=iterations_per_level, stop_distance=stop_distance,
        initialize_translation=initialize_translation,
        pyramid_min_size=pyramid_min_size, pyramid_max_size=pyramid_max_size,
        highres_iterations=highres_iterations, spatial_chunk_size=spatial_chunk_size,
        memory_budget_gb=memory_budget_gb, tf32=tf32,
    )
    with load_cpu_registration() as registration:
        if mode == "rigid-affine":
            return registration.cpu_robust_rigid_affine(
                source, target, stage_directory=stage_directory, **parameters)
        return registration.cpu_robust_register(source, target, mode=mode, **parameters)
