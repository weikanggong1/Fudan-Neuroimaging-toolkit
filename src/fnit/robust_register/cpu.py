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
