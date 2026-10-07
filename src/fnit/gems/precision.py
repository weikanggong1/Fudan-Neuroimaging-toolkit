"""Thread-local precision state used by the optional CUDA region scheduler."""

from __future__ import annotations

from threading import local


_state = local()


def parallel_precision_enabled() -> bool:
    """Return whether the current GEMS worker uses the shared FP32 policy."""
    return bool(getattr(_state, "parallel", False))


def set_parallel_precision(enabled: bool) -> bool:
    """Set worker-local parallel precision mode and return the previous value."""
    previous = parallel_precision_enabled()
    _state.parallel = bool(enabled)
    return previous


def restore_parallel_precision(previous: bool) -> None:
    """Restore a value returned by :func:`set_parallel_precision`."""
    _state.parallel = bool(previous)
