"""Independent hemisphere streams and bounded host search workers.

No library entry point resets the caller's CUDA allocator peak or changes
process-wide CPU thread settings. Streams are joined before output staging
can be released, including when a hemisphere raises an exception.
"""
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from .._hemisphere_parallel import map_hemispheres, resolve_cpu_threads

_CPU_THREADS = ContextVar("fnit_msm_cpu_threads", default=4)
_STATISTICS = ContextVar("fnit_msm_execution_statistics", default=None)


def cpu_workers():
    return _CPU_THREADS.get()


@contextmanager
def execution_budget(cpu_threads):
    token = _CPU_THREADS.set(resolve_cpu_threads(cpu_threads))
    statistics = _STATISTICS.set({"cuda_nearest_queries": 0, "cuda_nearest_proved_queries": 0,
                                  "tree_nearest_queries": 0, "nearest_tie_fallback_queries": 0,
                                  "cuda_adaptive_resampling_calls": 0,
                                  "tree_expanded_face_queries": 0,
                                  "adaptive_cpu_layout_fallback_calls": 0,
                                  "octree_device_containment_queries": 0,
                                  "octree_native_queries": 0})
    try:
        yield
    finally:
        _STATISTICS.reset(statistics)
        _CPU_THREADS.reset(token)



def record_statistics(**values):
    statistics = _STATISTICS.get()
    if statistics is not None:
        for key, value in values.items():
            statistics[key] += int(value)


def current_statistics():
    return dict(_STATISTICS.get() or {})

def register_hemispheres(function, device, *, parallel=True, cpu_threads=None):
    """Return ordered results from ``function(hemi, host_threads)``.

    CUDA tasks use independent streams on the selected device. Each stream
    waits for the caller's current stream and is synchronized on every exit.
    Host tensors/arrays and native graph states remain private to the worker.
    CPU PyTorch operations use its existing shared intra-op pool; the local
    budget bounds native ordered lookup workers and Workbench subprocesses,
    not that pool.
    """
    selected = torch.device(device)
    # Capture an unindexed CUDA device on the caller thread. New host
    # workers otherwise default to their own current device (usually zero).
    if selected.type == "cuda" and selected.index is None:
        selected = torch.device("cuda", torch.cuda.current_device())
    budget = resolve_cpu_threads(cpu_threads)
    origin = None
    if selected.type == "cuda":
        torch.cuda.init()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        origin = torch.cuda.current_stream(selected)
    use_streams = selected.type == "cuda" and parallel and budget > 1

    def run(hemisphere, threads):
        with execution_budget(threads):
            if not use_streams:
                return function(hemisphere, threads)
            stream = torch.cuda.Stream(device=selected)
            stream.wait_stream(origin)
            try:
                with torch.cuda.device(selected), torch.cuda.stream(stream):
                    return function(hemisphere, threads)
            finally:
                stream.synchronize()

    return map_hemispheres(run, parallel=parallel, cpu_threads=budget)


def execution_report(device, *, parallel, cpu_threads):
    selected = torch.device(device)
    budget = resolve_cpu_threads(cpu_threads)
    return {"parallel_requested": parallel, "hemispheres_parallel": bool(parallel and budget > 1),
            "cpu_threads": budget, "cuda_streams": (2 if parallel and budget > 1 else 1) if selected.type == "cuda" else 0,
            "native_graph_execution": "independent ordered CPU HOCR/FastPD",
            "peak_allocated_gb": torch.cuda.max_memory_allocated(selected)/1e9
            if selected.type == "cuda" else None,
            "peak_scope": "caller CUDA allocator since its last external reset; shared by both hemispheres"}
