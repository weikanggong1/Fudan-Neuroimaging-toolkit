"""Bounded independent L/R tasks; no global thread or CUDA-state changes."""

from concurrent.futures import ThreadPoolExecutor
import operator
import os


def resolve_cpu_threads(cpu_threads=None):
    """Resolve the total CPU budget from an explicit value, OMP or PyTorch."""
    if cpu_threads is None:
        configured = os.environ.get("OMP_NUM_THREADS")
        if configured is not None:
            try:
                cpu_threads = int(configured)
            except ValueError as error:
                raise ValueError("OMP_NUM_THREADS must be a positive integer") from error
        else:
            import torch
            cpu_threads = torch.get_num_threads()
    if isinstance(cpu_threads, bool):
        raise ValueError("cpu_threads must be a positive integer")
    try:
        budget = operator.index(cpu_threads)
    except TypeError as error:
        raise ValueError("cpu_threads must be a positive integer") from error
    if budget < 1:
        raise ValueError("cpu_threads must be a positive integer")
    return budget


def hemisphere_items(*, parallel=True, cpu_threads=None):
    """Return deterministic hemisphere/thread pairs within the total budget."""
    if not isinstance(parallel, bool):
        raise ValueError("parallel must be a boolean")
    budget = resolve_cpu_threads(cpu_threads)
    if parallel and budget > 1:
        return (("L", (budget + 1) // 2), ("R", budget // 2))
    return (("L", budget), ("R", budget))


def map_hemispheres(function, *, parallel=True, cpu_threads=None):
    """Run ``function(hemisphere, threads)`` and return ordered L/R results.

    Both submitted workers are joined before any exception leaves this
    function. Callers can therefore safely release their staging directory.
    A one-thread total budget executes the hemispheres sequentially.
    """
    budget = resolve_cpu_threads(cpu_threads)
    items = hemisphere_items(parallel=parallel, cpu_threads=budget)
    if not parallel or budget == 1:
        return tuple(function(hemisphere, threads) for hemisphere, threads in items)
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="fnit-hemisphere") as executor:
        futures = tuple(executor.submit(function, hemisphere, threads)
                        for hemisphere, threads in items)
        return tuple(future.result() for future in futures)


def workbench_environment(cpu_threads):
    """Give one child process its CPU budget without mutating parent state."""
    budget = resolve_cpu_threads(cpu_threads)
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        environment[name] = str(budget)
    return environment
